import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.dataset import CSLTDataset
from training.tokenizer import build_vocab
from models.cslt_model import CSLTModel


# ---------------------------------------------------------
# Word Error Rate
# ---------------------------------------------------------

def wer(ref, hyp):

    ref_words = ref.split()
    hyp_words = hyp.split()

    d = np.zeros((len(ref_words)+1, len(hyp_words)+1))

    for i in range(len(ref_words)+1):
        d[i][0] = i

    for j in range(len(hyp_words)+1):
        d[0][j] = j

    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )

    return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)


# ---------------------------------------------------------
# CTC Decoder
# ---------------------------------------------------------

def decode(log_probs, idx_to_word):

    preds = torch.argmax(log_probs, dim=2)

    sentences = []

    for seq in preds:

        prev  = -1
        words = []

        for p in seq:
            p = p.item()
            if p != prev and p != 0:
                words.append(idx_to_word.get(p, ""))
            prev = p

        sentences.append(" ".join(words))

    return sentences


# ---------------------------------------------------------
# Scheduler
# ---------------------------------------------------------

def get_scheduler(optimizer, warmup_epochs=3, total_epochs=30):

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(0.1, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, scaler, device, word_to_idx):

    model.train()

    total_loss = 0
    batches    = 0

    for x, text, actual_lens in tqdm(loader):

        x           = x.to(device)
        actual_lens = actual_lens.to(device)

        targets        = []
        target_lengths = []

        for sentence in text:
            tokens = [word_to_idx[w] for w in sentence.split() if w in word_to_idx]
            if len(tokens) == 0:
                continue
            targets.extend(tokens)
            target_lengths.append(len(tokens))

        if len(target_lengths) == 0:
            continue

        targets        = torch.tensor(targets,        dtype=torch.long).to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type):

            output    = model(x)
            log_probs = output.log_softmax(dim=2)

            input_lengths = actual_lens.clamp(min=1)

            if len(target_lengths) != len(input_lengths):
                continue

            if torch.any(target_lengths > input_lengths):
                continue

            loss = criterion(
                log_probs.permute(1, 0, 2),
                targets,
                input_lengths,
                target_lengths
            )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batches    += 1

    return total_loss / max(batches, 1)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate(model, loader, idx_to_word, device):

    model.eval()

    total_wer = 0
    count     = 0

    with torch.no_grad():

        for x, text, actual_lens in loader:

            x = x.to(device)

            output    = model(x)
            log_probs = output.log_softmax(dim=2)
            preds     = decode(log_probs, idx_to_word)

            for pred, gt in zip(preds, text):

                if count < 5:
                    print("\nGT :", gt)
                    print("PR :", pred)
                    print("------")

                total_wer += wer(gt, pred)
                count     += 1

    return total_wer / max(count, 1)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vocab, word_to_idx, idx_to_word = build_vocab()
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    dataset    = CSLTDataset(augment=True)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=16,
        num_workers=6
    )

    print("Train samples     :", len(train_set))
    print("Validation samples:", len(val_set))

    model = CSLTModel(vocab_size).to(device)

    # -------------------------------------------------
    # Load pretrained weights (strict=False skips
    # mismatched layers like embedding/gnn)
    # -------------------------------------------------

    pretrained = "best_cslt_model.pth"

    if os.path.exists(pretrained):

        state = torch.load(pretrained, map_location=device, weights_only=True)

        # filter out layers with shape mismatch
        filtered = {}
        for k, v in state.items():
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                filtered[k] = v
            else:
                print(f"  Skipping mismatched layer: {k}")
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        print(f"\nLoaded pretrained weights from {pretrained}")
        print(f"  Layers loaded    : {len(filtered)}")
        print(f"  New layers (GNN) : {len(missing)}")

    else:
        print("\nNo pretrained weights found — training from scratch")

    # -------------------------------------------------
    # Phase 1: freeze transformer, train only GNN
    # + embedding for 10 epochs
    # -------------------------------------------------

    print("\n--- Phase 1: Training GNN + Embedding (10 epochs) ---")

    for name, param in model.named_parameters():
        if any(x in name for x in ["transformer", "temporal", "pos", "classifier"]):
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler    = GradScaler(device.type)
    scheduler = get_scheduler(optimizer, warmup_epochs=2, total_epochs=10)
    best_wer  = 999

    for epoch in range(10):

        train_loss = train_epoch(
            model, train_loader, optimizer,
            criterion, scaler, device, word_to_idx
        )

        val_wer = validate(model, val_loader, idx_to_word, device)

        scheduler.step()

        print(f"\nPhase1 Epoch: {epoch+1}  Loss: {round(train_loss,4)}  WER: {round(val_wer,4)}")

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(model.state_dict(), "best_cslt_gnn.pth")
            print("New best saved")

    # -------------------------------------------------
    # Phase 2: unfreeze all, fine-tune everything
    # -------------------------------------------------

    print("\n--- Phase 2: Fine-tuning all layers (40 epochs) ---")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = get_scheduler(optimizer, warmup_epochs=3, total_epochs=40)

    for epoch in range(40):

        train_loss = train_epoch(
            model, train_loader, optimizer,
            criterion, scaler, device, word_to_idx
        )

        val_wer = validate(model, val_loader, idx_to_word, device)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nPhase2 Epoch: {epoch+1}  LR: {round(current_lr,6)}  "
              f"Loss: {round(train_loss,4)}  WER: {round(val_wer,4)}")

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(model.state_dict(), "best_cslt_gnn.pth")
            print("New best saved")

    print(f"\nTraining finished. Best WER: {round(best_wer, 4)}")


if __name__ == "__main__":
    main()