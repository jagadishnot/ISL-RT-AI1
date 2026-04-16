import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.dataset import CSLTDataset
from training.tokenizer import build_vocab
from models.cslt_model import CSLTModel

# ---------------------------------------------------------
# WER
# ---------------------------------------------------------

def wer(ref, hyp):

    r = ref.split()
    h = hyp.split()

    d = np.zeros((len(r)+1, len(h)+1))

    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)

    return d[len(r)][len(h)] / max(len(r), 1)


# ---------------------------------------------------------
# CTC DECODER
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
# MAIN
# ---------------------------------------------------------

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vocab, word_to_idx, idx_to_word = build_vocab()

    model = CSLTModel(len(vocab))
    model.load_state_dict(
        torch.load("best_cslt_gnn.pth", map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    dataset    = CSLTDataset(augment=False)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size

    _, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_set, batch_size=16, num_workers=4)

    print(f"Evaluating on {len(val_set)} validation samples...\n")

    # per-label tracking
    label_correct = defaultdict(int)
    label_total   = defaultdict(int)
    label_wer     = defaultdict(list)

    all_results   = []
    total_wer     = 0
    exact_matches = 0
    count         = 0

    with torch.no_grad():

        for x, text, actual_lens in val_loader:

            x = x.to(device)

            output    = model(x)
            log_probs = output.log_softmax(dim=2)
            preds     = decode(log_probs, idx_to_word)

            for pred, gt in zip(preds, text):

                w = wer(gt, pred)
                total_wer += w
                count     += 1

                label_total[gt] += 1
                label_wer[gt].append(w)

                if pred.strip() == gt.strip():
                    exact_matches       += 1
                    label_correct[gt]   += 1

                all_results.append((gt, pred, w))

    # ---------------------------------------------------------
    # OVERALL STATS
    # ---------------------------------------------------------

    avg_wer    = total_wer / max(count, 1)
    word_acc   = 1 - avg_wer
    exact_acc  = exact_matches / max(count, 1)

    print("=" * 55)
    print("OVERALL RESULTS")
    print("=" * 55)
    print(f"Samples evaluated : {count}")
    print(f"Average WER       : {avg_wer:.4f}")
    print(f"Word Accuracy     : {word_acc*100:.1f}%")
    print(f"Exact Match Acc   : {exact_acc*100:.1f}%")
    print()

    # ---------------------------------------------------------
    # PER LABEL ACCURACY
    # ---------------------------------------------------------

    print("=" * 55)
    print("PER LABEL ACCURACY")
    print("=" * 55)

    per_label = []
    for label in sorted(label_total.keys()):
        total   = label_total[label]
        correct = label_correct[label]
        acc     = correct / total * 100
        avg_w   = np.mean(label_wer[label])
        per_label.append((label, correct, total, acc, avg_w))

    # sort by accuracy ascending (worst first)
    per_label.sort(key=lambda x: x[3])

    print("\n--- Worst performing (fix these first) ---")
    for label, correct, total, acc, w in per_label[:15]:
        bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
        print(f"  {label:35s} {bar} {acc:5.1f}%  ({correct}/{total})")

    print("\n--- Best performing ---")
    for label, correct, total, acc, w in per_label[-15:]:
        bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
        print(f"  {label:35s} {bar} {acc:5.1f}%  ({correct}/{total})")

    # ---------------------------------------------------------
    # WRONG PREDICTIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 55)
    print("WRONG PREDICTIONS (sample)")
    print("=" * 55)

    wrong = [(gt, pred, w) for gt, pred, w in all_results if pred.strip() != gt.strip()]
    wrong.sort(key=lambda x: -x[2])

    for gt, pred, w in wrong[:20]:
        print(f"  GT : {gt}")
        print(f"  PR : {pred}")
        print()


if __name__ == "__main__":
    main()