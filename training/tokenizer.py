import pandas as pd
import os
import re


# ---------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------

def clean_text(text):

    text = str(text).lower().strip()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # normalize spaces
    text = " ".join(text.split())

    return text


# ---------------------------------------------------------
# Build Vocabulary
# ---------------------------------------------------------

def build_vocab():

    base_dir = os.path.dirname(os.path.dirname(__file__))

    labels_path = os.path.join(base_dir, "data", "labels.csv")

    df = pd.read_csv(labels_path)

    vocab = set()

    for text in df["text"]:

        text = clean_text(text)

        for word in text.split():
            vocab.add(word)

    vocab = sorted(list(vocab))

    # reserve tokens
    word_to_idx = {
        "<blank>": 0,
        "<unk>": 1
    }

    for i, word in enumerate(vocab, start=2):
        word_to_idx[word] = i

    idx_to_word = {i: w for w, i in word_to_idx.items()}

    vocab_list = list(word_to_idx.keys())

    print("\nVocabulary built")
    print("Vocabulary size:", len(vocab_list))

    return vocab_list, word_to_idx, idx_to_word


# ---------------------------------------------------------
# Encode Sentence
# ---------------------------------------------------------

def encode_sentence(sentence, word_to_idx):

    sentence = clean_text(sentence)

    tokens = []

    for word in sentence.split():

        if word in word_to_idx:
            tokens.append(word_to_idx[word])
        else:
            tokens.append(word_to_idx["<unk>"])

    return tokens


# ---------------------------------------------------------
# Decode Tokens
# ---------------------------------------------------------

def decode_tokens(tokens, idx_to_word):

    words = []

    for t in tokens:

        # FIX: also filter out <unk> from decoded output
        if t in idx_to_word and idx_to_word[t] not in ["<blank>", "<unk>"]:
            words.append(idx_to_word[t])

    return " ".join(words)