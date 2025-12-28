import numpy as np
import torch
from collections import Counter

def clean_line(line):
    """Standardizing the cleaning logic used in both training and inference."""
    return (line.lower()
            .replace(".", " . ")
            .replace(",", " , ")
            .replace("!", " ! ")
            .replace("?", " ? ")
            .split())

def run_gpt_prep(input_path, bin_path, vocab_path):
    print("Phase 1: Building Vocabulary (Streaming)...")
    vocab_counter = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab_counter.update(clean_line(line))
            if i % 500000 == 0 and i > 0: print(f"  Processed {i} lines...")
    
    word_to_id = {word: i for i, (word, count) in enumerate(vocab_counter.most_common())}
    word_to_id["<UNK>"] = len(word_to_id)
    
    vocab_size = len(word_to_id)
    # Use uint32 if vocab is huge, else uint16 saves 50% disk space
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    torch.save(word_to_id, vocab_path)
    
    print(f"Phase 2: Encoding to Binary (dtype={dtype})...")
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(bin_path, 'wb') as f_out:
        for i, line in enumerate(f_in):
            ids = [word_to_id.get(w, word_to_id["<UNK>"]) for w in clean_line(line)]
            if ids:
                f_out.write(np.array(ids, dtype=dtype).tobytes())
            if i % 500000 == 0 and i > 0: print(f"  Encoded {i} lines...")

    return vocab_size, dtype