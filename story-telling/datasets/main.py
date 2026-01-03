import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

data_dir = os.path.join(os.path.dirname(__file__), 'data')
input_path = os.path.join(data_dir, 'train.csv')
bin_path = os.path.join(data_dir, 'train.bin')
vocab_path = os.path.join(data_dir, 'vocab.pt')

def clean_line(line):
    """Standardizing the cleaning logic used in both training and inference."""
    return (line.lower()
            .replace(".", " . ")
            .replace(",", " , ")
            .replace("!", " ! ")
            .replace("?", " ? ")
            .split())

def build_vocab(input_path, bin_path, vocab_path):
    print("Phase 1: Building Vocabulary (Streaming)...")
    vocab_counter = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab_counter.update(clean_line(line))
            if i % 500000 == 0 and i > 0: print(f"  Processed {i} lines...")
    
    word_to_id = {word: i for i, (word, count) in enumerate(vocab_counter.most_common())}
    print(f"Word to ID mapping sample: {list(word_to_id.items())[:10]}")  # DEBUG PRINT
    word_to_id["<UNK>"] = len(word_to_id)
    
    vocab_size = len(word_to_id)
    # Use uint32 if vocab is huge, else uint16 saves 50% disk space
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    torch.save(word_to_id, vocab_path)
    return word_to_id, dtype

def encode_data(input_path, bin_path, word_to_id, dtype):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(bin_path, 'wb') as f_out:
        for i, line in enumerate(f_in):
            ids = [word_to_id.get(w, word_to_id["<UNK>"]) for w in clean_line(line)]
            if ids:
                f_out.write(np.array(ids, dtype=dtype).tobytes())
            if i % 500000 == 0 and i > 0: print(f"  Encoded {i} lines...")

word_to_id, dtype = build_vocab(input_path, bin_path, vocab_path)
encode_data(input_path, bin_path, word_to_id, dtype)

BLOCK_SIZE = 128 # context length
BATCH_SIZE = 32  # can be adjusted based based on system RAM

class GPTDataset(Dataset):
    def __init__(self, bin_path, dtype, block_size=BLOCK_SIZE):
        self.data = np.memmap(bin_path, dtype=dtype, mode='r')
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size - 1
    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx : idx + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]

if __name__ == "__main__":
    train_ds = GPTDataset(bin_path, np.uint32, BLOCK_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    for x, y in train_loader:
        print(f"Batch x shape: {x.shape}, y shape: {y.shape}")  # DEBUG PRINT
        print(f"Batch x sample: {x[0][:10]}")  # DEBUG PRINT
        print(f"Batch y sample: {y[0][:10]}")  # DEBUG PRINT
        break  # Just to verify one batch

    print("Data loading and batching verified.")