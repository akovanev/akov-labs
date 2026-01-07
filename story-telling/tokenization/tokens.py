import tiktoken
import numpy as np
from multiprocessing import Pool
import os

def tokenize_chunk(args):
    chunk, encoding_name = args
    enc = tiktoken.get_encoding(encoding_name)
    return enc.encode_ordinary(chunk) # use encode_ordinary to ignore special tokens

def parallel_tokenize(file_path, encoding_name="cl100k_base", n_workers=8, chunk_size=10**6):

    chunks = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            chunks.append((chunk, encoding_name))
    
    print(f"Tokenizing {len(chunks)} chunks...")
    with Pool(n_workers) as pool:
        all_tokens = []
        for tokens in pool.imap(tokenize_chunk, chunks):
            all_tokens.extend(tokens)
    
    return np.array(all_tokens, dtype=np.uint32)

if __name__ == "__main__":
    # Update paths as needed
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    input_file = os.path.join(data_dir, 'train.csv')
    output_bin = os.path.join(data_dir, 'train.bin')
    
    tokens = parallel_tokenize(input_file)
    tokens.tofile(output_bin)
    print(f"Saved {len(tokens)} tokens to {output_bin}")