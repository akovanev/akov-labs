import main as m
import os
import torch
import numpy as np
import tiktoken

PROMPT = "The cat looked towards"
TOKENS_TO_GENERATE = 100

def main():
    # 1. Setup Tokenizer
    # We use cl100k_base to match the training setup discussed
    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab 
    print(f"Tokenizer loaded. Vocab Size: {vocab_size}")

    # 2. Load Model
    # Important: vocab_size must match what was used during training
    model = m.NanoStoryGPTModel(
        vocab_size, 
        m.N_EMBD, 
        m.N_HEAD, 
        m.N_LAYER, 
        m.BLOCK_SIZE, 
        m.DROPOUT
    ).to(m.device)

    if not os.path.exists(m.model_path):
        print(f"FAILED: {m.model_path} not found! Did you run training with tiktoken?")
        return
        
    model.load_state_dict(torch.load(m.model_path, map_location=m.device))
    model.eval()
    print("Model weights loaded.")

    # 3. Process Prompt using tiktoken
    # No manual lower() or split() needed; tiktoken handles the raw string
    ids = enc.encode(PROMPT)
    print(f"Input IDs the model sees: {ids}")
    
    x = torch.tensor([ids], dtype=torch.long, device=m.device)

    # 4. Generate
    print(f"Generating {TOKENS_TO_GENERATE} tokens...", flush=True)
    
    with torch.no_grad():
        # model.generate returns the prompt + new tokens
        out_ids = model.generate(x, TOKENS_TO_GENERATE)[0].tolist()
    
    # 5. Decode using tiktoken
    # tiktoken's decode is "clean" by default—it knows where spaces belong
    result = enc.decode(out_ids)
    
    print("\n--- FINAL RESULT ---")
    print(result)

if __name__ == "__main__":
    main()