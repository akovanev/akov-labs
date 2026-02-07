import main as m
import os
import re
import torch

PROMPT = "the cat looked towards"
TOKENS_TO_GENERATE = 100

def clean_punctuation(text):
    # 1. Fix spaces around standard punctuation
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    
    # 2. Fix quotes: Remove space after opening quote or before closing quote
    # This logic assumes a quote followed by a word is starting, 
    # and a quote preceded by punctuation/word is ending.
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    
    # 3. Fix internal double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
     # 1. Load Vocab
    if not os.path.exists(m.vocab_path):
        print(f"FAILED: {m.vocab_path} not found!")
        return
    word_to_id = torch.load(m.vocab_path)
    id_to_word = {i: w for w, i in word_to_id.items()}
    print(f"Vocab loaded. Size: {len(word_to_id)}")

    # 2. Load Model
    model = m.NanoStoryGPTModel(len(word_to_id), m.N_EMBD, m.N_HEAD, m.N_LAYER, m.BLOCK_SIZE, m.DROPOUT).to(m.device)
    if not os.path.exists(m.model_path):
        print(f"FAILED: {m.model_path} not found! Did you run train.py?")
        return
    model.load_state_dict(torch.load(m.model_path, map_location=m.device))
    model.eval()
    print("Model weights loaded.")

    # 3. Process Prompt
    tokens = PROMPT.lower().replace(".", " . ").replace(",", " , ").split()
    ids = [word_to_id.get(w, word_to_id.get("<UNK>", 0)) for w in tokens]
    
    print(f"Input IDs the model sees: {ids}") # DEBUG PRINT
    
    x = torch.tensor([ids], dtype=torch.long, device=m.device)

    # 4. Generate
    print(f"Generating {TOKENS_TO_GENERATE} tokens...", flush=True)
    
    with torch.no_grad():
        out_ids = model.generate(x, TOKENS_TO_GENERATE)[0].tolist()
    
    # 5. Decode
    words = []
    for i in out_ids:
        word = id_to_word.get(i, "<UNK>")
        words.append(word)
    
    result = clean_punctuation(" ".join(words))
    
    print("\n--- FINAL RESULT ---")
    print(result)

if __name__ == "__main__":
    main()