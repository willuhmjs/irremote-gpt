import torch
from model import GPT
import json
import os
import sys
import re
from prep_data import normalize_button
from tokenizer import Tokenizer

# Check for python-Levenshtein
try:
    import Levenshtein
except ImportError:
    print("Please install python-Levenshtein: pip install python-Levenshtein")
    sys.exit(1)

# Hyperparameters (must match training)
n_embd = 384
n_head = 6
n_layer = 6
block_size = 1024
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test Data Directory
TEST_DATA_DIR = "test_data"

def parse_test_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return [], None, None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    buttons = []
    protocol = None
    address = None
    
    # Simple parsing logic similar to prep_data.py but specific for this test file structure
    sections = content.split('#')
    for section in sections:
        lines = section.strip().split('\n')
        name = None
        cmd = None
        proto = None
        addr = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip()
            elif line.startswith('command:'):
                cmd = line.split(':', 1)[1].strip().replace(' ', '')
            elif line.startswith('protocol:'):
                proto = line.split(':', 1)[1].strip()
            elif line.startswith('address:'):
                addr = line.split(':', 1)[1].strip().replace(' ', '')
                
        # Capture global protocol/address from first button that has them
        if proto and not protocol: protocol = proto
        if addr and not address: address = addr
        
        if name and cmd:
            token = normalize_button(name)
            if token:
                buttons.append((token, cmd))
            else:
                 # Fallback for some common names not in prep_data default
                 pass
                 
    return buttons, protocol, address

def test_file(model, tok, filepath):
    print(f"\nTesting file: {filepath}")
    buttons, protocol, address = parse_test_data(filepath)
    
    if not buttons:
        print("No valid buttons found in test data.")
        return
    
    if not protocol or not address:
        print("Protocol or Address not found in test data.")
        return
        
    print(f"Loaded {len(buttons)} valid buttons for testing: {[b[0] for b in buttons]}")
    print(f"Protocol: {protocol}, Address: {address}")
    print("-" * 80)
    print(f"{'Progress':<10} | {'Context Last':<20} | {'Pred Next':<20} | {'Actual Next':<20} | {'Code Dist':<10}")
    print("-" * 80)

    # Testing Loop
    total_dist = 0
    correct_token_count = 0
    steps = 0

    for i in range(1, len(buttons)):
        context_buttons = buttons[:i]
        target_token, target_code = buttons[i]
        
        # Construct Prompt
        prompt_str = f"<BOS> PROTOCOL:{protocol} ADDRESS:{address}"
        for t, c in context_buttons:
            prompt_str += f" {t}:{c}"
        prompt_str += " "
        
        # Encode
        input_ids = tok.encode(prompt_str)
        x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(x, max_new_tokens=50, top_k=50)
            
        generated_text = tok.decode(output_ids[0].tolist())
        new_text = generated_text[len(prompt_str):]
        
        # Parse
        match = re.search(r'(\[BTN_[^\]]+\]):([0-9A-Fa-f]+)', new_text)
        
        if match:
            pred_token = match.group(1)
            pred_code = match.group(2)
            
            # Since target_code is from file (string), and pred_code is from generation (string),
            # and our tokenizer might handle them differently in terms of case?
            # Usually hex is upper, let's normalize.
            
            dist = Levenshtein.distance(pred_code.upper(), target_code.upper())
            total_dist += dist
            
            if pred_token == target_token:
                correct_token_count += 1
            
            last_ctx = context_buttons[-1][0]
            progress_pct = int(i / len(buttons) * 100)
            print(f"{progress_pct}%".ljust(10) + f" | {last_ctx:<20} | {pred_token:<20} | {target_token:<20} | {dist:<10}")
        else:
            progress_pct = int(i / len(buttons) * 100)
            print(f"{progress_pct}%".ljust(10) + f" | {context_buttons[-1][0]:<20} | {'(Failed)':<20} | {target_token:<20} | {'-':<10}")
        
        steps += 1
        
    if steps > 0:
        print("-" * 80)
        print(f"Token Accuracy: {correct_token_count}/{steps} ({correct_token_count/steps*100:.1f}%)")
        print(f"Avg Code Levenshtein Dist: {total_dist/steps:.2f}")

def main():
    # Load Model & Vocab
    tok = Tokenizer()
    if not tok.load_vocab():
        sys.exit(1)
        
    vocab_size = len(tok.stoi)
    
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    if os.path.exists('model.pt'):
        model.load_state_dict(torch.load('model.pt', map_location=device))
    else:
        print("model.pt not found. Train first.")
        sys.exit(1)
    
    model.to(device)
    model.eval()

    # Find test files
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Test data directory '{TEST_DATA_DIR}' not found.")
        return

    test_files = [os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR) if f.endswith('.ir')]
    
    if not test_files:
        print(f"No .ir files found in {TEST_DATA_DIR}")
        return

    for test_file_path in sorted(test_files):
        test_file(model, tok, test_file_path)

if __name__ == "__main__":
    main()
