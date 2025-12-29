import os
import sys
import re
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import time
import random
# Try importing Levenshtein for testing, but don't fail if missing until needed
try:
    import Levenshtein
except ImportError:
    Levenshtein = None

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Model Hyperparameters
    n_embd = 384
    n_head = 6
    n_layer = 6
    block_size = 1024
    dropout = 0.1
    
    # Training Hyperparameters
    batch_size = 16
    max_iters = 2000
    eval_interval = 200
    learning_rate = 3e-4
    eval_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    train_data_path = 'ir_corpus.txt'
    vocab_path = 'vocab.json'
    model_path = 'model.pt'
    test_data_dir = 'test_data'
    irdb_dirs = ['IRDB', 'databases']
    
    # Generation
    temperature = 0.8
    top_k = 40

# ==========================================
# TOKENIZER
# ==========================================
class Tokenizer:
    def __init__(self, vocab_path='vocab.json'):
        self.vocab_path = vocab_path
        self.stoi = {}
        self.itos = {}
        # Special tokens: BOS, EOS, Protocol/Address markers, and Button Tokens
        self.special_token_pattern = r'(<BOS>|<EOS>|PROTOCOL:|ADDRESS:|\[BTN_[^\]]+\])'

    def build_vocab(self, text):
        parts = self._split_text(text)
        vocab = set()
        for part in parts:
            if self._is_special_token(part):
                vocab.add(part)
            else:
                clean_part = part.strip()
                if not clean_part: continue
                # Tokenize hex pairs
                if re.match(r'^[0-9A-Fa-f]+$', clean_part) and len(clean_part) % 2 == 0:
                    for i in range(0, len(clean_part), 2):
                        vocab.add(clean_part[i:i+2].upper())
                else:
                    # Fallback chars
                    for char in part:
                         if not char.isspace(): vocab.add(char)
        
        # Ensure standard hex bytes exist
        for i in range(256):
            vocab.add(f"{i:02X}")
            
        vocab = sorted(list(vocab))
        self.stoi = { token:i for i,token in enumerate(vocab) }
        self.itos = { i:token for i,token in enumerate(vocab) }
        return len(vocab)

    def save_vocab(self):
        with open(self.vocab_path, 'w') as f:
            json.dump(self.stoi, f, indent=2)

    def load_vocab(self):
        if not os.path.exists(self.vocab_path):
            print(f"Warning: {self.vocab_path} not found.")
            return False
        with open(self.vocab_path, 'r') as f:
            self.stoi = json.load(f)
        self.itos = {int(i): token for token, i in self.stoi.items()}
        return True

    def encode(self, text):
        parts = self._split_text(text)
        encoded = []
        for part in parts:
            if not part: continue
            if self._is_special_token(part):
                if part in self.stoi: encoded.append(self.stoi[part])
            else:
                clean_part = part.replace(" ", "").replace("\n", "")
                if not clean_part: continue
                if re.match(r'^[0-9A-Fa-f]+$', clean_part) and len(clean_part) % 2 == 0:
                    for i in range(0, len(clean_part), 2):
                        byte = clean_part[i:i+2].upper()
                        if byte in self.stoi: encoded.append(self.stoi[byte])
                        else:
                            for c in byte: 
                                if c in self.stoi: encoded.append(self.stoi[c])
                else:
                    for char in part:
                        if not char.isspace() and char in self.stoi:
                            encoded.append(self.stoi[char])
        return encoded

    def decode(self, ids):
        return "".join([self.itos.get(i, '') for i in ids])

    def _split_text(self, text):
        parts = re.split(self.special_token_pattern, text)
        return [p for p in parts if p]

    def _is_special_token(self, token):
        return re.match(self.special_token_pattern, token) is not None

# ==========================================
# MODEL (GPT)
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(Config.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.block_size = Config.block_size
        self.token_embedding_table = nn.Embedding(vocab_size, Config.n_embd)
        self.position_embedding_table = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[Block(Config.n_embd, Config.n_head) for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = self.block_size
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# PREP DATA
# ==========================================
def normalize_button(name):
    n = name.strip().lower()
    mappings = {
        'power': '[BTN_POWER]', 'pwr': '[BTN_POWER]', 'power button': '[BTN_POWER]', 'on/off': '[BTN_POWER]',
        'vol+': '[BTN_VOL_UP]', 'volume up': '[BTN_VOL_UP]', 'vol_up': '[BTN_VOL_UP]',
        'vol-': '[BTN_VOL_DOWN]', 'volume down': '[BTN_VOL_DOWN]', 'vol_dn': '[BTN_VOL_DOWN]',
        'mute': '[BTN_MUTE]',
        'ch+': '[BTN_CH_UP]', 'channel up': '[BTN_CH_UP]', 'ch up': '[BTN_CH_UP]',
        'ch-': '[BTN_CH_DOWN]', 'channel down': '[BTN_CH_DOWN]', 'ch down': '[BTN_CH_DOWN]',
        'menu': '[BTN_MENU]', 'back': '[BTN_BACK]', 'exit': '[BTN_EXIT]', 'info': '[BTN_INFO]',
        'up': '[BTN_UP]', 'down': '[BTN_DOWN]', 'left': '[BTN_LEFT]', 'right': '[BTN_RIGHT]', 'ok': '[BTN_OK]',
        'enter': '[BTN_ENTER]', 'select': '[BTN_SELECT]',
        'play': '[BTN_PLAY]', 'pause': '[BTN_PAUSE]', 'stop': '[BTN_STOP]',
        'rewind': '[BTN_REWIND]', 'fast forward': '[BTN_FAST_FORWARD]', 'next': '[BTN_NEXT]', 'prev': '[BTN_PREV]',
        'red': '[BTN_RED]', 'green': '[BTN_GREEN]', 'blue': '[BTN_BLUE]', 'yellow': '[BTN_YELLOW]',
        'input': '[BTN_INPUT]', 'source': '[BTN_SOURCE]', 'av': '[BTN_AV]', 'hdmi': '[BTN_HDMI]',
    }
    if n in mappings: return mappings[n]
    if n.isdigit() and len(n) == 1: return f'[BTN_{n}]'
    return None

def process_file_content(content):
    sections = content.split('#')
    buttons = []
    protocol = None
    address = None
    for section in sections:
        lines = section.strip().split('\n')
        btn_name = btn_proto = btn_addr = btn_cmd = None
        for line in lines:
            line = line.strip()
            if line.startswith('name:'): btn_name = line.split(':', 1)[1].strip()
            elif line.startswith('protocol:'): btn_proto = line.split(':', 1)[1].strip()
            elif line.startswith('address:'): btn_addr = line.split(':', 1)[1].strip().replace(' ', '')
            elif line.startswith('command:'): btn_cmd = line.split(':', 1)[1].strip().replace(' ', '')
        
        if btn_name and btn_proto and btn_addr and btn_cmd:
            token = normalize_button(btn_name)
            if token:
                if not protocol: protocol = btn_proto
                if not address: address = btn_addr
                buttons.append((token, btn_cmd))
    
    if buttons and protocol and address:
        cmds = " ".join([f"{t}:{c}" for t, c in buttons])
        return f"<BOS> PROTOCOL:{protocol} ADDRESS:{address} {cmds} <EOS>"
    return None

def run_prep_data():
    out_lines = []
    
    for d in Config.irdb_dirs:
        if not os.path.exists(d):
            print(f"Directory {d} not found. Skipping.")
            continue
            
        print(f"Scanning {d}...")
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith('.ir'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            res = process_file_content(f.read())
                        if res:
                            out_lines.append(res)
                    except Exception as e: print(f"Error processing {file}: {e}")
    
    count = len(out_lines)
    full_text = "\n".join(out_lines)
    with open(Config.train_data_path, 'w', encoding='utf-8') as f: f.write(full_text)
    print(f"Processed {count} remotes. Saved to {Config.train_data_path}.")
    
    print("Building vocabulary...")
    tok = Tokenizer(Config.vocab_path)
    sz = tok.build_vocab(full_text)
    tok.save_vocab()
    print(f"Vocab size: {sz}")

# ==========================================
# TRAIN
# ==========================================
def get_batch(data, split, device):
    ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i+Config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Config.block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(Config.eval_iters)
        for k in range(Config.eval_iters):
            X, Y = get_batch(data, split, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def run_train():
    print(f"Device: {Config.device}")
    if not os.path.exists(Config.train_data_path):
        print("Data not found. Run prep first.")
        return

    with open(Config.train_data_path, 'r', encoding='utf-8') as f: text = f.read()
    
    tok = Tokenizer(Config.vocab_path)
    if os.path.exists(Config.vocab_path): tok.load_vocab()
    else: tok.build_vocab(text); tok.save_vocab()
    
    vocab_size = len(tok.stoi)
    print(f"Vocab Size: {vocab_size}")
    
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    model = GPT(vocab_size)
    model.to(Config.device)
    
    if os.path.exists(Config.model_path):
        try: model.load_state_dict(torch.load(Config.model_path, map_location=Config.device)); print("Loaded model.")
        except: print("Starting from scratch.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    best_loss = float('inf')
    
    print("Starting training...")
    for iter in range(Config.max_iters):
        if iter % Config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, Config.device)
            print(f"Step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            if losses['val'] < best_loss:
                best_loss = losses['val']
                torch.save(model.state_dict(), Config.model_path)
        
        xb, yb = get_batch(train_data, 'train', Config.device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print(f"Training Done. Best Val Loss: {best_loss:.4f}")

# ==========================================
# GENERATE
# ==========================================
def format_hex(hex_str):
    clean = hex_str.replace(" ", "")
    if len(clean) % 2 != 0: clean = "0" + clean
    return " ".join(clean[i:i+2] for i in range(0, len(clean), 2))

def run_generate(protocol, address, known_btn, known_code, output_file):
    tok = Tokenizer(Config.vocab_path)
    if not tok.load_vocab(): return
    
    if not os.path.exists(Config.model_path):
        print("Model not found.")
        return
        
    model = GPT(len(tok.stoi))
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()
    
    btn_token = f"[BTN_{known_btn.upper()}]" if not known_btn.upper().startswith("[BTN_") else known_btn.upper()
    prompt = f"<BOS> PROTOCOL:{protocol} ADDRESS:{address.replace(' ','')} {btn_token}:{known_code.replace(' ','')}"
    print(f"Prompt: {prompt}")
    
    ids = tok.encode(prompt)
    if not ids: return
    x = torch.tensor(ids, dtype=torch.long, device=Config.device)[None, ...]
    
    with torch.no_grad():
        out_ids = model.generate(x, 500, temperature=Config.temperature, top_k=Config.top_k)
    
    generated = tok.decode(out_ids[0].tolist())
    if "<EOS>" in generated: generated = generated.split("<EOS>")[0]
    
    matches = re.findall(r'(\[BTN_[^\]]+\]):([0-9A-Fa-f]+)', generated)
    if not matches: print("No buttons generated."); return
    
    with open(output_file, 'w') as f:
        f.write(f"Filetype: IR signals file\nVersion: 1\n# Generated by IRRemoteGPT\n")
        for btn, code in matches:
            clean = btn.replace("[BTN_", "").replace("]", "")
            f.write(f"name: {clean}\ntype: parsed\nprotocol: {protocol}\naddress: {format_hex(address)}\ncommand: {format_hex(code)}\n#\n")
    print(f"Saved to {output_file}")

# ==========================================
# TEST
# ==========================================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run_test():
    if not Levenshtein: 
        print(f"{Colors.WARNING}python-Levenshtein not found. Distance metrics will be unavailable.{Colors.ENDC}")
    
    tok = Tokenizer(Config.vocab_path)
    if not tok.load_vocab(): return
    
    if not os.path.exists(Config.model_path): 
        print("No model."); return
        
    model = GPT(len(tok.stoi))
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()
    
    files = [os.path.join(Config.test_data_dir, f) for f in os.listdir(Config.test_data_dir) if f.endswith('.ir')]
    if not files: 
        print("No test files."); return
    
    for fp in sorted(files):
        print(f"\n{Colors.HEADER}TESTING: {os.path.basename(fp)}{Colors.ENDC}")
        with open(fp, 'r') as f: content = f.read()
        
        # Parse
        buttons = []
        proto = addr = None
        for sec in content.split('#'):
            lines = sec.strip().split('\n')
            name = cmd = None
            for line in lines:
                if 'name:' in line: name = line.split(':',1)[1].strip()
                if 'command:' in line: cmd = line.split(':',1)[1].strip().replace(' ','')
                if 'protocol:' in line: proto = line.split(':',1)[1].strip()
                if 'address:' in line: addr = line.split(':',1)[1].strip().replace(' ','')
            if name and cmd:
                t = normalize_button(name)
                if t: buttons.append((t, cmd))
        
        if not buttons: 
            print(f"{Colors.FAIL}No valid buttons found.{Colors.ENDC}")
            continue
        
        print(f"Protocol: {Colors.BOLD}{proto}{Colors.ENDC}, Address: {Colors.BOLD}{addr}{Colors.ENDC}")
        print(f"{'Context':<5} | {'Target Btn':<15} | {'Pred Code':<15} | {'Real Code':<15} | {'Status':<10}")
        print("-" * 75)
        
        total_dist = 0
        exact = 0
        steps = 0
        
        for i in range(1, len(buttons)):
            ctx = buttons[:i]
            target_t, target_c = buttons[i]
            
            # TEACHER FORCING PROMPT
            # <BOS> PROTOCOL:NEC ADDRESS:0400... [BTN_POWER]:CMD ... [BTN_TARGET]:
            # NOTE: We construct the prompt manually but ensure spaces are correct for tokenizer
            prompt = f"<BOS> PROTOCOL:{proto} ADDRESS:{addr} " + " ".join([f"{t}:{c}" for t,c in ctx]) + f" {target_t}:"
            
            x = torch.tensor(tok.encode(prompt), dtype=torch.long, device=Config.device)[None, ...]
            
            with torch.no_grad():
                # We only need enough tokens for the hex code (approx 8 chars + potential spaces)
                out = model.generate(x, 20, top_k=1)
            
            # Decode: Tokenizer might not handle partial spaces well, so we decode full then slice
            full_gen = tok.decode(out[0].tolist())
            
            # Find the target button in the generated string.
            # We expect the prompt to end with "{target_t}:"
            # But the tokenizer might encode "{target_t}:" as multiple tokens or one depending on vocab.
            # Robust way: Search for "{target_t}:" from the right side of the prompt area.
            
            try:
                # Find the LAST occurrence of the target button label in the full text
                # This handles cases where prompt reconstruction might be slightly off vs decoding
                start_marker = f"{target_t}:"
                start_idx = full_gen.rindex(start_marker) + len(start_marker)
                gen_txt = full_gen[start_idx:].strip()
            except ValueError:
                # Fallback: just slice by length (riskier if token decoding adds spaces)
                 gen_txt = full_gen[len(prompt):].strip()

            # Regex to find first hex sequence
            match = re.match(r'([0-9A-Fa-f\s]+)', gen_txt)
            
            pred_code = "???"
            dist = 999
            is_match = False
            
            if match:
                # clean up spaces
                raw_hex = match.group(1).replace(" ", "")
                pred_code = raw_hex[:8].upper() 
                
                target_c_clean = target_c.replace(" ", "").upper()
                
                if Levenshtein:
                    dist = Levenshtein.distance(pred_code, target_c_clean)
                else:
                    dist = 0 if pred_code == target_c_clean else 1
                
                is_match = (pred_code == target_c_clean)
                
            status = f"{Colors.OKGREEN}MATCH{Colors.ENDC}" if is_match else f"{Colors.FAIL}Diff({dist}){Colors.ENDC}"
            if is_match: exact += 1
            total_dist += dist
            
            print(f"{i:<5} | {target_t:<15} | {pred_code:<15} | {target_c:<15} | {status}")
            steps += 1
        
        if steps:
            acc = (exact / steps) * 100
            avg_d = total_dist / steps
            print(f"\n{Colors.BOLD}Summary:{Colors.ENDC} Accuracy: {acc:.1f}% | Avg Dist: {avg_d:.2f}")


def run_test_single(protocol, address, context_btn, context_code, target_btn):
    print(f"Testing Single Prediction: {target_btn}")
    
    tok = Tokenizer(Config.vocab_path)
    if not tok.load_vocab(): return
    
    if not os.path.exists(Config.model_path):
        print("Model not found.")
        return
        
    model = GPT(len(tok.stoi))
    model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
    model.to(Config.device)
    model.eval()
    
    # Normalize buttons
    ctx_token = normalize_button(context_btn)
    tgt_token = normalize_button(target_btn)
    
    if not ctx_token:
        print(f"Error: Unknown context button name '{context_btn}'")
        return
    if not tgt_token:
        print(f"Error: Unknown target button name '{target_btn}'")
        return

    # Clean code
    ctx_code = context_code.replace(" ", "").upper()
    addr = address.replace(" ", "").upper()
    
    # Construct Prompt
    # <BOS> PROTOCOL:NEC ADDRESS:0400 [BTN_POWER]:CMD [BTN_TARGET]:
    prompt = f"<BOS> PROTOCOL:{protocol} ADDRESS:{addr} {ctx_token}:{ctx_code} {tgt_token}:"
    print(f"Prompt: {prompt}")
    
    ids = tok.encode(prompt)
    if not ids: 
        print("Error: Could not encode prompt.")
        return
        
    x = torch.tensor(ids, dtype=torch.long, device=Config.device)[None, ...]
    
    with torch.no_grad():
        # Generate enough tokens for the hex code
        out_ids = model.generate(x, 20, temperature=Config.temperature, top_k=Config.top_k)
    
    generated = tok.decode(out_ids[0].tolist())
    
    # Extract result
    # We look for the part after our prompt
    try:
        start_marker = f"{tgt_token}:"
        # Find the last occurrence to be safe
        start_idx = generated.rindex(start_marker) + len(start_marker)
        gen_txt = generated[start_idx:].strip()
        
        # Regex for hex code
        match = re.match(r'([0-9A-Fa-f]+)', gen_txt)
        if match:
            pred_code = match.group(1).upper()
            print(f"Predicted Code for {target_btn}: {Colors.OKGREEN}{pred_code}{Colors.ENDC}")
        else:
            print(f"Predicted Code for {target_btn}: {Colors.FAIL}No hex found ({gen_txt}){Colors.ENDC}")
            
    except ValueError:
        print(f"Error: Could not find target marker in output.")
        print(f"Raw Output: {generated}")

# ==========================================
# MAIN ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="IRRemoteGPT Consolidated Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prep
    subparsers.add_parser('prep', help='Prepare data from IRDB')
    
    # Train
    subparsers.add_parser('train', help='Train the model')
    
    # Generate
    gen_parser = subparsers.add_parser('generate', help='Generate IR file')
    gen_parser.add_argument('--protocol', required=True)
    gen_parser.add_argument('--address', required=True)
    gen_parser.add_argument('--known_btn', required=True)
    gen_parser.add_argument('--known_code', required=True)
    gen_parser.add_argument('--output', default='generated.ir')
    
    # Test
    subparsers.add_parser('test', help='Test model on test_data')

    # Test Single
    ts_parser = subparsers.add_parser('test-single', help='Predict a single button code')
    ts_parser.add_argument('--protocol', required=True)
    ts_parser.add_argument('--address', required=True)
    ts_parser.add_argument('--context_btn', required=True, help='Name of known button')
    ts_parser.add_argument('--context_code', required=True, help='Hex code of known button')
    ts_parser.add_argument('--target_btn', required=True, help='Name of button to predict')
    
    args = parser.parse_args()
    
    if args.command == 'prep':
        run_prep_data()
    elif args.command == 'train':
        run_train()
    elif args.command == 'generate':
        run_generate(args.protocol, args.address, args.known_btn, args.known_code, args.output)
    elif args.command == 'test':
        run_test()
    elif args.command == 'test-single':
        run_test_single(args.protocol, args.address, args.context_btn, args.context_code, args.target_btn)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
