import re
import json
import os

class Tokenizer:
    def __init__(self, vocab_path='vocab.json'):
        self.vocab_path = vocab_path
        self.stoi = {}
        self.itos = {}
        self.special_token_pattern = r'(\[BTN_[^\]]+\]|<BOS>|<EOS>|PROTOCOL:|ADDRESS:)'

    def build_vocab(self, text):
        """Builds vocabulary from text, treating hex bytes as tokens."""
        parts = self._split_text(text)
        
        vocab = set()
        for part in parts:
            if self._is_special_token(part):
                vocab.add(part)
            else:
                # Assuming part is a sequence of hex digits (and maybe spaces)
                # We want to tokenize 2-char hex bytes: "0A", "FF"
                # First clean it
                clean_part = part.replace(" ", "")
                # If length is odd, something is wrong or it's just partial. 
                # For robustness, we can fall back to characters if it's not proper hex,
                # but let's try to enforce byte tokens for hex-like strings.
                
                # Check if it looks like hex
                if re.match(r'^[0-9A-Fa-f]+$', clean_part):
                     # Split into 2-char chunks
                     for i in range(0, len(clean_part), 2):
                         byte_token = clean_part[i:i+2]
                         if len(byte_token) == 2:
                             vocab.add(byte_token)
                         else:
                             # Orphan char
                             vocab.add(byte_token)
                else:
                    # Fallback to characters for non-hex non-special
                    for char in part:
                        vocab.add(char)
                        
        vocab = sorted(list(vocab))
        self.stoi = { token:i for i,token in enumerate(vocab) }
        self.itos = { i:token for i,token in enumerate(vocab) }
        return len(vocab)

    def save_vocab(self):
        with open(self.vocab_path, 'w') as f:
            json.dump(self.stoi, f)

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
                if part in self.stoi:
                    encoded.append(self.stoi[part])
            else:
                # Hex or other
                clean_part = part.replace(" ", "")
                if re.match(r'^[0-9A-Fa-f]+$', clean_part):
                     for i in range(0, len(clean_part), 2):
                         byte_token = clean_part[i:i+2]
                         if byte_token in self.stoi:
                             encoded.append(self.stoi[byte_token])
                         else:
                             # Unknown byte?
                             pass
                else:
                    for char in part:
                         if char in self.stoi:
                             encoded.append(self.stoi[char])
        return encoded

    def decode(self, ids):
        text = ""
        for i in ids:
            token = self.itos.get(i, '')
            # If it's a byte token (2 chars, hex), maybe add space?
            # Or just raw concat. Raw concat is safer for reconstruction, 
            # but for display we might want spaces.
            # Let's just concat. The model should learn to output tokens in order.
            text += token
        return text

    def _split_text(self, text):
        parts = re.split(self.special_token_pattern, text)
        return [p for p in parts if p]

    def _is_special_token(self, token):
        return re.match(self.special_token_pattern, token) is not None
