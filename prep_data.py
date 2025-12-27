import os

def normalize_button(name):
    n = name.strip().lower()
    
    # Power
    if n in ['power', 'pwr', 'power button']: return '[BTN_POWER]'
    
    # Volume
    if n in ['vol+', 'volume up', 'vol_up']: return '[BTN_VOL_UP]'
    if n in ['vol-', 'volume down', 'vol_dn']: return '[BTN_VOL_DOWN]'
    if n == 'mute': return '[BTN_MUTE]'
    
    # Channel
    if n in ['ch+', 'channel up']: return '[BTN_CH_UP]'
    if n in ['ch-', 'channel down']: return '[BTN_CH_DOWN]'
    
    # Navigation / Menu
    if n == 'menu': return '[BTN_MENU]'
    if n == 'enter': return '[BTN_ENTER]'
    if n == 'back': return '[BTN_BACK]'
    
    # Digits
    if n in [str(i) for i in range(10)]: return f'[BTN_{n}]'
    
    return None

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Split by '#' which delimits sections in .ir files
    sections = content.split('#')
    
    buttons = []
    protocol = None
    address = None
    
    for section in sections:
        lines = section.strip().split('\n')
        btn_name = None
        btn_proto = None
        btn_addr = None
        btn_cmd = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('name:'):
                btn_name = line.split(':', 1)[1].strip()
            elif line.startswith('protocol:'):
                btn_proto = line.split(':', 1)[1].strip()
            elif line.startswith('address:'):
                # Remove spaces for compact hex representation
                btn_addr = line.split(':', 1)[1].strip().replace(' ', '')
            elif line.startswith('command:'):
                # Remove spaces for compact hex representation
                btn_cmd = line.split(':', 1)[1].strip().replace(' ', '')
        
        # We need all fields to be present
        if btn_name and btn_proto and btn_addr and btn_cmd:
            token = normalize_button(btn_name)
            if token:
                # Capture the first valid protocol and address we see for the file header
                if not protocol: protocol = btn_proto
                if not address: address = btn_addr
                
                # Check for consistency (optional, but good for data quality)
                # If the protocol/address changes mid-file for mapped buttons, 
                # strictly speaking, the file header format might be misleading.
                # For this task, we assume the first one is representative 
                # or we just take the first one found.
                
                buttons.append((token, btn_cmd))

    if buttons and protocol and address:
        # Sort buttons? Not strictly required, preserving file order is usually better for sequence data
        # Format: <BOS> PROTOCOL:NECext ADDRESS:05000000 [BTN_POWER]:4000 ... <EOS>
        cmds = " ".join([f"{t}:{c}" for t, c in buttons])
        return f"<BOS> PROTOCOL:{protocol} ADDRESS:{address} {cmds} <EOS>"
    
    return None

def main():
    root_dir = 'IRDB'
    out_lines = []
    count = 0
    
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found.")
        return

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.ir'):
                filepath = os.path.join(root, file)
                result = process_file(filepath)
                if result:
                    out_lines.append(result)
                    count += 1
    
    output_file = 'ir_corpus.txt'
    # Join all lines with a space or newline? 
    # Usually GPT input is one giant string.
    # But we want EOS to be significant.
    # Let's verify how tokenizer handles it.
    # The tokenizer splits by special tokens. 
    # Newlines in between files are just whitespace.
    full_text = "\n".join(out_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
            
    print(f"Successfully processed {count} remotes. Output written to {output_file}.")
    
    # Build and save vocab immediately
    from tokenizer import Tokenizer
    tok = Tokenizer()
    vocab_size = tok.build_vocab(full_text)
    tok.save_vocab()
    print(f"Vocabulary built and saved to vocab.json. Size: {vocab_size}")

if __name__ == '__main__':
    main()
