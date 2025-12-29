import os
import re
import torch
import sys
import time

# Import core components from main.py
try:
    from main import Config, Tokenizer, GPT, normalize_button, format_hex
except ImportError:
    print("Error: Could not import components from main.py. Make sure you are running this from the project root.")
    sys.exit(1)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(step_num=None, title=None):
    clear_screen()
    print("🐬  Flipper Zero IR Wizard  🐬")
    print("====================================")
    if step_num is not None:
        print(f"Step {step_num}: {title}")
        print("------------------------------------")

def get_input(prompt, required=True, example=None):
    while True:
        display_prompt = f"{prompt}"
        if example:
            display_prompt += f" (e.g., {example})"
        display_prompt += ": "
        
        value = input(display_prompt).strip()
        if value or not required:
            return value
        print("❌ This field is required. Please try again.")

def explain_step_1():
    print("\nℹ️  To get started, we need the Protocol and Address from a captured signal.")
    print("   1. Open your Flipper Zero.")
    print("   2. Go to 'Infrared' -> 'Learn New Remote'.")
    print("   3. Point your remote at the Flipper and press 'Power'.")
    print("   4. You will see something like 'protocol: NEC, address: 04 00, command: ...'")
    print("\n")

def run_wizard():
    print_header()
    print("Welcome! I'll help you create a complete IR remote file for your Flipper Zero")
    print("even if you only have a few buttons captured.")
    print("\nPress Enter to start...")
    input()

    # --- Step 1: Protocol & Address ---
    print_header(1, "Protocol & Address")
    explain_step_1()
    
    protocol = get_input("Enter Protocol", example="NEC, Samsung32")
    address = get_input("Enter Address", example="04 00, 0x0400")
    
    # Clean address
    address = address.replace("0x", "").replace(" ", "").upper()
    
    # --- Step 2: Known Buttons ---
    print_header(2, "Teach Me Buttons")
    print("Now, tell me about the buttons you have ALREADY captured.")
    print("The more buttons you give me, the better I can guess the missing ones!")
    print("Type 'done' when you are finished adding buttons.")
    
    known_buttons = []
    while True:
        print(f"\n📋 Current Buttons: {[n for n, c in known_buttons]}")
        name = input("\nButton Name (e.g., Power, Vol+, or 'done'): ").strip()
        
        if name.lower() == 'done':
            if not known_buttons:
                print("❌ Please enter at least one button so I can learn the pattern.")
                continue
            break
        
        if not name: continue
            
        command = input(f"Hex Command for '{name}' (from Flipper): ").strip()
        if not command:
            print("❌ Command is required.")
            continue
            
        # Validate/Clean hex
        command = command.replace("0x", "").replace(" ", "").upper()
        if not re.match(r'^[0-9A-F]+$', command):
            print("❌ Invalid hex code. It should look like 'A1 F2' or 'A1F2'.")
            continue
            
        known_buttons.append((name, command))
        print(f"✅ Added {name}")

    # --- Step 3: Selection ---
    print_header(3, "What to Generate?")
    print("Which buttons are you missing? I can try to fill in the blanks.")
    print("\n1. 📺 Standard TV Set (Power, Vol, Ch, Nav, Numbers, Colors, Playback)")
    print("2. 🔈 Minimal Set (Power, Vol, Ch, Nav)")
    print("3. ✍️  Custom List (You type them)")
    
    choice = input("\nSelect an option (1-3): ").strip()
    
    standard_set = [
        'Power', 'Vol+', 'Vol-', 'Mute', 
        'Ch+', 'Ch-', 'Source', 'Menu', 'Exit', 'Back', 'Info',
        'Up', 'Down', 'Left', 'Right', 'OK',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'Red', 'Green', 'Yellow', 'Blue',
        'Play', 'Pause', 'Stop', 'Rewind', 'FastForward', 'Next', 'Prev'
    ]
    
    minimal_set = [
        'Power', 'Vol+', 'Vol-', 'Mute', 
        'Ch+', 'Ch-', 'Up', 'Down', 'Left', 'Right', 'OK'
    ]
    
    target_buttons = []
    if choice == '1': target_buttons = standard_set
    elif choice == '2': target_buttons = minimal_set
    else:
        print("\nEnter button names separated by commas.")
        raw = input("Buttons: ")
        target_buttons = [b.strip() for b in raw.split(',') if b.strip()]

    # Filter out known
    known_names_normalized = {normalize_button(n) for n, c in known_buttons}
    final_targets = []
    skipped = []
    
    for t in target_buttons:
        norm = normalize_button(t)
        if norm and norm not in known_names_normalized:
            final_targets.append(t)
        elif not norm:
            skipped.append(t)

    # --- Step 4: AI Generation ---
    print_header(4, "AI Magic")
    if skipped:
        print(f"⚠️  Skipping unknown button types: {skipped}")
        
    print(f"🤖 Loading AI Brain... (This might take a second)")
    
    # Load Model
    try:
        tok = Tokenizer(Config.vocab_path)
        if not tok.load_vocab():
            print("❌ Error: Vocabulary not found. Run 'python main.py prep' first.")
            return

        if not os.path.exists(Config.model_path):
            print("❌ Error: AI Model not found. Run 'python main.py train' first.")
            return

        model = GPT(len(tok.stoi))
        model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()
    except Exception as e:
        print(f"❌ Critical Error loading AI: {e}")
        return

    # Build Context
    context_parts = [f"<BOS> PROTOCOL:{protocol} ADDRESS:{address}"]
    for name, cmd in known_buttons:
        norm = normalize_button(name)
        if norm:
            context_parts.append(f"{norm}:{cmd}")
            
    base_context = " ".join(context_parts)
    
    print("\nThinking...")
    generated_data = []
    
    for i, btn_name in enumerate(final_targets):
        norm_token = normalize_button(btn_name)
        if not norm_token: continue
        
        # Fancy progress
        print(f"   🔮 Predicting {btn_name:<15}", end="", flush=True)
        
        prompt = f"{base_context} {norm_token}:"
        ids = tok.encode(prompt)
        x = torch.tensor(ids, dtype=torch.long, device=Config.device)[None, ...]
        
        with torch.no_grad():
            out_ids = model.generate(x, 20, temperature=0.8, top_k=40)
            
        full_text = tok.decode(out_ids[0].tolist())
        
        try:
            marker = f"{norm_token}:"
            start_idx = full_text.rindex(marker) + len(marker)
            gen_part = full_text[start_idx:].strip()
            match = re.match(r'([0-9A-Fa-f]+)', gen_part)
            
            if match:
                code = match.group(1).upper()
                generated_data.append((btn_name, code))
                print(f" -> ✨ {code}")
            else:
                print(" -> ❓ Failed")
        except ValueError:
            print(" -> ❓ Error")
            
    # --- Step 5: Save ---
    print_header(5, "Save & Finish")
    print(f"I generated {len(generated_data)} new buttons for you!")
    
    default_filename = f"{protocol}_{address}.ir"
    filename = input(f"Output filename [{default_filename}]: ").strip()
    if not filename:
        filename = default_filename
    if not filename.endswith(".ir"):
        filename += ".ir"
        
    try:
        with open(filename, 'w') as f:
            f.write("Filetype: IR signals file\n")
            f.write("Version: 1\n")
            f.write("# Generated by IRRemoteGPT Wizard 🐬\n")
            
            f.write("\n# 🟢 Known Buttons (Verified)\n")
            for name, cmd in known_buttons:
                f.write(f"name: {name}\n")
                f.write("type: parsed\n")
                f.write(f"protocol: {protocol}\n")
                f.write(f"address: {format_hex(address)}\n")
                f.write(f"command: {format_hex(cmd)}\n")
                f.write("#\n")
                
            f.write("\n# 🟣 AI Generated Buttons (Experimental)\n")
            for name, cmd in generated_data:
                f.write(f"name: {name}\n")
                f.write("type: parsed\n")
                f.write(f"protocol: {protocol}\n")
                f.write(f"address: {format_hex(address)}\n")
                f.write(f"command: {format_hex(cmd)}\n")
                f.write("#\n")
                
        print(f"\n✅ Success! File saved to: {os.path.abspath(filename)}")
        print("Copy this file to your Flipper Zero's SD card in /infrared/.")
        
    except IOError as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    run_wizard()
