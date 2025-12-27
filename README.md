# IRRemoteGPT

IRRemoteGPT is a deep learning project that utilizes a Transformer-based language model (NanoGPT) to generate Infrared (IR) remote control signals in the Flipper Zero `.ir` format. By training on a corpus of existing IR remote files, the model learns to synthesize valid signal configurations, including protocol, address, and button command codes.

## Project Structure

*   `prep_data.py`: Preprocessing script that scans the `IRDB/` directory, normalizes button names (e.g., "pwr" -> `[BTN_POWER]`), and compiles them into a text corpus.
*   `train.py`: The main training script that builds the tokenizer and trains the GPT model on the prepared corpus.
*   `model.py`: Contains the PyTorch implementation of the GPT architecture, including the attention mechanism and transformer blocks.
*   `generate.py`: The inference script used to generate new `.ir` files based on a starting prompt (Protocol, Address, and a seed button).
*   `test_model.py`: An evaluation script that tests the model's ability to complete a known IR sequence (Grundig MS240) given varying amounts of context.
*   `IRDB/`: Directory containing the source database of `.ir` files.

## Installation

### Prerequisites

*   Python 3.x

### Dependencies

1.  Clone this repository.
2.  Install the required Python packages:

```bash
pip install -r requirements.txt
pip install python-Levenshtein
```

*Required packages: `torch`, `numpy`, `tqdm`, `python-Levenshtein`*

## Usage

### 1. Data Preparation

Before training, you must prepare the dataset from the `IRDB/` directory.

```bash
python prep_data.py
```

This command will:
*   Scan `IRDB/` for `.ir` files.
*   Normalize common button names (Power, Volume, Channel, etc.).
*   Extract protocol, address, and command data.
*   Generate a single training corpus file named `ir_corpus.txt`.

### 2. Training the Model

Once the data is prepared, you can start training the model.

```bash
python train.py
```

The script will:
*   Build a character-level tokenizer with special tokens for buttons.
*   Train the NanoGPT model on `ir_corpus.txt`.
*   Save the trained weights to `model.pt` and the vocabulary to `vocab.json`.

*Note: The script automatically detects and uses CUDA if available.*

### 3. Generating IR Files

Use the trained model to generate a new IR remote file. You need to provide the target protocol, device address, and a known button to "seed" the generation process.

```bash
python generate.py --protocol <PROTOCOL> --address <ADDRESS_HEX> --known_btn <BUTTON_NAME> --known_code <CODE_HEX>
```

**Arguments:**
*   `--protocol`: The IR protocol to use (e.g., `NEC`, `Samsung32`).
*   `--address`: The device address in hex format (e.g., `04000000`).
*   `--known_btn`: The name of the button to start with (e.g., `POWER`).
*   `--known_code`: The hex command code for the known button.
*   `--output`: (Optional) Output filename (default: `generated.ir`).

**Example:**
To generate a remote file for an NEC device with address `04000000`, starting with the Power button:

```bash
python generate.py --protocol NEC --address 04000000 --known_btn POWER --known_code 0A1F0000
```

The output will be saved to `generated.ir` in the Flipper Zero compatible format.

### 4. Evaluating the Model

You can evaluate the model's performance on a held-out test case (Grundig MS240) using the test script. This script runs inference on a "curve," iteratively adding buttons to the context and checking if the model correctly predicts the next button and command code.

```bash
python test_model.py
```

It outputs a table showing the prediction accuracy and Levenshtein distance of the generated command codes as more context is provided.

## Model Architecture

The core of this project is a Generative Pre-trained Transformer (GPT) defined in `model.py`.

**Hyperparameters:**
*   **Embedding Dimension (`n_embd`):** 256
*   **Attention Heads (`n_head`):** 4
*   **Layers (`n_layer`):** 6
*   **Block Size (Context Window):** 256 tokens
*   **Dropout:** 0.1

**Tokenizer:**
The model uses a custom hybrid tokenizer. It treats specific button identifiers (e.g., `[BTN_POWER]`, `[BTN_VOL_UP]`) and structural markers (`<BOS>`, `PROTOCOL:`) as single tokens, while processing hex codes and other text at the character level. This allows the model to learn high-level structural patterns while maintaining the flexibility to generate precise hex sequences.

## Testing & Evaluation

To evaluate the model's performance on a known test case (Grundig MS240), use the `test_model.py` script. This script runs a "curve" test, providing the model with increasing amounts of context (e.g., 10%, 20%... of the file) and measuring its ability to predict the next button and code.

```bash
python test_model.py
```

The script outputs a step-by-step evaluation table and a final accuracy summary.
