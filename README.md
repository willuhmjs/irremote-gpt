# IRRemoteGPT

IRRemoteGPT is a deep learning project that utilizes a Transformer-based language model (NanoGPT) to generate Infrared (IR) remote control signals in the Flipper Zero `.ir` format. By training on a corpus of existing IR remote files, the model learns to synthesize valid signal configurations, including protocol, address, and button command codes.

## Project Structure

*   `main.py`: The single entry point for all operations: data preparation, training, generation, and testing. It consolidates the configuration, tokenizer, model architecture, and logic.
*   `IRDB/`: Directory containing the source database of `.ir` files.
*   `test_data/`: Directory containing `.ir` files used for evaluation.
*   `requirements.txt`: List of Python dependencies.

## Installation

### Prerequisites

*   Python 3.x
*   PyTorch (CUDA recommended for training)

### Dependencies

1.  Clone this repository.
2.  Install the required Python packages:

```bash
pip install -r requirements.txt
pip install python-Levenshtein
```

## Usage

All operations are performed through the `main.py` script.

### 1. Data Preparation

Scan the `IRDB/` directory, normalize button names, and create the training corpus and vocabulary.

```bash
python main.py prep
```

### 2. Training

Train the NanoGPT model. This will automatically use a GPU if available.

```bash
python main.py train
```

The script will:
*   Train the model on the prepared corpus.
*   Periodically evaluate on a validation split.
*   Save the best model to `model.pt`.

### 3. Generation

Generate a new IR remote file by providing a starting context (protocol, address, and a known button).

```bash
python main.py generate --protocol NEC --address 04000000 --known_btn POWER --known_code 0A1F0000
```

*   **Arguments:**
    *   `--protocol`: The IR protocol (e.g., NEC).
    *   `--address`: The device address in hex.
    *   `--known_btn`: A known button name (e.g., POWER) to seed the generation.
    *   `--known_code`: The hex code for the known button.
    *   `--output`: Output filename (default: `generated.ir`).

### 4. Evaluation

Evaluate the model's accuracy on the test files in `test_data/`.

```bash
python main.py test
```

This performs a "next-button" prediction test, measuring how accurately the model can predict the next button's token and command code given the previous buttons. Metrics include Levenshtein distance and exact code match percentage.

## Configuration

Configuration parameters (hyperparameters, file paths, etc.) are defined in the `Config` class at the top of `main.py`. You can modify them directly in the file to adjust batch size, learning rate, model depth, etc.
