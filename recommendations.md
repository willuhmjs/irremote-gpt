# IR Code Generation Model Improvement Plan

Based on the analysis of the codebase and dataset, here are the recommended improvements to enhance the model's accuracy, efficiency, and code quality.

## 1. Data Representation & Tokenization
**Problem:** The current character-level tokenization for hex codes (e.g., `1`, `0`, `E`, `F`...) is inefficient, unnecessarily increasing sequence length.
**Solution:** Implement a **Byte-Level Tokenizer** for hex values.
- **Change:** instead of `['1', '0', 'E', 'F', '0', '0', '0', '0']`, represent hex strings as bytes `['10', 'EF', '00', '00']`.
- **Impact:** This effectively **halves** the number of tokens required for hex codes, allowing the model to fit twice as much information into the same context window.
- **Action:** Create a shared `tokenizer.py` module to ensure training and generation use identical logic.

## 2. Model Architecture
**Problem:** The context window (`block_size=256`) is too small. A remote with 30+ buttons can easily exceed 256 tokens using the current scheme, causing the model to "forget" the protocol or address defined at the start of the file.
**Solution:** Scale up the model parameters.
- **Context Window:** Increase `block_size` from `256` to **`1024`**. This ensures the entire remote definition fits in context.
- **Embedding Size:** Increase `n_embd` from `256` to **`384`** or **`512`** to capture more complex relationships between protocols and commands.
- **Layers/Heads:** Slightly increase `n_layer` (e.g., to 8) and `n_head` (e.g., to 6 or 8) to support the larger embedding dimension.

## 3. Training Logic
**Problem:** The tokenizer logic is duplicated in `train.py` and `generate.py`, violating DRY (Don't Repeat Yourself) and risking inconsistency.
**Solution:** Refactor to use the shared `tokenizer.py`.
- **Action:** Update `train.py` to import the tokenizer.
- **Action:** Implement a custom `collate_fn` or pre-processing step if we switch to a more complex tokenization strategy.

## 4. Generation Strategy
**Problem:** `generate.py` uses basic multinomial sampling without temperature control, leading to potentially erratic outputs. It also lacks robust parsing validation.
**Solution:** Enhance the generation script.
- **Temperature:** Add a `--temperature` argument to control randomness (lower for more deterministic codes, higher for variety).
- **Validation:** Add a post-generation check to ensure the generated hex codes match the expected length for the chosen protocol (if known) or just strictly enforce hex format.

## 5. Directory Structure
**Recommendation:** Clean up the project structure.
- Move core logic into a `src/` directory.
- `src/model.py`
- `src/tokenizer.py`
- `src/utils.py`

## Action Plan (Next Steps)
1.  **Create `tokenizer.py`:** Implement the byte-level hex tokenization.
2.  **Refactor `model.py`:** Update hyperparameters (`block_size=1024`, `n_embd=384`).
3.  **Refactor `train.py`:** Integrate the new tokenizer and updated model config.
4.  **Refactor `generate.py`:** Integrate the new tokenizer and add temperature control.
