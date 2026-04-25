# DynaPrompt Code Changes Documentation

This document outlines the major architectural and functional changes implemented in `dynap_classification.py` compared to the original version.

## 1. Architectural Enhancements

### **Proactive Latent Routing (PLR)**
- **Feature Extraction**: Added `model.get_visual_features()` to extract latent vectors ($z_x$) for routing decisions.
- **Hit/Miss Logic**: Implemented a similarity-based routing mechanism where the input is compared against a bank of visual prototypes.
- **False Hit Detection**: Added a post-adaptation entropy check. If entropy remains high ($> \text{fallback\_threshold}$), the system rolls back and forces a "Discovery" pass.

### **Dynamic Prompt Buffer**
- **Unlimited Growth**: Introduced `--unlimited_buffer` which allows the prompt learner to add new expert prompts dynamically when a novel domain is discovered.
- **Replacement Strategy**: For fixed buffers, an LRU-style replacement is used to overwrite the oldest expert.
- **Partial Encoding**: Updated `test_time_tuning` to encode only `Top-K` candidate prompts during backpropagation, preventing OOM errors as the buffer grows.

## 2. Algorithmic Logic Updates

### **Prototype Management**
- **Prototype Bank**: Maintains a mapping of `(prompt_index, class_index) -> mean_feature_vector`.
- **Cumulative Updating**: Prototypes are updated using a running average:  
  $\mathbf{c}_{new} = \frac{n \cdot \mathbf{c}_{old} + z_x}{n+1}$
- **Conditional Updates**: Updates only occur if the adapted entropy is below the `entropy_threshold` and shows improvement over the raw entropy.

### **Confident Selection Sub-batching**
- Added a safety check in `select_confident_samples` to ensure at least one sample is selected even with small batches or high entropy, preventing division-by-zero or empty tensor errors.

## 3. Infrastructure & Compatibility

### **Device Agnostic Support**
- Full support for **MPS (Metal Performance Shaders)** on macOS.
- Explicit memory management with `torch.mps.empty_cache()` to prevent high-watermark memory leaks.
- Universal `autocast` handling across CUDA and MPS.

### **Experiment Tracking**
- **Unique Run Directories**: Logic to automatically create unique `run_dir_1`, `run_dir_2`, etc., to prevent overwriting results.
- **Routing Stats**: Exports comprehensive metrics (`routing_stats.json`) including Hit/Miss rates, False Hits, and prompt usage frequency.

## 4. New Configuration Flags
| Flag | Description |
| :--- | :--- |
| `--proactive_routing` | Enables similarity-based expert selection. |
| `--routing_threshold` | Similarity cut-off for a "Hit" decision. |
| `--unlimited_buffer` | Allows the expert pool to grow indefinitely. |
| `--fallback_threshold`| Triggers discovery if tuned entropy is too high. |
| `--sample_size` | Allows running experiments on a subset of data. |
| `--run_dir` | Custom directory for logs and statistics. |
