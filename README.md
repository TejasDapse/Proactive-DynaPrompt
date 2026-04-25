# Proactive DynaPrompt: Scalable Test-Time Adaptation with Latent Routing

Proactive DynaPrompt is an advanced Test-Time Adaptation (TTA) framework for Vision-Language Models (like CLIP). It extends the original DynaPrompt by introducing **Proactive Latent Routing (PLR)**, an **Unlimited Prompt Buffer**, and **Entropy-Gated verification** to handle extreme distribution shifts with constant-time efficiency.

## 🚀 Key Features

- **Proactive Latent Routing (PLR):** Uses visual similarity to reuse previously tuned "expert" prompts, bypassing expensive discovery loops.
- **Unlimited Prompt Buffer:** Dynamically adds new experts as novel domains are discovered without memory explosion.
- **Memory-Efficient TTA:** Implements "Subset Selection" to materialize only necessary prompts during backpropagation.
- **Cross-Platform Support:** Native support for **Apple Silicon (MPS)** and NVIDIA CUDA.
- **Robustness:** Entropy-based fallback mechanisms to detect "False Hits" and trigger discovery.

---

## 🛠️ Installation

### 1. Clone & Environment
```bash
git clone https://github.com/TejasDapse/Proactive-DynaPrompt.git
cd Proactive-DynaPrompt

# Create a virtual environment
python -m venv .dyna_prompt_env
source .dyna_prompt_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Preparation

### Oxford Pets Setup
To run experiments on the Oxford Pets dataset, follow these steps to download and organize the data:

1. **Download the Dataset & Annotations:**
   ```bash
   mkdir -p data/pets
   cd data/pets
   wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
   wget https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
    ```
2. Extract the Files:
    ```bash
    tar -xzf images.tar.gz
    tar -xzf annotations.tar.gz
    ```
3. Download the JSON Split: The training/test split used in this project can be downloaded here:
    Oxford Pets Split (Google Drive): https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view

    mkdir -p data/data_splits
    mv path/to/downloaded/split_zhou_OxfordPets.json data/data_splits/

## Running experiments:
```bash
 python ./dynap_classification.py ./data --test_sets pets -a ViT-B/16 -b 64 --lr 5e-3 --seed 0 --n_ctx 4 --tpt --ctx_init fication.py ./data --test_sets pets -a ViT-B/16 -b 64 --lr 5e-3 --seed 0 --n_ctx 4 --tpt --ctx_init a_photo_of_a --log_date 20260407 --num_prompts 5 --onlinetpt --selection_p 0.1 --sample_size 1000 --proactive_routing  --routing_threshold 0.85 --top_k_prompts 5 --unlimited_buffer
 ```