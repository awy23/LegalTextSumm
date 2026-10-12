# Legal Texts Translation and Summarization using Large Language Models

**Thesis:**
Vasile Andrei-Alexandru. Legal Texts Translation and Summarization using Large Language Models. Bachelor's Thesis. 
National University of Science and Technology Politehnica Bucharest, Faculty of Automatic Control and Computers, July 2026.

## Code

**Packages needed (see requirements.txt for exact versions):**
* torch==2.10.0+cu128
* torchvision==0.25.0+cu128
* torchaudio==2.10.0+cu128
* triton==3.6.0
* xformers==0.0.35
* transformers==4.57.6
* unsloth==2026.3.15
* unsloth_zoo==2026.3.6
* accelerate>=1.4.0
* trl==0.24.0
* peft==0.18.0
* datasets==3.4.1
* bitsandbytes
* rouge-score
* bert-score
* deep_translator
* pandas
* numpy
* scikit-learn
* gradio
* matplotlib
* seaborn
* keybert
* nltk

**Dataset format:**
The datasets used are specific subsets of BillSum (English) and ReaderBench (Romanian). For the training and pre-processing scripts, the expected CSV headers include:
* `Content` / `text` - the textual content of the legal document
* `Summary` / `summary` - the reference summary for the document
Note: All scripts contain Google Drive paths (`/content/drive/MyDrive/Licenta/...`) as they were executed in Google Colab Pro. Please update these paths to your local directories if you plan to run the code locally.

## Setup & Cache Warmup

Pre-downloads and caches the baseline LLM architectures for faster processing. Run it before executing GUI.py. Found in the root directory.
* `Setup_Models.py` - initializes the Unsloth instances for RoLlama-3.1,LLaMA-3.1, Phi-3.5, and RoMistral.
* **Run:** `python Setup_Models.py`

## Statistics

Found in the root directory.
* `EDA.py` - calculates corpus statistics, document lengths, compression ratios, extracts top N-grams, and KeyBERT keywords. Generates word frequency plots for both BillSum and ReaderBench datasets.
* **Run:** `python EDA.py`

## Data Preparation

Found in the root directory.
* `prepare_dataset.py` - cleans the datasets, applies HTML tag removal, calculates precise token lengths using specific tokenizers, and extracts exactly 800 valid documents per language. 
* **Run:** `python prepare_dataset.py`

## Models

Run all the Supervised Fine-Tuning (SFT) models using 5-Fold Cross Validation and Grid Search. Found in the root directory.
* **Files:** 
  * `Llama3.1_Training.py`
  * `Phi3-5_Training.py`
  * `RoLlama3.1_Training.py`
  * `RoMistral_Training.py`
* **Parameters inside scripts:**
  * `DATASET_PATH` - path to the preprocessed CSV dataset
  * `CALE_SALVARE_DRIVE` - directory where to save the optimized LoRA weights
  * `LEARNING_RATES` / `WEIGHT_DECAYS` - grid search parameters
* **Run:** `python Llama3.1_Training.py` (repeat for each architecture)

## Evaluation & Inference (GUI)

Found in the root directory.
* `GUI.py` - launches a Gradio web application for interactive inference. Features a dynamic VRAM management system to swap active models. It performs native text summarization, cross-lingual summarization via translation pipelines, and computes evaluation metrics (ROUGE-1, BERTScore).
* **Run:** `python GUI.py`