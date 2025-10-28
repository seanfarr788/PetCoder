# 🧠 DiseaseCoder  
> Automated disease code mapping for veterinary and human clinical text using ICD-11, ICD-10, or SNOMED frameworks.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Hugging Face](https://img.shields.io/badge/HuggingFace-transformers-yellow.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![Release](https://img.shields.io/badge/Release-Stable-green.svg)]()

---

## 🧩 Overview

**DiseaseCoder** automatically maps free-text medical or veterinary notes to structured disease codes (ICD-11, ICD-10, or SNOMED).  
It integrates:
- 🧬 **Named Entity Recognition (NER)** to identify relevant medical terms  
- 🧠 **Sentence embeddings** for semantic understanding  
- 🗂️ **Embedding-based lookup** for precise code mapping  
- ⚡ **Caching and batch support** for large-scale EHR datasets  

This tool is designed for researchers and practitioners working with electronic health records (EHRs), epidemiological data, or clinical NLP pipelines.

---

## ⚙️ Installation

```bash
pip install pettag
```

If you're using GPU acceleration, ensure you have the CUDA-enabled version of PyTorch installed.

---

## 🚀 Quickstart

### 🔹 Single Text Input

```python
from disease_coder import DiseaseCoder

# Initialize the coder
coder = DiseaseCoder()

# Example text
text = "Cookie presented with vomiting and diarrhea. Suspected gastroenteritis."

# Predict the disease code(s)
output = coder.predict(text=text)
print(output)
```

**Example Output:**

```json
{
    "Code": [
        {
            "Chapter": "Certain infectious or parasitic diseases",
            "Code": "1A40.Z",
            "Framework": "ICD11",
            "Input Disease": "gastroenteritis",
            "Similarity": 0.9393,
            "Title": "Infectious gastroenteritis or colitis without specification of infectious agent",
            "URI": "https://icd.who.int/browse/2025-01/mms/en#1688127370/"
        }
    ],
    "pathogen_extraction": [],
    "symptom_extraction": [
        "vomiting",
        "diarrhea"
    ]
}
```

---

### 🔹 Dataset Input

```python
coder = DiseaseCoder(
    dataset="data/clinical_notes.csv",
    text_column="note",
    framework="icd11",
    output_dir="outputs/icd11_coded/"
)

# Run predictions on an entire dataset
coder.predict()
```

The coded dataset will be saved automatically to the specified `output_dir`.

---

## 🧠 Parameters

| Parameter | Type | Default | Description |
|------------|------|----------|-------------|
| `framework` | `str` | `'icd11'` | Coding framework: `'icd11'`, `'icd10'`, or `'snomed'` |
| `dataset` | `str` or `Dataset` | `None` | Path to dataset or HuggingFace `Dataset` |
| `split` | `str` | `'train'` | Dataset split (e.g., `'train'`, `'test'`) |
| `model` | `str` | `'seanfarrell/bert-base-uncased'` | Token classification model |
| `tokenizer` | `str` | `None` | Tokenizer name (defaults to model) |
| `embedding_model` | `str` | `'sentence-transformers/embeddinggemma-300m-medical'` | Sentence embedding model |
| `synonyms_dataset` | `str` | `'seanfarrell/ICD-11_synonyms'` | ICD synonym dataset |
| `synonyms_embeddings_dataset` | `str` | `'cache/ICD-11_synonyms_embeddings.pt'` | Cached ICD embeddings |
| `text_column` | `str` | `'text'` | Text column name |
| `label_column` | `str` | `'labels'` | Label column name |
| `cache` | `bool` | `True` | Enable caching |
| `cache_path` | `str` | `'petharbor_cache/'` | Cache directory |
| `logs` | `str` | `None` | Log file path (logs to console if `None`) |
| `device` | `str` | `'cuda:0'` or `'cpu'` | Device for computation |
| `output_dir` | `str` | `None` | Directory to save outputs |

---

## 🧩 How It Works

1. **Entity Extraction**  
   Identifies medically relevant entities using a pretrained token-classification model.

2. **Semantic Embedding**  
   Converts entities to dense embeddings with a SentenceTransformer model.

3. **Code Matching**  
   Finds the most semantically similar ICD-11 / ICD-10 / SNOMED entry using cosine similarity.

4. **Caching & Efficiency**  
   ICD embeddings are saved to disk on the first run (`.pt` format) for faster reuse later.

---

## 📦 Output

Depending on the input mode:

- **Single text input:** returns a structured Python dictionary with predicted codes.  
- **Dataset input:** saves a processed dataset with new code columns to `output_dir`.

---

## 🔧 Advanced Usage

### 💾 Regenerate ICD Embedding Store

If the ICD embedding store doesn’t exist, it will be created automatically.  
To rebuild it manually:

```python
from datasets import load_dataset

coder = DiseaseCoder()
dataset = load_dataset("seanfarrell/ICD-11_synonyms", split="train")
coder._preprocess_icd_lookup(disease_code_lookup=dataset, save_path="cache/icd_lookup.pt")
```

### 🧾 Logging

Enable persistent logs:

```python
coder = DiseaseCoder(logs="logs/run.log")
```

### 🧬 Framework Switching

Switch easily between ICD and SNOMED frameworks:

```python
coder = DiseaseCoder(framework="snomed")
```

---

## 📂 Recommended Project Structure

```
project/
│
├── data/
│   └── clinical_notes.csv
│
├── cache/
│   └── ICD-11_synonyms_embeddings.pt
│
├── outputs/
│   └── icd11_coded/
│
├── logs/
│   └── run.log
│
└── disease_coder.py
```

---

## 🧾 Citation

If you use this tool in your research, please cite the PetHarbor and PetTag projects:

```bibtex
@misc{pettag2025,
  author       = {Farrell, Sean},
  title        = {PetHarbor: Veterinary Language Models for Structured Health Record Coding},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/sean-farrell/petharbor}
}
```

---

## ❤️ Acknowledgements

This package is part of the **PetTag / PetHarbor** ecosystem —  
a suite of NLP tools for large-scale veterinary EHR data analysis.

Built with:
- [Transformers](https://huggingface.co/transformers)
- [SentenceTransformers](https://www.sbert.net)
- [PyTorch](https://pytorch.org)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)

---

## 🐾 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
