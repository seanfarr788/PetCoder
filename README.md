# PetHarbor

[![PyPI version](https://badge.fury.io/py/petharbor.svg)](https://badge.fury.io/py/petharbor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

PetHarbor is a Python package designed for anonymizing veterinary electronic health record (EHR) datasets using either a pre-trained model or a hash-based approach. It provides two main classes for anonymization: `lite` and `advance`.

## 🔒 Privacy Protection for Veterinary Data

We introduce two anonymisation models to address the critical need for privacy protection in veterinary EHRs:

### PetHarbor-Advanced
A state-of-the-art solution for clinical note anonymisation, leveraging an ensemble of two specialised large language models (LLMs). Each model is tailored to detect and process distinct types of identifiers within the text. Trained extensively on a diverse corpus of authentic veterinary EHR notes, these models are adept at parsing and understanding the unique language and structure of veterinary documentation. Due to its high performance and comprehensive approach, PetHarbor Advanced is our recommended solution for data sharing beyond controlled laboratory environments.

### PetHarbor-Lite
A lightweight alternative to accommodate organisations with limited computational resources. This solution employs a two-step pipeline: first, trusted partners use shared lookup hash list derived from the SAVSNET dataset to remove common identifiers. These hash lists utilise a one-way cryptographic hashing algorithm (SHA-256) with an additional protected salt. Therefore, this hash list can be made available and shared with approved research groups without the need for raw text to be transfered or viewed by end users. Second, a spaCy-based model identifies and anonymises any remaining sensitive information. This approach drastically reduces computational requirements while maintaining effective anonymisation.

![model overview](img/model_diff.png)

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
  - [PetHarbor-Advanced](#petharbor-advanced)
  - [PetHarbor-Lite](#petharbor-lite)
- [Configuration](#configuration)
- [Example Use Cases](#example-use-cases)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## ⚡ Installation <a name="installation"></a>

```bash
pip install petharbor
```

If you just want Lite or Advance then 

Lite Only:
```bash
pip install petharbor[lite]
```
Advance only:
```bash
pip install petharbor[advance]
```


### Dependencies

PetHarbor requires:
- Python >=3.7
- pandas
- datasets
- colorlog
- transformers [advance]
- torch [advance]
- accelerate [advance]
- spacy [lite]

To install a spacy model: `python -m spacy download en_core_web_sm`

## 🚀 Quick Start <a name="quick-start"></a>

You can simply pass text to the initialized class (first use may be slow as the model downloads):

```python
from petharbor.advance import Anonymiser

# Initialize the anonymizer
petharbor = Anonymiser()

# Anonymize single text
anonymized_text = petharbor.anonymise("Cookie presented to Jackson's on 25th May 2025 before travel to Hungary. Issued passport (GB52354324)")

print(anonymized_text)
# Output: <<NAME>> presented to <<ORG>> on <<TIME>> before travel to <<LOCATION>>. Issued passport (<<MISC>>)
```

> **Note:** For processing large datasets, use the batch processing approach described below for significantly better performance.

## 🛠️ Models <a name="models"></a>

### PetHarbor-Advanced Anonymization <a name="petharbor-advanced"></a>

The `advance` anonymization class uses a pre-trained model to anonymize text data.

#### Arguments

| Argument       | Type              | Default                                                                 | Description                                                                                       |
|----------------|-------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `dataset`      | `str`             | `None`                                                                  | **Required.** Path to the dataset file (e.g., `.csv`, `.arrow`).                                 |
| `split`        | `str`             | `"train"`                                                               | The split of the dataset to use. Typical options include `"train"`, `"test"`, or `"eval"`.       |
| `model`        | `str`             | `"SAVSNET/PetHarbor"`                                                   | Path to the pre-trained model or model identifier from Hugging Face.                             |
| `tokenizer`    | `str`             | `None`                                                                  | Path to the tokenizer. If not specified, defaults to the tokenizer associated with the model.    |
| `text_column`  | `str`             | `"text"`                                                                | Column name in the dataset that contains the text input data.                                    |
| `cache`        | `bool`            | `True`                                                                  | Whether to enable caching of processed datasets to speed up subsequent runs.                     |
| `cache_path`   | `str`             | `"petharbor_cache/"`                                                    | Directory path to store cache files.                                                             |
| `logs`         | `Optional[str]`   | `None`                                                                  | Optional path to save logs generated during processing.                                          |
| `device`       | `str`             | `"cuda"` if available, otherwise `"cpu"`                                | Device to run the model on. Automatically detects GPU if available.                              |
| `tag_map`      | `Dict[str, str]`  | `{ "PER": "<<NAME>>", "LOC": "<<LOCATION>>", "TIME": "<<TIME>>", "ORG": "<<ORG>>", "MISC": "<<MISC>>" }` | A dictionary mapping entity tags to replacement strings. Useful for masking/anonymizing entities. |
| `output_dir`   | `str`             | `None`                                                                  | Directory to save the processed outputs, such as transformed datasets or predictions.            |

#### Methods

- `anonymise()`: Overwrites the text_column with tag_map tags
- `predict()`: Creates a new column called labels, puts found entities in this column
- `anonymise_predict()`: Performs anonymise() and predict()

#### Example Usage

```python
from petharbor.advance import Anonymiser

if __name__ == "__main__":
    # Initialize the Anonymiser with your configuration
    advance = Anonymiser(
        dataset="path/to/dataset.csv",              # Path to input dataset
        split="train",                              # Optional: dataset split for arrow
        model="SAVSNET/PetHarbor",                  # Optional: path or name of the model
        text_column="text",                         # Column containing text to process
        cache=True,                                 # Use cache
        cache_path="petharbor_cache/",              # Where to store cache files
        logs="logs/",                               # Path to store logs
        device="cuda",                              # Device to run on: "cuda" or "cpu"
        tag_map={                                   # Entity replacement map
            "PER": "<<NAME>>",
            "LOC": "<<LOCATION>>",
            "TIME": "<<TIME>>",
            "ORG": "<<ORG>>",
            "MISC": "<<MISC>>"
        },
        output_dir="output/anonymized_data.csv"     # Where to save anonymised data
    )

    # Run the anonymisation process
    advance.anonymise()
```

### Lite Anonymization <a name="petharbor-lite"></a>

The `lite` anonymization class uses a hash-based approach to anonymize text data, requiring fewer computational resources.

#### Arguments

| Argument       | Type    | Default          | Description                                                            |
|----------------|---------|------------------|------------------------------------------------------------------------|
| `dataset_path` | `str`   | None             | The path to the dataset file (.csv or Arrow Dataset)                   |
| `hash_table`   | `str`   | None             | The path to the hash table file                                        |
| `salt`         | `str`   | None             | An optional salt value for hashing                                     |
| `cache`        | `bool`  | True             | Whether to use caching for the dataset processing                      |
| `use_spacy`    | `bool`  | False            | Whether to use spaCy for additional text processing                    |
| `spacy_model`  | `str`   | "en_core_web_sm" | The spaCy model to use for text processing                             |
| `text_column`  | `str`   | "text"           | The name of the text column in the dataset                             |
| `output_dir`   | `str`   | None             | The directory where the output files will be saved                     |

#### Methods

- `anonymise()`: Anonymizes the dataset by hashing the text data and optionally using spaCy for additional processing.

#### Example Usage

```python
from petharbor.lite import Anonymiser

lite = Anonymiser(
    dataset_path="path/to/dataset.csv",
    hash_table="path/to/pet_names_hashed.txt",
    salt="your_salt_here",
    text_column="text",
    cache=True,
    use_spacy=True,
    output_dir="output/lite_anonymized.csv",
)
lite.anonymise()
```

## ⚙️ Configuration <a name="configuration"></a>

### Device Configuration

The device (CPU or CUDA) can be configured by passing the `device` parameter to the anonymization classes. If not specified, the package will automatically configure the device.

```python
anonymizer = Anonymiser(device="cuda")  # Use GPU
# or
anonymizer = Anonymiser(device="cpu")   # Force CPU usage
```

### Caching Options

Both methods support caching to avoid re-anonymising records that have already been processed:

#### Option 1: ID-based caching (Recommended)

If your dataset includes a unique identifier for each consultation (e.g., a consult ID), you can pass this column name to enable ID-based caching:

```python
anonymizer = Anonymiser(
    dataset="path/to/dataset.csv",
    cache="consult_id",  # Name of the column containing unique identifiers
    cache_path="my_cache_folder/"
)
```

- A folder will be created to store processed IDs
- The model reads this list and skips records whose IDs are already logged
- Ideal for incremental processing of large datasets

#### Option 2: Flag-based caching

```python
anonymizer = Anonymiser(
    dataset="path/to/dataset.csv",
    cache=True  # Use a flag column 'anonymised' to track processed records
)
```

- Adds/uses an 'anonymised' flag to the dataset (1 = processed)
- Records marked as processed are skipped
- Added back to the complete dataset at the end

#### Option 3: No caching

```python
anonymizer = Anonymiser(
    dataset="path/to/dataset.csv",
    cache=False  # Process full dataset each time
)
```

## 📊 Example Use Cases <a name="example-use-cases"></a>

### Preparing Veterinary Data for Research

```python
from petharbor.advance import Anonymiser

# Initialize the anonymizer
anonymizer = Anonymiser(
    dataset="dataset.csv",
    text_column="consult_note",
    cache="consult_id"
    output_dir="anonymised_dataset.csv"
)

# Process the dataset
anonymizer.anonymise()
```

## 📈 Benchmarks <a name="benchmarks"></a>

PetHarbor was evaluated against PetEVAL

| Model            | Precision | Recall | F1-Score | Speed (docs/sec) | Memory Usage |
|------------------|-----------|--------|----------|------------------|--------------|
| PetHarbor-Advanced | 0.96      | 0.92   | 0.94     | 150.3             | ~4GB         |
| PetHarbor-Lite    | 0.89      | 0.85   | 0.87     | 87.6             | ~500MB       |

*Benchmarks performed on NVIDIA A6000 GPU with batch size=32*

## 🤝 Contributing <a name="contributing"></a>

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation <a name="citation"></a>

If you use PetHarbor in your research, please cite:

```
@article{petharbor2025,
  title={PetHarbor: Privacy-Preserving Anonymization for Veterinary Electronic Health Records},
  author={[]},
  journal={},
  year={}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.