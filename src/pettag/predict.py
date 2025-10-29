from pettag.utils.dataset import DatasetProcessor
from pettag.utils.processor_model import ModelProcessor
from pettag.utils.logging_setup import get_logger
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from collections import defaultdict
from typing import Optional, Dict, Any, Union
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
from datasets import Dataset, load_dataset
from functools import lru_cache


class DiseaseCoder:
    """Anonymises text data using a pre-trained model.

    Args:
        framework (str): Coding framework ('icd11', 'icd10', 'snomed'). Defaults to 'icd11'.
        dataset (Union[str, Dataset], optional): Path to dataset file or HuggingFace Dataset.
        split (str): Dataset split to use. Defaults to 'train'.
        model (str): HuggingFace model path. Defaults to 'seanfarrell/bert-base-uncased'.
        tokenizer (str, optional): Tokenizer path. Defaults to model if not provided.
        batch_size (int): Batch size for processing. Defaults to 16.
        synonyms_dataset (str): Path to synonyms dataset.
        synonyms_embeddings_dataset (str): Path to precomputed embeddings.
        embedding_model (str): Sentence transformer model name.
        text_column (str): Column containing text. Defaults to 'text'.
        label_column (str): Column containing labels. Defaults to 'labels'.
        cache (bool): Whether to use caching. Defaults to True.
        cache_path (str): Cache directory. Defaults to 'petharbor_cache/'.
        logs (str, optional): Path to save logs.
        device (str, optional): Computation device. Auto-detected if None.
        output_dir (str, optional): Output directory for results.
    """

    def __init__(
        self,
        framework: str = 'icd11',
        dataset: Optional[Union[str, Dataset]] = None,
        split: str = "train",
        model: str = "seanfarrell/bert-base-uncased",
        tokenizer: Optional[str] = None,
        batch_size: int = 16,
        synonyms_dataset: str = "seanfarrell/ICD-11_synonyms",
        synonyms_embeddings_dataset: str = "cache/ICD-11_synonyms_embeddings.pt",
        embedding_model: str = "sentence-transformers/embeddinggemma-300m-medical",
        text_column: str = "text",
        label_column: str = "labels",
        cache: bool = True,
        cache_path: str = "petharbor_cache/",
        logs: Optional[str] = None,
        device: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        # Core attributes
        self.framework = framework
        self.dataset = dataset
        self.split = split
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.cache = cache
        self.cache_path = cache_path
        self.logs = logs
        self.output_dir = output_dir
        self.num_runs = 0
        
        # Device setup with optimization
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Logger setup
        self.logger = self._setup_logger()
        
        # Dataset processor (lazy init if needed)
        self.dataset_processor = DatasetProcessor(cache_path=self.cache_path)
        
        # Model initialization with optimizations
        self.logger.info(f"Initializing NER pipeline on {self.device}")
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        
        self.model = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer or model,
            aggregation_strategy="simple",
            device=self.device,
            batch_size=batch_size,  # Enable batching in pipeline
            dtype=dtype,
        )
        
        # Embedding model initialization
        self.logger.info("Initializing embedding pipeline")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        # Use compile for PyTorch 2.0+ speedup
        if hasattr(torch, 'compile') and "cuda" in self.device:
            try:
                self.embedding_model = torch.compile(self.embedding_model, mode="reduce-overhead")
            except Exception as e:
                self.logger.warning(f"Could not compile embedding model: {e}")
        
        # ICD lookup initialization
        icd_data = None
        if framework:
            icd_data = self._load_or_create_icd_lookup(
                synonyms_dataset, synonyms_embeddings_dataset
            )
            self.logger.info(f"Initialized {framework.upper()} ModelProcessor")
        else:
            self.logger.warning("framework=None specified. NER extraction only.")
        
        # Model processor
        self.model_processor = ModelProcessor(
            framework=self.framework,
            model=self.model,
            icd_embedding=icd_data,
            text_column=text_column,
            label_column=label_column,
            embedding_model=self.embedding_model,
            device=self.device,
            batch_size=batch_size,
        )

    def _setup_logger(self) -> Any:
        """Setup logger with lazy import."""
        return get_logger(log_dir=self.logs) if self.logs else get_logger()

    def _load_or_create_icd_lookup(
        self, synonyms_dataset: str, synonyms_embeddings_dataset: str
    ) -> Dict[str, Any]:
        """Load precomputed embeddings or create new ones."""
        try:
            self.logger.info(f"Loading ICD lookup from {synonyms_embeddings_dataset}")
            data = torch.load(
                synonyms_embeddings_dataset,
                map_location=self.device,
                weights_only=True
            )
            # Move tensors to device efficiently
            return {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }
        except (FileNotFoundError, Exception) as e:
            self.logger.info(
                "Generating new ICD embedding store (first run only, may take a few minutes)"
            )
            icd_dataset = load_dataset(
                synonyms_dataset,
                split="train",
                download_mode='force_redownload'
            )
            return self._preprocess_icd_lookup(icd_dataset, synonyms_embeddings_dataset)

    def _preprocess_icd_lookup(
        self, disease_code_lookup, save_path: str = "icd_lookup.pt"
    ) -> Dict[str, Any]:
        """
        Preprocess ICD lookup into optimized PyTorch format.
        
        Returns normalized embeddings with metadata for fast lookup.
        """
        self.logger.info("Preprocessing ICD lookup into PyTorch format...")

        # Extract and normalize embeddings efficiently
        embeddings = torch.tensor(
            np.asarray(disease_code_lookup["embeddings"], dtype=np.float32),
            dtype=torch.float32
        )
        embeddings = F.normalize(embeddings, dim=1)
        self.logger.info(f"Embeddings shape: {embeddings.shape}")

        # Extract metadata (vectorized where possible)
        metadata_fields = {
            "icd11Code": "icd11Code",
            "icd11Title": "icd11Title",
            "Title_synonym": "Title_synonym",
            "icd11URI": "icd11URI",
            "ChapterNo": "ChapterNo",
            "icd10Code": "icd10Code",
            "icd10Title": "icd10Title",
            "snomedCode": "snomedCode",
            "snomedTitle": "snomedTitle",
        }
        
        metadata = {
            key: [str(c) for c in disease_code_lookup[field]]
            for key, field in metadata_fields.items()
        }

        # Build parent → subcode mapping efficiently
        icd11_codes = metadata["icd11Code"]
        parent_groups = {}
        for idx, code in enumerate(icd11_codes):
            parent = code.split(".")[0]
            parent_groups.setdefault(parent, []).append(idx)

        parent_to_subcodes_keys = []
        parent_to_subcodes_values = []
        for parent, indices in parent_groups.items():
            if len(indices) > 1:
                parent_to_subcodes_keys.append(parent)
                parent_to_subcodes_values.append(torch.tensor(indices, dtype=torch.long))

        # Create .Z code mask
        z_code_mask = torch.tensor(
            [c.endswith(".Z") for c in icd11_codes],
            dtype=torch.bool
        )

        # Ensure save path
        save_path = save_path if save_path.endswith(".pt") else f"{save_path}.pt"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Create lookup dictionary
        lookup_dict = {
            "lookup_embeddings": embeddings.cpu(),
            "parent_to_subcodes_keys": parent_to_subcodes_keys,
            "parent_to_subcodes_values": parent_to_subcodes_values,
            "z_code_mask": z_code_mask.cpu(),
            **metadata,
        }

        torch.save(lookup_dict, save_path)
        self.logger.info(f"ICD lookup saved to {save_path}")

        # Return device-ready dictionary
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in lookup_dict.items()
        }

    @lru_cache(maxsize=128)
    def _prepare_single_text_cached(self, text: str) -> Dataset:
        """Cache single text preparations for repeated calls."""
        df = pd.DataFrame({self.text_column: [text.strip()]})
        return Dataset.from_pandas(df)

    def _prepare_single_text(self, text: str) -> Dataset:
        """Prepare single text input with validation."""
        if not isinstance(text, str):
            error_msg = "Input text must be a string."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self._prepare_single_text_cached(text)

    def _prepare_dataset(self) -> tuple[Dataset, Dataset]:
        """Load and validate dataset with caching."""
        if isinstance(self.dataset, Dataset):
            original_data = self.dataset
        elif isinstance(self.dataset, str):
            original_data = self.dataset_processor.load_dataset_file(
                self.dataset, split=self.split
            )
        else:
            raise ValueError("`dataset` must be a filepath or HuggingFace Dataset.")

        validated = self.dataset_processor.validate_dataset(
            dataset=original_data, text_column=self.text_column
        )
        
        return self.dataset_processor.load_cache(dataset=validated, cache=self.cache)

    def _print_output(self, input_text: str, output_data: Any) -> None:
        """Print formatted output."""
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        pretty_output = json.dumps(
            output_data, indent=4, ensure_ascii=False, sort_keys=True
        )
        print(f"[{timestamp} | SUCCESS | PetCoder] Input:  {input_text}")
        print(f"[{timestamp} | SUCCESS | PetCoder] Output:\n{pretty_output}")

    def _run(self, text: Optional[str] = None, dataset: Optional[str] = None) -> Any:
        """
        Internal method to code text or dataset.
        
        Args:
            text: Single text string to code
            dataset: Path to dataset file
            
        Returns:
            Coded output for text, None for dataset (saves to file)
        """
        self.num_runs += 1
        
        # Validation
        if text and self.dataset:
            raise ValueError("Provide either text or dataset, not both.")
        
        # Prepare input
        if text:
            self.logger.warning("Coding single text. Use dataset for bulk processing.")
            target_dataset = self._prepare_single_text(text)
            result = self.model_processor.single_predict(dataset=target_dataset)
            self._print_output(text, result[0])
            return result[0]
        
        # Dataset processing
        if dataset:
            self.dataset = dataset
        
        if not self.dataset:
            raise ValueError("Provide either text string or dataset path.")
        
        completed_dataset, target_dataset = self._prepare_dataset()
        processed = self.model_processor.predict(dataset=target_dataset)
        
        self.dataset_processor.save_dataset_file(
            target_dataset=processed,
            completed_dataset=completed_dataset,
            cache=self.cache,
            output_dir=self.output_dir,
        )
        return None

    def predict(
        self, text: Optional[str] = None, dataset: Optional[str] = None
    ) -> Optional[Any]:
        """
        Predict on provided text or dataset.
        
        Args:
            text: Single text string to code
            dataset: Path to dataset file
            
        Returns:
            Coded output for text, None for dataset operations
        """
        if self.num_runs > 1:
            self.logger.warning(
                "Sequential model runs detected. Consider passing 'dataset' to class init."
            )

        dataset = dataset or self.dataset
        
        if text is not None:
            return self._run(text=text, dataset=None)
        elif dataset is not None:
            self._run(text=None, dataset=dataset)
            return None
        else:
            self.logger.warning("No text or dataset provided.")
            return None

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[Any]:
        """
        Efficiently predict on multiple texts at once.
        
        Args:
            texts: List of text strings to code
            
        Returns:
            List of coded outputs
        """
        if not texts:
            self.logger.warning("Empty text list provided.")
            return []
        
        # Create temporary dataset
        df = pd.DataFrame({self.text_column: texts})
        temp_dataset = Dataset.from_pandas(df)
        
        # Process in batches
        results = self.model_processor.predict(dataset=temp_dataset)
        return results
    
if __name__ == "__main__":
    # Example usage
    coder = DiseaseCoder()

    text = "Cookie present with vomiting and diarrhea. Suspected gastroenteritis."
    coder.predict(text.upper())