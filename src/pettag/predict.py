from pettag.utils.dataset import DatasetProcessor
from pettag.utils.processor_model import ModelProcessor
from pettag.utils.logging_setup import get_logger
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from collections import defaultdict

from torch.nn import functional as F
from typing import Optional, Dict, Any
import torch
import logging
import pandas as pd
import numpy as np
import os


class DiseaseCoder:
    """Anonymises text data using a pre-trained model.

    Args:
        dataset (Union[str, Dataset], optional): Path to dataset file (CSV, Arrow, etc.) or a HuggingFace Dataset.
        split (str): Dataset split to use ('train', 'test', etc.). Defaults to 'train'.
        model (str): HuggingFace model path or name. Defaults to 'SAVSNET/PetHarbor'.
        tokenizer (str, optional): Tokenizer path or name. Defaults to model if not provided.
        text_column (str): Column in dataset containing text. Defaults to 'text'.
        cache (bool): Whether to use caching. Defaults to True.
        cache_path (str): Directory to store cache files. Defaults to 'petharbor_cache/'.
        logs (str, optional): Path to save logs. If None, logs to console.
        device (str, optional): Device to use for computation. Defaults to 'cuda' if available else 'cpu'.
        tag_map (Dict[str, str], optional): Entity tag to replacement string mapping.
        output_dir (str, optional): Directory to save output dataset.
    """

    def __init__(
        self,
        dataset: str = None,  # Path to the dataset file (CSV, Arrow, etc.)
        split: str = "train",  # Split of the dataset to use (e.g., 'train', 'test', 'eval')
        model: str = "seanfarrell/bert-base-uncased",  # Path to the model
        tokenizer: str = None,  # Path to the tokenizer
        synonyms_dataset="seanfarrell/ICD-11_synonyms",
        synonyms_embeddings_dataset="cache/ICD-11_synonyms_embeddings.pt",
        embedding_model: str = "sentence-transformers/embeddinggemma-300m-medical",
        text_column: str = "text",  # Column name in the dataset containing text data
        label_column: str = "labels",  # Column name in the dataset containing labels
        cache: bool = True,  # Whether to use cache
        cache_path: str = "petharbor_cache/",  # Path to save cache files
        logs: Optional[str] = None,  # Path to save logs
        device: Optional[str] = "cuda:0" if torch.cuda.is_available() else "cpu",
        output_dir: str = None,  # Directory to save the output files
    ):
        self.dataset = dataset
        self.split = split
        self.tokenizer = tokenizer if tokenizer else model
        self.text_column = text_column
        self.label_column = label_column
        self.cache = cache
        self.cache_path = cache_path
        self.logs = logs
        self.device = device
        self.output_dir = output_dir
        self.logger = self._setup_logger()

        self.dataset_processor = DatasetProcessor(cache_path=self.cache_path)
        self.logger.info(f"Initializing NER pipeline. Using {device}.")
        self.model = pipeline(
            "token-classification",
            model=model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        self.logger.info("Initializing embedding pipeline")
        self.embedding_model = SentenceTransformer(embedding_model).to(self.device)

        try:
            self.logger.info(
                f"Loading precomputed ICD lookup from {synonyms_embeddings_dataset} ..."
            )
            data = torch.load(
                synonyms_embeddings_dataset, map_location=self.device, weights_only=True
            )
        except:
            self.logger.info(
                "No ICD embedding store found. Generating a new one. This should only happen on first run. Should take a few minutes..."
            )
            icd_lookup_daaset = load_dataset(synonyms_dataset, split="train")
            data = self._preprocess_icd_lookup(
                disease_code_lookup=icd_lookup_daaset,
                save_path=synonyms_embeddings_dataset,
            )

        self.logger.info("Initializing ModelProcessor")
        self.model_processor = ModelProcessor(
            model=self.model,
            icd_embedding=data,
            text_column=self.text_column,
            label_column=self.label_column,
            embedding_model=self.embedding_model,
            device=self.device,
        )
        self.num_runs = 0

    def _setup_logger(self) -> Any:
        return get_logger(log_dir=self.logs) if self.logs else get_logger()

    def _print_output(self, input_text: str, output_text: str):
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp} | SUCCESS | PetCoder] Input: {input_text}")
        print(f"[{timestamp} | SUCCESS | PetCoder] Output: {output_text}")

    def _prepare_single_text(self, text: str) -> Dataset:
        if not isinstance(text, str):
            error_message = "Input text must be a string."
            self.logger.error(error_message)
            raise ValueError(error_message)
        clean_text = text.strip()
        df = pd.DataFrame({self.text_column: [clean_text]})
        return Dataset.from_pandas(df)

        # -----------------------------------------------------

    # Create Embedding Store
    # -----------------------------------------------------

    def _preprocess_icd_lookup(self, disease_code_lookup, save_path="icd_lookup.pt"):
        """
        Preprocess an ICD lookup table into a compact, normalized, and efficiently loadable
        PyTorch dictionary. The resulting .pt file includes:
        - Normalized embeddings
        - Metadata (Code, Title, ChapterNo, URI)
        - Parent–subcode index mappings
        - .Z code mask

        Parameters
        ----------
        disease_code_lookup : pandas.DataFrame or dict-like
            ICD lookup data containing at least the columns:
            ['Code', 'Title', 'ChapterNo', 'URI', 'embeddings'].
        save_path : str, optional
            Output path for the saved lookup file (default: "icd_lookup.pt").

        Returns
        -------
        dict
            A dictionary containing tensors and metadata ready for downstream use.
        """

        self.logger.info("📦 Preprocessing ICD lookup into safe PyTorch format...")

        # -------------------------------------------------------------------------
        # ✅ Step 1. Normalize embeddings
        # -------------------------------------------------------------------------
        embeddings = torch.tensor(
            np.asarray(disease_code_lookup["embeddings"], dtype=np.float32)
        )
        embeddings = F.normalize(embeddings, dim=1)

        # -------------------------------------------------------------------------
        # ✅ Step 2. Extract metadata columns
        # -------------------------------------------------------------------------
        codes = [str(c) for c in disease_code_lookup["Code"]]
        titles = [str(t) for t in disease_code_lookup["Title"]]
        chapters = [str(ch) for ch in disease_code_lookup["ChapterNo"]]
        uris = [str(u) for u in disease_code_lookup["URI"]]

        # -------------------------------------------------------------------------
        # ✅ Step 3. Build parent → subcode mapping
        # -------------------------------------------------------------------------
        parent_groups = {}
        for idx, code in enumerate(codes):
            parent = code.split(".")[0]
            parent_groups.setdefault(parent, []).append(idx)

        parent_to_subcodes_keys, parent_to_subcodes_values = [], []
        for parent, indices in parent_groups.items():
            if len(indices) > 1:
                parent_to_subcodes_keys.append(parent)
                parent_to_subcodes_values.append(
                    torch.tensor(indices, dtype=torch.long)
                )

        # -------------------------------------------------------------------------
        # ✅ Step 4. Create a ".Z" code mask (terminal codes)
        # -------------------------------------------------------------------------
        z_code_mask = torch.tensor([c.endswith(".Z") for c in codes], dtype=torch.bool)

        # -------------------------------------------------------------------------
        # ✅ Step 5. Handle save path and ensure directory exists
        # -------------------------------------------------------------------------
        save_path = save_path if save_path.endswith(".pt") else f"{save_path}.pt"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # -------------------------------------------------------------------------
        # ✅ Step 6. Save lookup dictionary
        # -------------------------------------------------------------------------
        lookup_dict = {
            "lookup_embeddings": embeddings.cpu(),
            "codes": codes,
            "titles": titles,
            "chapters": chapters,
            "uris": uris,
            "parent_to_subcodes_keys": parent_to_subcodes_keys,
            "parent_to_subcodes_values": parent_to_subcodes_values,
            "z_code_mask": z_code_mask,
        }

        torch.save(lookup_dict, save_path)
        self.logger.info(f"💾 ICD lookup saved successfully → {save_path}")

        # -------------------------------------------------------------------------
        # ✅ Step 8. Return device-ready dictionary
        # -------------------------------------------------------------------------
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in lookup_dict.items()
        }

    def _prepare_dataset(self) -> Dataset:
        if isinstance(self.dataset, Dataset):
            original_data = self.dataset
        elif isinstance(self.dataset, str):
            original_data = self.dataset_processor.load_dataset_file(
                self.dataset, split=self.split
            )
        else:
            raise ValueError("`dataset` must be a filepath or a HuggingFace Dataset.")

        validated = self.dataset_processor.validate_dataset(
            dataset=original_data, text_column=self.text_column
        )
        completed_dataset, target_dataset = self.dataset_processor.load_cache(
            dataset=validated, cache=self.cache
        )
        return completed_dataset, target_dataset

    def _run(
        self,
        text: str = None,
        dataset: str = None,
    ) -> None:
        """Coder the single text data or in a dataset and output/saves the results.

        Args:
        text (str, optional): Text to code
        dataset (str, optional): Path to the dataset file (CSV, Arrow, etc.)

        Raises:
        ValueError: If both text and dataset are provided or neither is provided.

        """
        self.num_runs += 1
        if text and self.dataset:
            raise ValueError(
                "Please provide either a text string or a dataset path, not both."
            )
        # Prepare input
        if text:  # If text is provided
            self.logger.warning(
                "Anonymising single text input. For bulk processing, use a dataset."
            )
            target_dataset = self._prepare_single_text(text)
        elif dataset:  # If dataset is provided to class
            self.dataset = dataset
            target_dataset, completed_dataset = self._prepare_dataset()
        elif self.dataset:  # If dataset is initialized
            target_dataset, completed_dataset = self._prepare_dataset()
        else:
            raise ValueError("Please provide either a text string or a dataset path.")
        print("Starting disease coding process...")
        if text:
            target_dataset = self.model_processor.single_predict(dataset=target_dataset)
            self._print_output(text, target_dataset[0])
            return target_dataset[0]
        else:
            target_dataset = self.model_processor.predict(
                dataset=target_dataset, batch_size=16
            )
            self.dataset_processor.save_dataset_file(
                target_dataset=target_dataset,
                completed_dataset=completed_dataset,
                cache=self.cache,
                output_dir=self.output_dir,
            )

    def predict(self, text: str = None, dataset: str = None) -> Optional[str]:
        """
        Anonymises text data.

        n.b 'anonymise' method overwrites the text column if a dataset is provided.
        If only text is provided, the anonymised text is returned.
        """
        if self.num_runs > 1:
            self.logger.warning(
                "It appears you are sequentially running the model. Reccomended: Pass 'dataset' to the class."
            )

        if dataset is None:
            dataset = self.dataset
        if text is not None:
            return self._run(text=text, dataset=None)
        elif dataset is not None:
            self._run(text=None, dataset=dataset)
            return None  # Explicitly return None when operating on the dataset
        else:
            self.logger.warning("No text or dataset provided for anonymisation.")
            return None
