from pettag.utils.dataset import DatasetProcessor
from pettag.utils.processor_model import ModelProcessor
from pettag.utils.logging_setup import get_logger
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from collections import defaultdict

from typing import Optional, Dict, Any
import torch
import logging
import pandas as pd


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
        code_lookup="seanfarrell/ICD-11_synonyms",
        embedding_model: str = "sentence-transformers/embeddinggemma-300m-medical",
        text_column: str = "text",  # Column name in the dataset containing text data
        label_column: str = "labels",  # Column name in the dataset containing labels
        cache: bool = True,  # Whether to use cache
        cache_path: str = "petharbor_cache/",  # Path to save cache files
        logs: Optional[str] = None,  # Path to save logs
        device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = None,  # Directory to save the output files
    ):
        self.dataset = dataset
        self.split = split
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.cache = cache
        self.cache_path = cache_path
        self.logs = logs
        self.device = device
        self.output_dir = output_dir

        self.logger = self._setup_logger()
        logger = logging.getLogger(__name__)
        self.dataset_processor = DatasetProcessor(cache_path=self.cache_path)
        logger.info("Initializing NER pipeline")
        self.model = pipeline(
            "token-classification",
            model=model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        logger.info("Initializing embedding pipeline")
        self.embedding_model = SentenceTransformer(embedding_model).to(self.device)
        self.code_lookup = load_dataset(code_lookup, split="train")

        logger.info("Initializing ModelProcessor")
        self.model_processor = ModelProcessor(
            model=self.model,
            tokenizer=self.tokenizer,
            text_column=self.text_column,
            label_column=self.label_column,
            disease_code_lookup=self.code_lookup,
            embedding_model=self.embedding_model,
            device=self.device,
        )
        self.num_runs = 0

    def __repr__(self):
        return f"<Anonymiser model={self.model} dataset={self.dataset} device={self.device}>"

    def _setup_logger(self) -> Any:
        return (
            get_logger(log_dir=self.logs)
            if self.logs
            else get_logger()
        )

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

        target_dataset = self.model_processor.predict(dataset=target_dataset)
        if text:
            print(target_dataset)
            self._print_output(text, target_dataset[0])
            return target_dataset[0]

        else:
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
