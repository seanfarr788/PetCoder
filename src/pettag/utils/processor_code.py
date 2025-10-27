from pettag.utils.logging_setup import get_logger

logger = get_logger()

from transformers import pipeline
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd
import torch


class ModelProcessor:
    def __init__(
        self,
        model: str,
        tokenizer: str = None,
        replaced: bool = True,
        text_column: str = "text",
        label_column: str = "predictions",
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.device = device

        logger.info("Initializing NER pipeline")
        self.ner_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )

        logger.info(f"Tag map: {self.tag_map}")


    def _process_batch(self, examples):
        original_texts = examples[self.text_column]
        lower_texts = [str(text).lower() for text in original_texts]  # lowercase for NER

        try:
            # Run NER on lowercased texts
            ner_results = self.ner_pipeline(lower_texts)
        except Exception as e:
            logger.error(f"Error during NER pipeline processing: {e}")
            raise

        anonymized_texts = []
        for i, entities in enumerate(ner_results):
            text = original_texts[i]  # use original case for replacements
            for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
                tag = self.tag_map.get(entity["entity_group"])
                if tag:
                    text = self.replace_token(text, entity["start"], entity["end"], tag)
            anonymized_texts.append(text)

        # Return anonymized or raw NER results as before
        if self.replaced is True:
            return {self.text_column: anonymized_texts}
        elif self.replaced is False:
            return {self.label_column: ner_results}
        else:
            return {self.label_column: ner_results, self.text_column: anonymized_texts}


    def anonymise(self, dataset, replace=True):
        """Apply NER-based anonymisation to a dataset.

        args:
            dataset (Dataset): The dataset to process.
            replace (bool): Whether to replace the text with anonymised text or not.
        """
        self.replaced = replace
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with logging_redirect_tqdm():
            processed_dataset = dataset.map(
                self._process_batch,
                batched=True,
                desc=f"[{date_time} |   INFO  | PetHarbor-Advance]",
            )
        logger.info("Predictions obtained and text anonymised successfully")
        return processed_dataset
