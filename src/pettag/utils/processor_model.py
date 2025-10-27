from pettag.utils.logging_setup import get_logger

logger = get_logger()
from sentence_transformers import SentenceTransformer
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from collections import defaultdict


import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from functools import lru_cache
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class ModelProcessor:
    def __init__(
        self,
        model,
        tokenizer=None,
        replaced=True,
        text_column="text",
        label_column="ICD_11_code",
        disease_code_lookup=None,
        embedding_model=None,
        device="cpu",
    ):
        self.model = model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.device = device
        self.disease_code_lookup = disease_code_lookup
        self.embedding_model = embedding_model

        # Initialize NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if "cuda" in device else -1,
        )

        # --- Precompute embeddings for lookup table ---
        logger.info("Precomputing and caching ICD embeddings tensor...")
        all_embeddings = np.vstack(self.disease_code_lookup["embeddings"])
        self.lookup_embeddings = F.normalize(
            torch.tensor(all_embeddings, dtype=torch.float32, device=device),
            dim=1,
        )
        logger.info(f"Lookup embeddings loaded: {self.lookup_embeddings.shape}")

    # -----------------------------------------------------
    # Cached encoding for repeated disease strings
    # -----------------------------------------------------
    @lru_cache(maxsize=5000)
    def _cached_encode(self, text: str):
        return self.embedding_model.encode(text, convert_to_numpy=True)

    # -----------------------------------------------------
    # Disease coder (batched version)
    # -----------------------------------------------------
    def batch_disease_coder(self, diseases, Z_BOOST=0.06):
        """Return best ICD code matches for a batch of input disease phrases."""

        if not diseases:
            return []

        # --- Step 1: Encode all diseases in batch ---
        encoded_diseases = []
        for d in diseases:
            try:
                encoded_diseases.append(self._cached_encode(d))
            except Exception as e:
                logger.warning(f"Encoding failed for '{d}': {e}")
                encoded_diseases.append(np.zeros(self.lookup_embeddings.shape[1]))

        encoded_diseases = np.stack(encoded_diseases)
        encoded_tensor = torch.tensor(
            encoded_diseases, dtype=torch.float32, device=self.device
        )
        encoded_tensor = F.normalize(encoded_tensor, dim=1)

        # --- Step 2: Compute cosine similarities for all diseases at once ---
        similarities = torch.matmul(encoded_tensor, self.lookup_embeddings.T)
        top_idx = torch.argmax(similarities, dim=1)
        top_scores = torch.max(similarities, dim=1).values

        coded_diseases = []
        for i, (idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist())):
            entry = self.disease_code_lookup[idx]
            parent_code = entry["Code"].split(".")[0]
            final_entry, final_score = entry, score

            # --- Step 3: refine if there are subcodes of this parent ---
            subcodes_mask = [
                str(code).startswith(parent_code + ".")
                for code in self.disease_code_lookup["Code"]
            ]
            subcodes_idx = np.where(subcodes_mask)[0]

            if len(subcodes_idx) > 0 and entry["Code"] == parent_code:
                sub_embeddings = self.lookup_embeddings[subcodes_idx]
                sub_sims = torch.matmul(
                    encoded_tensor[i].unsqueeze(0), sub_embeddings.T
                ).squeeze(0)

                # Apply Z_BOOST for unspecified subcodes
                for j, code in enumerate(
                    np.array(self.disease_code_lookup["Code"])[subcodes_idx]
                ):
                    if code.endswith(".Z"):
                        sub_sims[j] += Z_BOOST

                sub_idx = int(torch.argmax(sub_sims))
                final_entry = self.disease_code_lookup[subcodes_idx[sub_idx]]
                final_score = float(sub_sims[sub_idx])

            coded_diseases.append(
                {
                    "Title": final_entry["Title"],
                    "Code": final_entry["Code"],
                    "ChapterNo": final_entry["ChapterNo"],
                    "Foundation URI": final_entry["Foundation URI"],
                    "Similarity": float(final_score),
                    "Input Disease": diseases[i],
                }
            )

        return coded_diseases

    # -----------------------------------------------------
    # Batch processing
    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [str(t) for t in examples[self.text_column]]
        ner_results = self.ner_pipeline(texts, batch_size=16)

        batch_diseases, batch_pathogens, batch_symptoms = [], [], []

        for doc_result in ner_results:
            diseases, pathogens, symptoms = [], [], []

            for entity in doc_result:
                label = entity["entity_group"]
                if label == "DISEASE":
                    diseases.append(entity["word"])
                elif label == "SYMPTOM":
                    symptoms.append(entity["word"])
                elif label == "ETIOLOGY":
                    pathogens.append(entity["word"])

            # Early skip if no diseases detected
            if not diseases:
                batch_diseases.append([])
                batch_pathogens.append(list(set(pathogens)))
                batch_symptoms.append(list(set(symptoms)))
                continue

            # Batch encode + code diseases
            coded_diseases = self.batch_disease_coder(diseases)
            batch_diseases.append(coded_diseases)
            batch_pathogens.append(list(set(pathogens)))
            batch_symptoms.append(list(set(symptoms)))

        return {
            "disease_extraction": batch_diseases,
            "pathogen_extraction": batch_pathogens,
            "symptom_extraction": batch_symptoms,
            self.label_column: batch_diseases,
        }

    # -----------------------------------------------------
    # Main prediction function
    # -----------------------------------------------------
    def predict(self, dataset):
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with logging_redirect_tqdm():
            processed_dataset = dataset.map(
                self._process_batch,
                batched=True,
                num_proc=4,  # Parallelize across CPU cores
                desc=f"[{date_time} | INFO | PetHarbor-Advance]",
            )
        logger.info("Predictions obtained and text coded successfully.")
        return processed_dataset
