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
        self.ner_pipeline = model
        self.embedding_model = embedding_model

        # --- Precompute embeddings for lookup table ---
        logger.info("Precomputing and caching ICD embeddings tensor...")
        all_embeddings = np.vstack(self.disease_code_lookup["embeddings"])
        self.lookup_embeddings = torch.tensor(
            all_embeddings, dtype=torch.float32, device=device
        )
        logger.info(f"Lookup embeddings loaded: {self.lookup_embeddings.shape}")

    # -----------------------------------------------------
    # Extract entities from NER output
    # -----------------------------------------------------
    @staticmethod
    def model_extractions(petbert_disease_out):
        disease_out, pathogen_out, symptom_out = [], [], []
        output = []

        for entity in petbert_disease_out[0]:
            start, end, label = entity["start"], entity["end"], entity["entity_group"]
            output.append((start, end, label))
            if label == "DISEASE":
                disease_out.append(entity["word"])
            elif label == "SYMPTOM":
                symptom_out.append(entity["word"])
            elif label == "ETIOLOGY":
                pathogen_out.append(entity["word"])

        # deduplicate
        return list(set(disease_out)), list(set(pathogen_out)), list(set(symptom_out))

    # -----------------------------------------------------
    # Disease coder with efficient cosine similarity
    # -----------------------------------------------------
    def disease_coder(self, disease, Z_BOOST=0.06):
        """Return best ICD code match for an input disease phrase."""

        def get_parent_code(code: str) -> str:
            return code.split(".")[0]

        # --- Step 1: encode disease and normalize ---
        encoded_disease = self.embedding_model.encode(disease, convert_to_numpy=True)
        encoded_tensor = torch.tensor(
            encoded_disease, dtype=torch.float32, device=self.device
        )
        encoded_tensor = F.normalize(encoded_tensor.unsqueeze(0), dim=1)  # shape [1, D]

        # --- Step 2: compute cosine similarities (torch, GPU-accelerated) ---
        similarities = torch.matmul(encoded_tensor, self.lookup_embeddings.T).squeeze(0)
        top_idx = int(torch.argmax(similarities))
        top_score = float(similarities[top_idx])

        top_entry = self.disease_code_lookup[top_idx]
        top_code, top_title = top_entry["Code"], top_entry["Title"]
        parent_code = get_parent_code(top_code)
        final_entry, final_score = top_entry, top_score

        # --- Step 3: check subcodes efficiently ---
        subcodes_mask = [
            str(code).startswith(parent_code + ".")
            for code in self.disease_code_lookup["Code"]
        ]
        subcodes_idx = np.where(subcodes_mask)[0]

        if len(subcodes_idx) > 0 and top_code == parent_code:
            sub_embeddings = self.lookup_embeddings[subcodes_idx]
            sub_sims = torch.matmul(encoded_tensor, sub_embeddings.T).squeeze(0)

            # Apply Z_BOOST for unspecified subcodes
            for i, code in enumerate(
                np.array(self.disease_code_lookup["Code"])[subcodes_idx]
            ):
                if code.endswith(".Z"):
                    sub_sims[i] += Z_BOOST

            sub_idx = int(torch.argmax(sub_sims))
            final_entry = self.disease_code_lookup[subcodes_idx[sub_idx]]
            final_score = float(sub_sims[sub_idx])

        return {
            "Title": final_entry["Title"],
            "Code": final_entry["Code"],
            "ChapterNo": final_entry["ChapterNo"],
            "Foundation URI": final_entry["Foundation URI"],
            "Similarity": float(final_score),
            "Input Disease": disease,
        }

    # -----------------------------------------------------
    # Batch processing
    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [str(t).lower() for t in examples[self.text_column]]
        ner_results = self.ner_pipeline(texts)

        # NOTE: assuming each NER output is a list of entities per text
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

            coded_diseases = [self.disease_coder(d) for d in diseases]
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
                desc=f"[{date_time} | INFO | PetHarbor-Advance]",
            )
        logger.info("Predictions obtained and text coded successfully.")
        return processed_dataset
