from pettag.utils.logging_setup import get_logger
logger = get_logger()

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
        device=None,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.disease_code_lookup = disease_code_lookup
        self.embedding_model = embedding_model
        self.ner_pipeline = model
        self.logger = get_logger()

        self.logger.info("Precomputing ICD embeddings and lookup tables...")

        # Preload embeddings as normalized tensor
        self.lookup_embeddings = F.normalize(
            torch.as_tensor(
                np.stack(disease_code_lookup["embeddings"]),
                dtype=torch.float32,
                device=self.device,
            ),
            dim=1,
        )

        self.codes = np.array(disease_code_lookup["Code"])
        self.num_codes = len(self.codes)

        # Build parent→subcode tensor map for vectorized lookup
        parent_groups = {}
        for idx, code in enumerate(self.codes):
            parent = code.split(".")[0]
            parent_groups.setdefault(parent, []).append(idx)

        self.parent_to_subcodes = {
            p: torch.tensor(v, device=self.device, dtype=torch.long)
            for p, v in parent_groups.items()
            if len(v) > 1
        }

        # Boolean tensor for ".Z" codes (kept on GPU)
        self.z_code_mask = torch.tensor(
            [str(c).endswith(".Z") for c in self.codes],
            dtype=torch.bool,
            device=self.device,
        )

        self.logger.info(f"Loaded {self.num_codes} ICD codes on {self.device}.")

    # -----------------------------------------------------
    # Fast batched disease coding
    # -----------------------------------------------------
    @torch.inference_mode()
    def disease_coder_batch(self, diseases, Z_BOOST=0.06):
        if not diseases:
            return []

        # Encode all diseases together (stays on GPU)
        encoded = self.embedding_model.encode(
            diseases,
            convert_to_tensor=True,
            device=self.device,
            batch_size=64,
            show_progress_bar=False,
        )
        encoded = F.normalize(encoded, dim=1)

        # Cosine similarity using efficient F.linear
        sims = F.linear(encoded, self.lookup_embeddings)  # [N, num_codes]

        # Initial top-1 predictions
        top_scores, top_idx = sims.max(dim=1)
        results = []

        for i, disease in enumerate(diseases):
            idx = top_idx[i].item()
            score = top_scores[i].item()
            top_code = self.codes[idx]
            parent_code = top_code.split(".")[0]
            final_idx = idx
            final_score = score

            # Vectorized subcode check
            if top_code == parent_code and parent_code in self.parent_to_subcodes:
                sub_indices = self.parent_to_subcodes[parent_code]
                sub_sims = sims[i, sub_indices]
                sub_sims = sub_sims + (self.z_code_mask[sub_indices] * Z_BOOST)
                sub_best = sub_sims.argmax()
                final_idx = sub_indices[sub_best].item()
                final_score = sub_sims[sub_best].item()

            entry = self.disease_code_lookup.iloc[final_idx]
            results.append(
                {
                    "Title": entry["Title"],
                    "Code": entry["Code"],
                    "ChapterNo": entry["ChapterNo"],
                    "Foundation URI": f'https://icd.who.int/browse/2025-01/mms/en#{entry["URI"]}',
                    "Similarity": float(final_score),
                    "Input Disease": disease,
                }
            )
        return results

    # -----------------------------------------------------
    def disease_coder(self, disease, Z_BOOST=0.06):
        return self.disease_coder_batch([disease], Z_BOOST)[0]

    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [t.lower() for t in examples[self.text_column]]
        ner_results = self.ner_pipeline(texts)

        all_diseases = []
        disease_spans = []
        batch_symptoms, batch_pathogens = [], [],

        for doc in ner_results:
            diseases, symptoms, pathogens = set(), set(), set()
            for ent in doc:
                label = ent["entity_group"]
                word = ent["word"]
                if label == "DISEASE":
                    diseases.add(word)
                elif label == "SYMPTOM":
                    symptoms.add(word)
                elif label == "ETIOLOGY":
                    pathogens.add(word)

            start = len(all_diseases)
            all_diseases.extend(diseases)
            disease_spans.append((start, len(all_diseases)))
            batch_symptoms.append(list(symptoms))
            batch_pathogens.append(list(pathogens))

        if all_diseases:
            coded = self.disease_coder_batch(all_diseases)
            batch_diseases = [
                coded[s:e] for (s, e) in disease_spans
            ]
        else:
            batch_diseases = [[] for _ in range(len(texts))]

        return {
            "disease_extraction": batch_diseases,
            "pathogen_extraction": batch_pathogens,
            "symptom_extraction": batch_symptoms,
            self.label_column: batch_diseases,
        }

    # -----------------------------------------------------
    def predict(self, dataset, batch_size=32):
        with logging_redirect_tqdm():
            date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            return dataset.map(
                self._process_batch,
                batched=True,
                batch_size=batch_size,
                desc=f"[{date_time} | INFO | PetHarbor-Advance]",
            )
