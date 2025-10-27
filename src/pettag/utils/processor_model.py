import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm
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
        cache_dir="./embedding_cache",
        cache_name="icd_embeddings.npz",
    ):
        self.model = model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.disease_code_lookup = disease_code_lookup
        self.ner_pipeline = model
        self.embedding_model = embedding_model

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, cache_name)

        logger.info("Initializing ICD embeddings lookup...")
        self.lookup_embeddings, self.codes, self.z_code_mask, self.parent_to_subcodes = (
            self._load_or_compute_lookup()
        )

    # -----------------------------------------------------
    # Compute or load cached embeddings
    # -----------------------------------------------------
    def _load_or_compute_lookup(self):
        if os.path.exists(self.cache_path):
            logger.info(f"Loading cached ICD embeddings from {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=False)
            lookup_embeddings = torch.tensor(
                data["embeddings"], dtype=torch.float32, device=self.device
            )
            codes = data["codes"]
            z_code_mask = data["z_mask"]
            parent_to_subcodes = {p: list(s) for p, s in data["parent_to_subcodes"].item().items()}
            return (
                F.normalize(lookup_embeddings, dim=1),
                codes,
                z_code_mask,
                parent_to_subcodes,
            )

        logger.info("No cache found — computing ICD embeddings from scratch...")

        # Get ICD codes and precomputed embeddings if available
        codes = np.array(self.disease_code_lookup["Code"])
        if "embeddings" in self.disease_code_lookup:
            all_embeddings = np.vstack(self.disease_code_lookup["embeddings"])
        else:
            logger.info("Encoding ICD Titles using embedding model...")
            all_embeddings = self.embedding_model.encode(
                list(self.disease_code_lookup["Title"]),
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32,
            )

        # Normalize and save cache
        normalized = F.normalize(
            torch.tensor(all_embeddings, dtype=torch.float32), dim=1
        ).cpu().numpy()

        z_code_mask = np.array([str(c).endswith(".Z") for c in codes])
        parent_to_subcodes = self._build_parent_subcodes(codes)

        np.savez_compressed(
            self.cache_path,
            embeddings=normalized,
            codes=codes,
            z_mask=z_code_mask,
            parent_to_subcodes=parent_to_subcodes,
        )
        logger.info(f"Cached embeddings saved to {self.cache_path}")

        lookup_embeddings = torch.tensor(normalized, dtype=torch.float32, device=self.device)
        return lookup_embeddings, codes, z_code_mask, parent_to_subcodes

    # -----------------------------------------------------
    # Build mapping of parent -> subcodes
    # -----------------------------------------------------
    def _build_parent_subcodes(self, codes):
        parent_map = {}
        for idx, code in enumerate(codes):
            parent = code.split(".")[0]
            if parent not in parent_map:
                parent_map[parent] = []
            if "." in code:  # It's a subcode
                parent_map[parent].append(idx)
        return parent_map

    # -----------------------------------------------------
    # Batch disease coding (fast vectorized)
    # -----------------------------------------------------
    def disease_coder_batch(self, diseases, Z_BOOST=0.06):
        if not diseases:
            return []

        encoded_diseases = self.embedding_model.encode(
            diseases, convert_to_numpy=True, show_progress_bar=False, batch_size=64
        )
        encoded_diseases = F.normalize(
            torch.tensor(encoded_diseases, dtype=torch.float32, device=self.device), dim=1
        )

        similarities = torch.matmul(encoded_diseases, self.lookup_embeddings.T)

        results = []
        for i, disease in enumerate(diseases):
            sims = similarities[i]
            top_idx = int(torch.argmax(sims).item())
            top_score = float(sims[top_idx].item())

            top_entry = self.disease_code_lookup.iloc[top_idx]
            top_code = top_entry["Code"]
            parent_code = top_code.split(".")[0]

            final_entry, final_score = top_entry, top_score

            if top_code == parent_code and parent_code in self.parent_to_subcodes:
                sub_idxs = self.parent_to_subcodes[parent_code]
                sub_sims = sims[sub_idxs].clone()
                z_mask_sub = torch.tensor(self.z_code_mask[sub_idxs], device=self.device)
                sub_sims[z_mask_sub] += Z_BOOST

                sub_idx = int(torch.argmax(sub_sims).item())
                final_entry = self.disease_code_lookup.iloc[sub_idxs[sub_idx]]
                final_score = float(sub_sims[sub_idx].item())

            results.append(
                {
                    "Title": final_entry["Title"],
                    "Code": final_entry["Code"],
                    "ChapterNo": final_entry["ChapterNo"],
                    "Foundation URI": f'https://icd.who.int/browse/2025-01/mms/en#{final_entry["URI"]}',
                    "Similarity": float(final_score),
                    "Input Disease": disease,
                }
            )

        return results

    def disease_coder(self, disease, Z_BOOST=0.06):
        return self.disease_coder_batch([disease], Z_BOOST)[0]

    # -----------------------------------------------------
    # Batch NER + coding
    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [str(t).lower() for t in examples[self.text_column]]
        ner_results = self.ner_pipeline(texts)

        all_diseases, disease_indices = [], []
        batch_diseases, batch_pathogens, batch_symptoms = [], [], []

        for doc_idx, doc_result in enumerate(ner_results):
            diseases, pathogens, symptoms = set(), set(), set()
            for entity in doc_result:
                label = entity["entity_group"]
                word = entity["word"]
                if label == "DISEASE":
                    diseases.add(word)
                elif label == "SYMPTOM":
                    symptoms.add(word)
                elif label == "ETIOLOGY":
                    pathogens.add(word)

            start_idx = len(all_diseases)
            all_diseases.extend(diseases)
            disease_indices.append((start_idx, len(all_diseases)))
            batch_pathogens.append(list(pathogens))
            batch_symptoms.append(list(symptoms))

        if all_diseases:
            all_coded = self.disease_coder_batch(all_diseases)
            for start, end in disease_indices:
                batch_diseases.append(all_coded[start:end])
        else:
            batch_diseases = [[] for _ in texts]

        return {
            "disease_extraction": batch_diseases,
            "pathogen_extraction": batch_pathogens,
            "symptom_extraction": batch_symptoms,
            self.label_column: batch_diseases,
        }

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------
    def predict(self, dataset, batch_size=32):
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with logging_redirect_tqdm():
            processed_dataset = dataset.map(
                self._process_batch,
                batched=True,
                batch_size=batch_size,
                desc=f"[{date_time} | INFO | PetHarbor-Advance]",
            )
        logger.info("Predictions obtained and text coded successfully.")
        return processed_dataset
