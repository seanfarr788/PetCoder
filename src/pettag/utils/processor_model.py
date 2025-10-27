import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Persistent embedding cache (for repeated diseases)
# -----------------------------------------------------
class EmbeddingCache:
    def __init__(self, path="disease_cache.pt"):
        self.path = path
        if os.path.exists(path):
            try:
                self.cache = torch.load(path)
                logger.info(f"Loaded embedding cache: {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

    def save(self):
        torch.save(self.cache, self.path)
        logger.info(f"Saved embedding cache ({len(self.cache)} entries) to {self.path}")


# -----------------------------------------------------
# Main ModelProcessor class
# -----------------------------------------------------
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
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        icd_cache_path="icd_lookup_cache.pt",
        embedding_cache_path="disease_cache.pt",
    ):
        self.model = model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.device = device
        self.disease_code_lookup = disease_code_lookup
        self.embedding_model = embedding_model
        self.ner_pipeline = model

        # ---- Load or precompute ICD lookup embeddings ----
        if os.path.exists(icd_cache_path):
            logger.info(f"Loading cached ICD lookup from {icd_cache_path}")
            cache = torch.load(icd_cache_path, map_location=device)
            self.lookup_embeddings = cache["lookup_embeddings"].to(device)
            self.codes = cache["codes"]
            self.parent_to_subcodes = cache["parent_to_subcodes"]
            self.z_code_mask = cache["z_code_mask"]
        else:
            logger.info("Precomputing and caching ICD lookup embeddings...")
            all_embeddings = np.vstack(self.disease_code_lookup["embeddings"])
            lookup_embeddings = torch.tensor(all_embeddings, dtype=torch.float32)
            lookup_embeddings = F.normalize(lookup_embeddings, dim=1)

            codes = np.array(self.disease_code_lookup["Code"])
            parent_to_subcodes = {}
            for idx, code in enumerate(codes):
                parent = code.split(".")[0]
                if parent not in parent_to_subcodes:
                    parent_to_subcodes[parent] = []
                if "." in code:
                    parent_to_subcodes[parent].append(idx)

            z_code_mask = np.array([str(c).endswith(".Z") for c in codes])

            torch.save(
                {
                    "lookup_embeddings": lookup_embeddings.cpu(),
                    "codes": codes,
                    "parent_to_subcodes": parent_to_subcodes,
                    "z_code_mask": z_code_mask,
                },
                icd_cache_path,
            )
            logger.info(f"Saved ICD lookup cache to {icd_cache_path}")

            self.lookup_embeddings = lookup_embeddings.to(device)
            self.codes = codes
            self.parent_to_subcodes = parent_to_subcodes
            self.z_code_mask = z_code_mask

        # ---- Normalize once ----
        self.lookup_embeddings = F.normalize(self.lookup_embeddings, dim=1)
        logger.info(f"Lookup embeddings loaded: {self.lookup_embeddings.shape}")

        # ---- Initialize persistent embedding cache ----
        self.embedding_cache = EmbeddingCache(embedding_cache_path)

    # -----------------------------------------------------
    # Batch disease coder (main speedup)
    # -----------------------------------------------------
    def disease_coder_batch(self, diseases, Z_BOOST=0.06):
        if not diseases:
            return []

        # --- Retrieve cached embeddings first ---
        uncached = [d for d in diseases if d not in self.embedding_cache.cache]
        if uncached:
            new_embs = self.embedding_model.encode(
                uncached,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for d, e in zip(uncached, new_embs):
                self.embedding_cache.set(d, torch.tensor(e, dtype=torch.float32))

        # --- Assemble embeddings ---
        embeddings = torch.stack(
            [self.embedding_cache.get(d) for d in diseases]
        ).to(self.device)
        embeddings = F.normalize(embeddings, dim=1)

        # --- Similarity computation ---
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            similarities = torch.matmul(embeddings, self.lookup_embeddings.T)

        results = []
        for i, disease in enumerate(diseases):
            sims = similarities[i]
            top_idx = int(torch.argmax(sims).item())
            top_score = float(sims[top_idx].item())
            top_entry = self.disease_code_lookup[top_idx]
            top_code = top_entry["Code"]
            parent_code = top_code.split(".")[0]

            final_entry, final_score = top_entry, top_score

            if top_code == parent_code and parent_code in self.parent_to_subcodes:
                subcode_indices = self.parent_to_subcodes[parent_code]
                if subcode_indices:
                    subcode_tensor = torch.tensor(subcode_indices, device=self.device)
                    sub_sims = sims[subcode_tensor].clone()
                    z_mask_sub = torch.tensor(
                        self.z_code_mask[subcode_indices], device=self.device
                    )
                    sub_sims[z_mask_sub] += Z_BOOST
                    sub_idx = int(torch.argmax(sub_sims).item())
                    final_entry = self.disease_code_lookup[subcode_indices[sub_idx]]
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

        # Save cache every batch (optional)
        self.embedding_cache.save()

        return results

    # -----------------------------------------------------
    # Single disease coder
    # -----------------------------------------------------
    def disease_coder(self, disease, Z_BOOST=0.06):
        return self.disease_coder_batch([disease], Z_BOOST)[0]

    # -----------------------------------------------------
    # Optimized batch processing
    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [str(t).lower() for t in examples[self.text_column]]
        ner_results = self.ner_pipeline(texts)

        all_diseases = []
        disease_indices = []
        batch_diseases, batch_pathogens, batch_symptoms = [], [], []

        for doc_idx, doc_result in enumerate(ner_results):
            diseases, pathogens, symptoms = [], [], []

            for entity in doc_result:
                label = entity["entity_group"]
                word = entity["word"]
                if label == "DISEASE":
                    diseases.append(word)
                elif label == "SYMPTOM":
                    symptoms.append(word)
                elif label == "ETIOLOGY":
                    pathogens.append(word)

            diseases = list(set(diseases))
            start_idx = len(all_diseases)
            all_diseases.extend(diseases)
            disease_indices.append((start_idx, len(all_diseases)))

            batch_pathogens.append(list(set(pathogens)))
            batch_symptoms.append(list(set(symptoms)))

        if all_diseases:
            all_coded = self.disease_coder_batch(all_diseases)
            for start_idx, end_idx in disease_indices:
                batch_diseases.append(all_coded[start_idx:end_idx])
        else:
            batch_diseases = [[] for _ in range(len(texts))]

        return {
            "disease_extraction": batch_diseases,
            "pathogen_extraction": batch_pathogens,
            "symptom_extraction": batch_symptoms,
            self.label_column: batch_diseases,
        }

    # -----------------------------------------------------
    # Main prediction function
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
