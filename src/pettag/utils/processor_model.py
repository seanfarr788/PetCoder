from pettag.utils.logging_setup import get_logger

from collections import defaultdict


import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from functools import lru_cache
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import pipeline
import logging



import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from functools import lru_cache
from tqdm.contrib.logging import logging_redirect_tqdm

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
        # Auto-detect multi-GPU setup
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                logger.info(f"Detected {n_gpus} GPUs — using DataParallel.")
                self.device = "cuda"
                model = torch.nn.DataParallel(model)
            else:
                self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.disease_code_lookup = disease_code_lookup
        self.embedding_model = embedding_model
        self.ner_pipeline = model
        self.logger = get_logger()

        self.logger.info("Precomputing ICD embeddings and lookup tables...")

        # Preload ICD embeddings tensor
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

        # Build parent → subcode map
        parent_groups = {}
        for idx, code in enumerate(self.codes):
            parent = code.split(".")[0]
            parent_groups.setdefault(parent, []).append(idx)

        self.parent_to_subcodes = {
            p: torch.tensor(v, device=self.device, dtype=torch.long)
            for p, v in parent_groups.items()
            if len(v) > 1
        }

        # Boolean tensor mask for ".Z" codes
        self.z_code_mask = torch.tensor(
            [str(c).endswith(".Z") for c in self.codes],
            dtype=torch.bool,
            device=self.device,
        )
        self.logger.info(f"Loaded {self.num_codes} ICD codes on {self.device}.")

        # Compile the high-frequency function (requires PyTorch 2.1+)
        self._compiled_disease_coder_batch = torch.compile(
            self._disease_coder_batch_core, dynamic=True
        )

    # -----------------------------------------------------
    # Cached encoding (massive speed gain on repeated text)
    # -----------------------------------------------------
    @lru_cache(maxsize=4096)
    def _cached_encode(self, text):
        emb = self.embedding_model.encode(
            [text],
            convert_to_tensor=True,
            device=self.device if torch.cuda.is_available() else None,
            show_progress_bar=False,
        )
        return F.normalize(emb, dim=1)[0]

    # -----------------------------------------------------
    # Core vectorized batch logic (compiled)
    # -----------------------------------------------------
    def _disease_coder_batch_core(self, encoded, Z_BOOST):
        sims = F.linear(encoded, self.lookup_embeddings)  # cosine similarities
        top_scores, top_idx = sims.max(dim=1)

        final_idx = top_idx.clone()
        final_score = top_scores.clone()

        # Vectorized subcode refinement
        for parent_code, sub_idx in self.parent_to_subcodes.items():
            mask = torch.tensor(
                [self.codes[i].split(".")[0] == parent_code for i in top_idx],
                device=self.device,
            )
            if mask.any():
                sub_sims = sims[mask][:, sub_idx]
                sub_sims = sub_sims + (self.z_code_mask[sub_idx] * Z_BOOST)
                best_sub = sub_sims.argmax(dim=1)
                final_idx[mask] = sub_idx[best_sub]
                final_score[mask] = sub_sims.gather(1, best_sub.unsqueeze(1)).squeeze()

        return final_idx, final_score

    # -----------------------------------------------------
    # Public disease coding interface
    # -----------------------------------------------------
    @torch.inference_mode()
    def disease_coder_batch(self, diseases, Z_BOOST=0.06):
        if not diseases:
            return []

        # Encode all diseases (GPU preferred)
        try:
            encoded = self.embedding_model.encode(
                diseases,
                convert_to_tensor=True,
                device=self.device if torch.cuda.is_available() else None,
                batch_size=128,  # increase for large GPUs
                show_progress_bar=False,
            )
        except Exception:
            # fallback to cached encode (for CPU-only models)
            encoded = torch.stack([self._cached_encode(t) for t in diseases]).to(self.device)

        encoded = F.normalize(encoded, dim=1)

        # Run compiled similarity computation
        final_idx, final_score = self._compiled_disease_coder_batch(encoded, Z_BOOST)

        results = []
        for i, disease in enumerate(diseases):
            idx = int(final_idx[i].item())
            score = float(final_score[i].item())
            entry = self.disease_code_lookup[idx]
            results.append(
                {
                    "Title": entry["Title"],
                    "Code": entry["Code"],
                    "ChapterNo": entry["ChapterNo"],
                    "Foundation URI": f'https://icd.who.int/browse/2025-01/mms/en#{entry["URI"]}',
                    "Similarity": score,
                    "Input Disease": disease,
                }
            )
        return results

    def disease_coder(self, disease, Z_BOOST=0.06):
        return self.disease_coder_batch([disease], Z_BOOST)[0]

    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [t.lower() for t in examples[self.text_column]]
        ner_results = self.ner_pipeline(texts)

        all_diseases = []
        disease_spans = []
        batch_symptoms, batch_pathogens = [], []

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
            batch_diseases = [coded[s:e] for s, e in disease_spans]
        else:
            batch_diseases = [[] for _ in range(len(texts))]

        return {
            "disease_extraction": batch_diseases,
            "pathogen_extraction": batch_pathogens,
            "symptom_extraction": batch_symptoms,
            self.label_column: batch_diseases,
        }

    # -----------------------------------------------------
    def predict(self, dataset, batch_size=64):
        with logging_redirect_tqdm():
            date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            processed_dataset = dataset.map(
                self._process_batch,
                batched=True,
                batch_size=batch_size,
                desc=f"[{date_time} | INFO | PetHarbor-Advance]",
            )
        logger.info("Predictions obtained and text coded successfully.")
        return processed_dataset
