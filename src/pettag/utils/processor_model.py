from pettag.utils.logging_setup import get_logger
import torch
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.nn import functional as F
import pandas as pd
import numpy as np


class ModelProcessor:
    def __init__(
        self,
        model,
        icd_embedding="icd_lookup.pt",  # new default filename
        replaced=True,
        text_column="text",
        label_column="ICD_11_code",
        embedding_model=None,
        device=None,
    ):
        self.model = model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.embedding_model = embedding_model
        self.device = (
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.logger = get_logger()

        self.lookup_embeddings = icd_embedding["lookup_embeddings"].to(self.device)
        self.codes = icd_embedding["codes"]
        self.titles = icd_embedding.get("titles", None)
        self.chapters = icd_embedding.get("chapters", None)
        self.uris = icd_embedding.get("uris", None)
        self.num_codes = len(self.codes)

        self.parent_to_subcodes = {
            k: v.to(self.device)
            for k, v in zip(
                icd_embedding["parent_to_subcodes_keys"],
                icd_embedding["parent_to_subcodes_values"],
            )
        }

        self.z_code_mask = icd_embedding["z_code_mask"].to(self.device)
        self.logger.info(f"✅ Loaded {self.num_codes} ICD codes on {self.device}.")

    # -----------------------------------------------------
    # Internal similarity computation
    # -----------------------------------------------------
    def _disease_coder_batch(self, encoded, Z_BOOST):
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
        """
        Encode a batch of disease descriptions and return the most similar ICD entries.

        Parameters
        ----------
        diseases : list[str]
            A list of free-text disease descriptions to map to ICD codes.
        Z_BOOST : float, optional
            A boost applied to ".Z" codes to slightly prioritize terminal classifications.

        Returns
        -------
        list[dict]
            A list of dictionaries containing ICD metadata and similarity scores for each input disease.
        """
        if not diseases:
            return []

        # -------------------------------------------------------------------------
        # ✅ Step 1. Encode input diseases using the embedding model
        # -------------------------------------------------------------------------
        encoded = self.embedding_model.encode(
            diseases,
            convert_to_tensor=True,
            device=self.device if torch.cuda.is_available() else None,
            batch_size=128,
            show_progress_bar=False,
        )
        encoded = F.normalize(encoded, dim=1)

        # -------------------------------------------------------------------------
        # ✅ Step 2. Perform vector similarity search
        # -------------------------------------------------------------------------
        final_idx, final_score = self._disease_coder_batch(encoded, Z_BOOST)

        # -------------------------------------------------------------------------
        # ✅ Step 3. Compile full metadata for top matches
        # -------------------------------------------------------------------------
        results = []
        for i, disease in enumerate(diseases):
            idx = int(final_idx[i].item())
            score = float(final_score[i].item())

            # Retrieve ICD metadata from preloaded lookup
            code = self.codes[idx]
            title = self.titles[idx] if hasattr(self, "titles") else None
            chapter = self.chapters[idx] if hasattr(self, "chapters") else None
            uri = self.uris[idx] if hasattr(self, "uris") else None

            results.append(
                {
                    "Input Disease": disease,
                    "Code": code,
                    "Title": title,
                    "ChapterNo": chapter,
                    "URI": uri,
                    "Similarity": score,
                }
            )

        return results

    def disease_coder(self, disease, Z_BOOST=0.06):
        return self.disease_coder_batch([disease], Z_BOOST)[0]

    # -----------------------------------------------------
    # NER and batch processing
    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [t.lower() for t in examples[self.text_column]]
        ner_results = self.model(texts)
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
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        processed_dataset = dataset.map(
            self._process_batch,
            batched=True,
            batch_size=batch_size,
            desc=f"[{date_time} | INFO | PetCoder]",
            load_from_cache_file=False,
        )
        self.logger.info("Predictions obtained and text coded successfully.")
        return processed_dataset

    def single_predict(self, dataset):
        dataset = dataset.select([0])
        processed = self._process_batch(examples=dataset)
        processed_dataset = {
            self.text_column: dataset[self.text_column],
            "disease_extraction": processed["disease_extraction"],
            "pathogen_extraction": processed["pathogen_extraction"],
            "symptom_extraction": processed["symptom_extraction"],
            self.label_column: processed[self.label_column],
        }
        self.logger.info("Single prediction obtained and text coded successfully.")

        from datasets import Dataset

        processed_dataset = Dataset.from_dict(processed_dataset)
        return processed_dataset
