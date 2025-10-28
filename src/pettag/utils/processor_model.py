from pettag.utils.logging_setup import get_logger
import torch
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.nn import functional as F
import pandas as pd
import numpy as np
from datasets import Dataset


class ModelProcessor:
    def __init__(
        self,
        framework,
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
        self.framework = framework
        self.logger = get_logger()

        self.lookup_embeddings = icd_embedding["lookup_embeddings"].to(self.device)
        self.icd11_codes = icd_embedding["icd11Code"]
        self.icd11_titles = icd_embedding.get("icd11Title", None)
        self.title_synonyms = icd_embedding.get("Title_synonym", None)
        self.icd11_uris = icd_embedding.get("icd11URI", None)
        self.chapters = icd_embedding.get("ChapterNo", None)
        self.icd10_codes = icd_embedding.get("icd10Code", None)
        self.icd10_titles = icd_embedding.get("icd10Title", None)
        self.snomed_codes = icd_embedding.get("snomedCode", None)
        self.snomed_titles = icd_embedding.get("snomedTitle", None)

        self.num_codes = len(self.icd11_codes)

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
                [self.icd11_codes[i].split(".")[0] == parent_code for i in top_idx],
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
    def disease_coder_batch(self, diseases, Z_BOOST=0.06, framework=None):
        """
        Encode a batch of disease descriptions and return the most similar ICD/SNOMED entries.

        Parameters
        ----------
        diseases : list[str]
            A list of free-text disease descriptions to map to codes.
        Z_BOOST : float, optional
            A boost applied to ".Z" codes to slightly prioritize terminal classifications.

        Returns
        -------
        list[dict]
            A list of dictionaries containing framework-specific metadata and similarity scores.
        """
        if not diseases:
            return []
        framework = framework or self.framework
        framework = framework.lower()
        valid_frameworks = ["icd11", "icd10", "snomed"]
        if framework not in valid_frameworks:
            raise ValueError(
                f"Invalid framework '{framework}'. Must be one of {valid_frameworks}."
            )

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
        # ✅ Step 3. Compile metadata for top matches according to framework
        # -------------------------------------------------------------------------
        results = []
        for i, disease in enumerate(diseases):
            idx = int(final_idx[i].item())
            score = min(float(final_score[i].item()), 1.0)

            if framework == "icd11":
                code = self.icd11_codes[idx]
                title = (
                    self.icd11_titles[idx] if hasattr(self, "icd11_titles") else None
                )
                chapter = self.chapters[idx] if hasattr(self, "chapters") else None
                uri = self.icd11_uris[idx] if hasattr(self, "icd11_uris") else None

            elif framework == "icd10":
                code = self.icd10_codes[idx] if hasattr(self, "icd10_codes") else None
                title = (
                    self.icd10_titles[idx] if hasattr(self, "icd10_titles") else None
                )
                chapter = self.chapters[idx] if hasattr(self, "chapters") else None
                uri = ""
                synonym = ""

            elif framework == "snomed":
                code = self.snomed_codes[idx] if hasattr(self, "snomed_codes") else None
                title = (
                    self.snomed_titles[idx] if hasattr(self, "snomed_titles") else None
                )
                chapter = ""
                uri = ""
                synonym = ""
                # if the first character of the code is a ( then add warning message that this is actually an ICD 11 code
            if code and str(code.strip()).startswith("("):
                self.logger.warning(
                    f"⚠️ The matched {framework} code '{code}' is an ICD-11 code."
                )

            results.append(
                {
                    "Input Disease": disease,
                    "Framework": framework.upper(),
                    "Code": code,
                    "Title": title,
                    "Chapter": chapter,
                    "URI": (
                        f"https://icd.who.int/browse/2025-01/mms/en#{uri}"
                        if uri
                        else None
                    ),
                    "Similarity": round(score, 4),
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
            # self.label_column: batch_diseases,
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
        return Dataset.from_dict(
            {
                self.text_column: dataset[self.text_column],
                "Code": processed["disease_extraction"],
                "pathogen_extraction": processed["pathogen_extraction"],
                "symptom_extraction": processed["symptom_extraction"],
                # self.label_column: processed[self.label_column],
            }
        )
