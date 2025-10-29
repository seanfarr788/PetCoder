from pettag.utils.logging_setup import get_logger
import torch
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.nn import functional as F
import pandas as pd
import numpy as np
from datasets import Dataset


class ModelProcessor:
    """Handles NER extraction and disease coding with optimized batch processing."""
    
    def __init__(
        self,
        framework,
        model,
        icd_embedding=None,
        replaced=True,
        text_column="text",
        label_column="ICD_11_code",
        embedding_model=None,
        device=None,
        batch_size=256,
    ):
        from pettag.utils.logging_setup import get_logger
        from tqdm.contrib.logging import logging_redirect_tqdm
        
        self.model = model
        self.replaced = replaced
        self.text_column = text_column
        self.label_column = label_column
        self.embedding_model = embedding_model
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.framework = framework
        self.logger = get_logger()
        
        # Initialize ICD lookup if framework is specified
        if self.framework is not None and icd_embedding is not None:
            # Move embeddings to device with non_blocking for efficiency
            self.lookup_embeddings = icd_embedding["lookup_embeddings"].to(
                self.device, non_blocking=True
            )
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

            # Build parent to subcodes mapping
            self.parent_to_subcodes = {
                k: v.to(self.device, non_blocking=True)
                for k, v in zip(
                    icd_embedding["parent_to_subcodes_keys"],
                    icd_embedding["parent_to_subcodes_values"],
                )
            }

            self.z_code_mask = icd_embedding["z_code_mask"].to(
                self.device, non_blocking=True
            )

            # Pre-compute parent codes for faster matching (optimization)
            self.code_parents = [code.split(".")[0] for code in self.icd11_codes]

            # Pre-identify codes with parentheses (for warnings - optional)
            self.codes_with_paren = {
                i for i, code in enumerate(self.icd11_codes)
                if str(code).strip().startswith("(")
            }
            if self.icd10_codes:
                self.codes_with_paren.update({
                    i for i, code in enumerate(self.icd10_codes)
                    if str(code).strip().startswith("(")
                })
            if self.snomed_codes:
                self.codes_with_paren.update({
                    i for i, code in enumerate(self.snomed_codes)
                    if str(code).strip().startswith("(")
                })

            self.logger.info(f"Loaded {self.num_codes} ICD codes on {self.device}.")

    @torch.inference_mode()
    def _disease_coder_batch(self, encoded, Z_BOOST):
        """Internal optimized similarity computation."""
        # Compute cosine similarities using efficient matrix multiplication
        sims = F.linear(encoded, self.lookup_embeddings)
        top_scores, top_idx = sims.max(dim=1)

        final_idx = top_idx.clone()
        final_score = top_scores.clone()

        # Vectorized subcode refinement
        top_idx_cpu = top_idx.cpu().tolist()
        for parent_code, sub_idx in self.parent_to_subcodes.items():
            # Use pre-computed parent codes instead of runtime string splitting
            mask = torch.tensor(
                [self.code_parents[i] == parent_code for i in top_idx_cpu],
                dtype=torch.bool,
                device=self.device,
            )
            
            if mask.any():
                sub_sims = sims[mask][:, sub_idx]
                # Apply Z-code boost
                sub_sims = sub_sims + (self.z_code_mask[sub_idx] * Z_BOOST)
                best_sub = sub_sims.argmax(dim=1)
                final_idx[mask] = sub_idx[best_sub]
                final_score[mask] = sub_sims.gather(1, best_sub.unsqueeze(1)).squeeze()

        return final_idx, final_score

    @torch.inference_mode()
    def disease_coder_batch(self, diseases, Z_BOOST=0.06, framework=None):
        """
        Encode disease descriptions and return most similar ICD/SNOMED entries.

        Args:
            diseases: List of disease descriptions
            Z_BOOST: Boost for terminal .Z codes
            framework: Override framework (icd11, icd10, snomed)

        Returns:
            List of dictionaries with coding results
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

        # Encode diseases using embedding model
        encoded = self.embedding_model.encode(
            diseases,
            convert_to_tensor=True,
            device=self.device if torch.cuda.is_available() else None,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Let the model normalize if supported
        )
        
        # Ensure normalization (some models may not support normalize_embeddings param)
        if not hasattr(self.embedding_model, 'normalize_embeddings'):
            encoded = F.normalize(encoded, dim=1)

        # Perform similarity search
        final_idx, final_score = self._disease_coder_batch(encoded, Z_BOOST)

        # Compile results
        final_idx_cpu = final_idx.cpu().tolist()
        final_score_cpu = torch.clamp(final_score, max=1.0).cpu().tolist()

        results = []
        for i, disease in enumerate(diseases):
            idx = final_idx_cpu[i]
            score = round(final_score_cpu[i], 4)

            if framework == "icd11":
                code = self.icd11_codes[idx]
                title = self.icd11_titles[idx] if self.icd11_titles else None
                chapter = self.chapters[idx] if self.chapters else None
                uri = self.icd11_uris[idx] if self.icd11_uris else None
            elif framework == "icd10":
                code = self.icd10_codes[idx] if self.icd10_codes else None
                title = self.icd10_titles[idx] if self.icd10_titles else None
                chapter = self.chapters[idx] if self.chapters else None
                uri = ""
            elif framework == "snomed":
                code = self.snomed_codes[idx] if self.snomed_codes else None
                title = self.snomed_titles[idx] if self.snomed_titles else None
                chapter = ""
                uri = ""

            results.append({
                "Input Disease": disease,
                "Framework": framework.upper(),
                "Code": code,
                "Title": title,
                "Chapter": chapter,
                "URI": f"https://icd.who.int/browse/2025-01/mms/en#{uri}" if uri else None,
                "Similarity": score,
            })

        return results

    def disease_coder(self, disease, Z_BOOST=0.06):
        """Code a single disease."""
        return self.disease_coder_batch([disease], Z_BOOST)[0]

    def _process_batch(self, examples):
        """Process batch with NER and disease coding."""
        from tqdm.contrib.logging import logging_redirect_tqdm
        
        # Lowercase if your NER model requires it
        texts = [t.lower() for t in examples[self.text_column]]
        
        # Run NER
        ner_results = self.model(texts)
        
        all_diseases = []
        disease_spans = []
        batch_symptoms, batch_pathogens = [], []

        # Extract entities
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

        # Code diseases if framework is specified
        if self.framework is None:
            batch_diseases = [
                [{"Entity": d} for d in all_diseases[s:e]] 
                for s, e in disease_spans
            ]
        else:
            if all_diseases:
                coded = self.disease_coder_batch(all_diseases, framework=self.framework)
                batch_diseases = [coded[s:e] for s, e in disease_spans]
            else:
                batch_diseases = [[] for _ in range(len(texts))]

        return {
            "disease_extraction": batch_diseases,
            "pathogen_extraction": batch_pathogens,
            "symptom_extraction": batch_symptoms,
        }

    def predict(self, dataset):
        """Predict on entire dataset with progress bar."""
        from tqdm.contrib.logging import logging_redirect_tqdm
        
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with logging_redirect_tqdm():
            processed_dataset = dataset.map(
                self._process_batch,
                batched=True,
                batch_size=self.batch_size,
                desc=f"[{date_time} |   INFO  | PetCoder]",
                load_from_cache_file=False,
            )
        return processed_dataset

    def single_predict(self, dataset):
        """Predict on single text example."""
        dataset = dataset.select([0])
        processed = self._process_batch(examples=dataset)
        
        return Dataset.from_dict({
            self.text_column: dataset[self.text_column],
            "Code": processed["disease_extraction"],
            "pathogen_extraction": processed["pathogen_extraction"],
            "symptom_extraction": processed["symptom_extraction"],
        })
