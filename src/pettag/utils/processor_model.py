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
        device="cuda:0" if torch.cuda.is_available() else "cpu",
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
            all_embeddings, dtype=torch.float16, device=device
        )
        # Normalize once at initialization
        self.lookup_embeddings = F.normalize(self.lookup_embeddings, dim=1)
        logger.info(f"Lookup embeddings loaded: {self.lookup_embeddings.shape}")
        
        # --- Precompute parent code mapping for faster subcode lookup ---
        self.codes = np.array(self.disease_code_lookup["Code"])
        self.parent_to_subcodes = {}
        for idx, code in enumerate(self.codes):
            parent = code.split(".")[0]
            if parent not in self.parent_to_subcodes:
                self.parent_to_subcodes[parent] = []
            if "." in code:  # It's a subcode
                self.parent_to_subcodes[parent].append(idx)
        
        # Precompute Z-code mask
        self.z_code_mask = np.array([str(c).endswith(".Z") for c in self.codes])

    # -----------------------------------------------------
    # Batch disease coding (MAJOR SPEEDUP)
    # -----------------------------------------------------
    def disease_coder_batch(self, diseases, Z_BOOST=0.06):
        """Process multiple diseases at once for massive speedup."""
        if not diseases:
            return []
        
        # --- Step 1: Encode all diseases at once ---
        encoded_diseases = self.embedding_model.encode(
            diseases, convert_to_numpy=False, batch_size=32, show_progress_bar=False
        )
                    
        # Convert to tensor and normalize
        if isinstance(encoded_diseases, list):
            if isinstance(encoded_diseases[0], torch.Tensor):
                encoded_diseases = torch.stack(encoded_diseases).to(self.device)
            else:
                encoded_diseases = torch.tensor(
                    np.vstack(encoded_diseases), dtype=torch.float16, device=self.device
                )
        elif isinstance(encoded_diseases, np.ndarray):
            encoded_diseases = torch.tensor(
                encoded_diseases, dtype=torch.float16, device=self.device
            )
        encoded_diseases = F.normalize(encoded_diseases, dim=1)
        
        # --- Step 2: Compute all similarities at once ---
        similarities = torch.matmul(
            encoded_diseases, self.lookup_embeddings.T
        )  # [N, num_codes]
        
        # --- Step 3: Process each disease ---
        results = []
        for i, disease in enumerate(diseases):
            sims = similarities[i]
            top_idx = int(torch.argmax(sims).item())
            top_score = float(sims[top_idx].item())
            
            top_entry = self.disease_code_lookup[top_idx]
            top_code = top_entry["Code"]
            parent_code = top_code.split(".")[0]
            
            final_entry, final_score = top_entry, top_score
            
            # --- Check subcodes if we matched a parent code ---
            if top_code == parent_code and parent_code in self.parent_to_subcodes:
                subcode_indices = self.parent_to_subcodes[parent_code]
                
                if subcode_indices:
                    # Convert to tensor for indexing
                    subcode_tensor = torch.tensor(subcode_indices, device=self.device)
                    
                    # Get similarities for subcodes
                    sub_sims = sims[subcode_tensor].clone()
                    
                    # Apply Z_BOOST efficiently
                    z_mask_sub = torch.tensor(
                        self.z_code_mask[subcode_indices], 
                        device=self.device
                    )
                    sub_sims[z_mask_sub] += Z_BOOST
                    
                    # Find best subcode
                    sub_idx = int(torch.argmax(sub_sims).item())
                    final_entry = self.disease_code_lookup[subcode_indices[sub_idx]]
                    final_score = float(sub_sims[sub_idx].item())
            
            results.append({
                "Title": final_entry["Title"],
                "Code": final_entry["Code"],
                "ChapterNo": final_entry["ChapterNo"],
                "Foundation URI": f'https://icd.who.int/browse/2025-01/mms/en#{final_entry["URI"]}',
                "Similarity": float(final_score),
                "Input Disease": disease,
            })
        
        return results

    # -----------------------------------------------------
    # Keep single disease coder for backward compatibility
    # -----------------------------------------------------
    def disease_coder(self, disease, Z_BOOST=0.06):
        """Single disease coding (wraps batch method)."""
        return self.disease_coder_batch([disease], Z_BOOST)[0]

    # -----------------------------------------------------
    # Optimized batch processing
    # -----------------------------------------------------
    def _process_batch(self, examples):
        texts = [str(t).lower() for t in examples[self.text_column]]
        ner_results = self.ner_pipeline(texts)

        # Collect all unique diseases across batch for one encoding call
        all_diseases = []
        disease_indices = []  # Track which diseases belong to which doc
        
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
            
            # Deduplicate within document
            diseases = list(set(diseases))
            
            # Track indices for this document
            start_idx = len(all_diseases)
            all_diseases.extend(diseases)
            disease_indices.append((start_idx, len(all_diseases)))
            
            batch_pathogens.append(list(set(pathogens)))
            batch_symptoms.append(list(set(symptoms)))
        
        # Code ALL diseases at once
        if all_diseases:
            all_coded = self.disease_coder_batch(all_diseases)
            
            # Distribute coded diseases back to documents
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
        """
        Added batch_size parameter to control processing batches.
        Larger batches = more GPU utilization but more memory.
        """
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with logging_redirect_tqdm():
            processed_dataset = dataset.map(
                self._process_batch,
                batched=True,
                batch_size=batch_size,  # Controllable batch size
                desc=f"[{date_time} | INFO | PetHarbor-Advance]",
            )
        logger.info("Predictions obtained and text coded successfully.")
        return processed_dataset