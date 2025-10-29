import os
import logging
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from typing import Dict, Any, Optional
import os
import datetime


class DatasetProcessor:
    def __init__(
        self,
    ):
        self.logger = logging.getLogger(__name__)
        self._last_input_format = None  # Initialize

    def validate_dataset(self, dataset, text_column) -> None:
        if text_column not in dataset.column_names:
            error_message = f"Text column '{text_column}' not found in dataset. Please specifiy 'text_column' column to the class."
            self.logger.error(error_message)
            raise ValueError(error_message)
        # drop missing rows
        clean_dataset = dataset.filter(lambda example: example[text_column] is not None)
        self.logger.info(
            f"Dataset contains {len(dataset)} rows. After removing missing rows, {len(clean_dataset)} rows remain."
        )
        return clean_dataset

    def load_dataset_file(self, file_path: str, split: str = "train") -> Dataset:
        """
        Loads a dataset from a file or directory, attempting to use HuggingFace's
        `load_dataset` for known file types and `load_from_disk` for directories
        or other formats.

        Args:
            file_path (str): Path to the dataset file or directory.
            split (str): The split of the HuggingFace dataset to load (e.g., 'train', 'test', 'eval').
                        Defaults to 'train'.

        Returns:
            A HuggingFace Dataset object for the specified split.

        Raises:
            ValueError: If the file format is unsupported or the dataset/split cannot be loaded.
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        self._last_input_format = file_extension  # Store the input format
        _LOAD_MAPPING = {
            ".csv": "csv",
            ".arrow": "arrow",
            ".json": "json",
            ".parquet": "parquet",
        }
        if file_extension in _LOAD_MAPPING:
            dataset_format = _LOAD_MAPPING[file_extension]
            try:
                loaded_dataset = load_dataset(dataset_format, data_files=file_path)
                self.logger.info(f"Loaded '{dataset_format}' dataset from {file_path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to load '{dataset_format}' dataset from {file_path}: {e}"
                )
                raise
        elif file_extension == ".pkl":
            try:
                loaded_dataset = pd.read_pickle(file_path)
                loaded_dataset = Dataset.from_pandas(loaded_dataset)
                loaded_dataset = DatasetDict({"train": loaded_dataset})
                self.logger.info(f"Loaded dataset from pickle file at {file_path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to load dataset from pickle file at {file_path}: {e}"
                )
                raise
        else:
            try:
                loaded_dataset = load_from_disk(file_path)
                self.logger.info(f"Loaded dataset from disk at {file_path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to load dataset from disk at {file_path}: {e}"
                )
                raise
        if not loaded_dataset:
            error_message = f"Dataset not found or could not be loaded from {file_path}"
            self.logger.error(error_message)
            raise ValueError(error_message)
        if split in loaded_dataset:
            self.logger.info(f"Returning '{split}' split from the loaded dataset.")
            return loaded_dataset[split]
        elif "train" in loaded_dataset and split != "train":
            self.logger.warning(
                f"Split '{split}' not found. Returning the default 'train' split. "
                f"Available splits are: {list(loaded_dataset.keys())}"
            )
            return loaded_dataset["train"]
        elif len(loaded_dataset.keys()) == 1:
            default_split = list(loaded_dataset.keys())[0]
            if split != default_split:
                self.logger.info(
                    f"Returning the only available split: '{default_split}'."
                )
            return loaded_dataset[default_split]
        else:
            error_message = (
                f"Split '{split}' not found in the loaded dataset. "
                f"Available splits are: {list(loaded_dataset.keys())}"
            )
            self.logger.error(error_message)
            raise ValueError(error_message)

    def load_cache(self, dataset, cache_column='label') -> tuple:
        """
        Filter out anonymised data from the dataset using a cache.

        Args:
            dataset: The dataset to filter.
            cache (bool | str): If True, removes examples where "annonymised" == 1.
                                If str, treats it as a column name and filters out rows
                                based on cached record IDs from a text file.
            cache_path (str): Path to cache directory (only used if `cache` is a str).

        Returns:
            tuple: (filtered_dataset, original_dataset)
        """
        target_dataset = dataset
        completed_dataset = None
        if cache:
            try:
                # If the goal is: Non-empty string means "completed"
                completed_dataset = dataset.filter(
                    lambda example: example.get(label_column, "") != "" # Use "" as a safe default
                )

                # Target: The column is an empty string (not completed)
                target_dataset = dataset.filter(
                    lambda example: example.get(label_column, "") == "" # Use "" as a safe default
                )
                else 
                    raise ValueError(
                        "`cache` must be either a boolean."
                    )

                self.logger.info(
                    f"Cache enabled | Skipping {len(completed_dataset)} Coded rows | Processing {len(target_dataset)} rows"
                )

            except Exception as e:
                self.logger.error(f"Failed to apply cache filtering: {e}")

            if not target_dataset:
                self.logger.info("All data appears to have been Coded. Exiting...")
                self.logger.warning(
                    "If this was unexpected, please check your cache file or delete a column called 'annonymised' in your dataset."
                )
                import sys

                sys.exit(0)
            else:
                self.logger.info(f"Processing {len(target_dataset)} non-Coded rows")
                return (
                    target_dataset,
                    completed_dataset,
                )
        else:
            self.logger.info("Cache disabled | Processing all data")
            return (
                target_dataset,
                completed_dataset,
            )

    def save_dataset_file(
        self,
        target_dataset: Dataset,
        completed_dataset: Dataset,
        output_dir: Optional[str] = None,
        cache: bool = False,
    ):
        """
        Save the target dataset to a file, attempting to use the same format
        as the original input.

        Args:
            original_data (Dataset): The original HuggingFace Dataset (used to infer format).
            target_dataset (Dataset): The HuggingFace Dataset to save.
            output_dir (str, optional): The directory or full file path for saving.
                                        If None, defaults to a CSV in the current directory.
            cache (bool or str, optional): If True, adds an 'annonymised' column and logs.
                                          If a string, appends the values from that column
                                          to the cache file. Defaults to False.
        """
        target_df = target_dataset.to_pandas()

        date = datetime.datetime.now().strftime("%Y_%m_%d")
        save_path = output_dir
        if cache:
            if isinstance(cache, bool):
                target_df["annonymised"] = 1
                self.logger.info(
                    "Cache enabled || 'annoymised' column added to dataset. Note: Review our documentation for more details."
                )
            elif isinstance(cache, str):
                cache_ids = target_df[cache].tolist()
                # Read in the cache file and append the new ids to the bottom
                with open(self.full_cache_path, "a") as f:
                    for id in cache_ids:
                        f.write(f"{id}\n")
            if completed_dataset is not None:
                original_df = completed_dataset.to_pandas()
                target_df = pd.concat([original_df, target_df], ignore_index=True)

            save_extensions = [".csv", ".json", ".parquet", ".arrow", ".pkl"]

            # Determine save path and format
            if output_dir and output_dir.endswith(tuple(save_extensions)):
                save_format = os.path.splitext(output_dir)[1]
                save_path = output_dir
                logging.info(f"Output format set to: {save_format}")
            elif output_dir:
                # If output_dir has no extension, assume it's a directory
                if hasattr(self, "_last_input_format") and self._last_input_format:
                    save_format = self._last_input_format
                    logging.info(
                        f"Using original input format for saving: {save_format}"
                    )
                else:
                    save_format = ".csv"
                    logging.info(
                        "No original input format found, defaulting to CSV for saving."
                    )
                base_name = f"petharbor_Coded_{date}{save_format}"
                save_path = os.path.join(output_dir, base_name)
            else:
                # Default if no output_dir provided
                save_format = ".csv"
                save_path = f"petharbor_Coded_{date}.csv"

            logging.info(f"Saving dataset using format: {save_format}")

            # Ensure output directory exists
            output_dirname = os.path.dirname(save_path)
            if output_dirname and not os.path.exists(output_dirname):
                os.makedirs(output_dirname, exist_ok=True)

            # Save based on format
            try:
                if save_format == ".csv":
                    target_df.to_csv(save_path, index=False)
                elif save_format == ".json":
                    target_df.to_json(save_path, orient="records", lines=True)
                elif save_format == ".parquet":
                    target_df.to_parquet(save_path, index=False)
                elif save_format == ".arrow":
                    target_dataset.save_to_disk(save_path)
                elif save_format == ".pkl":
                    target_df.to_pickle(save_path)
                else:
                    target_df.to_csv(save_path, index=False)
                    self.logger.warning(
                        f"Unsupported format '{save_format}', saved as CSV."
                    )

                self.logger.info(f"Saved dataset to {save_path} ({save_format}).")

            except Exception as e:
                self.logger.error(
                    f"Failed to save dataset to {save_path}: {e}. Aborting, saving output as petharbor_anonymised.csv to current directory"
                )
                target_df.to_csv("petharbor_anonymised.csv", index=False)
                raise
