import os
import zipfile
import gdown
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size, save_json
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        try:
            dataset_url = self.config.source_URL
            zip_download_path = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_path), exist_ok=True)

            logger.info(f"📥 Downloading data from {dataset_url} into {zip_download_path}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?export=download&id='
            gdown.download(prefix + file_id, zip_download_path, quiet=False)

            logger.info(f"✅ Download complete: {zip_download_path} ({get_size(Path(zip_download_path))})")

            return zip_download_path

        except Exception as e:
            logger.error(f"❌ Failed to download file from {dataset_url}: {e}")
            raise e

    def extract_zip_file(self) -> None:
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"📦 Extracting zip file: {self.config.local_data_file} to {unzip_path}")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info("✅ Extraction complete.")

            self._save_class_names()

        except Exception as e:
            logger.error(f"❌ Failed to extract zip file: {e}")
            raise e

    def _save_class_names(self):
        """
        Save class names inferred from folder structure after unzip.
        """
        try:
            extracted_root = Path(self.config.unzip_dir)

            # Try to go one level deeper if a single folder exists
            subdirs = [d for d in extracted_root.iterdir() if d.is_dir() and not d.name.startswith("__")]

            if len(subdirs) == 1:
                class_dir = subdirs[0]
            else:
                class_dir = extracted_root

            class_names = sorted([
                folder.name for folder in class_dir.iterdir()
                if folder.is_dir() and not folder.name.startswith("__")
            ])

            output_path = Path(self.config.root_dir) / "class_names.json"
            save_json(output_path, {"class_names": class_names})

            logger.info(f"✅ Saved class names to {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save class names: {e}")
            raise e
