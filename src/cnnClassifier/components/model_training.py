import tensorflow as tf
import numpy as np
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.augmentation import mixup_tf, cutmix_tf


class Training:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_data_size = None  # Will hold computed train data size

    def get_base_model(self):
        """Loads the base model from a given path."""
        try:
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
            print(f"✅ Model loaded from {self.config.updated_base_model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def load_dataset_from_directory(self):
        """Loads training and validation datasets using tf.data."""
        # Load training data
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.training_data,
            validation_split=0.30,
            subset="training",
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            label_mode='categorical',
            shuffle=True,
            seed=42
        )

        # Load validation data
        valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.training_data,
            validation_split=0.30,
            subset="validation",
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            label_mode='categorical',
            shuffle=False,
            seed=42
        )

        # Apply MixUp or CutMix
        if self.config.use_mixup:
            print("🧪 Using MixUp augmentation...")
            train_dataset = train_dataset.map(
                lambda x, y: mixup_tf(x, y, alpha=self.config.mixup_alpha), 
                num_parallel_calls=tf.data.AUTOTUNE
            )
        elif self.config.use_cutmix:
            print("🧪 Using CutMix augmentation...")
            train_dataset = train_dataset.map(
                lambda x, y: cutmix_tf(x, y, alpha=self.config.cutmix_alpha),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Prefetch and cache for better performance
        train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
        valid_dataset = valid_dataset.cache().prefetch(tf.data.AUTOTUNE)

        return train_dataset, valid_dataset

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the trained model to disk."""
        model.save(path)
        print(f"💾 Model saved at {path}")

    def train(self):
        """Executes the training process with callbacks, steps, and model saving."""
        if self.model is None:
            print("ℹ️ Loading base model before training...")
            self.get_base_model()

        # Load datasets
        train_dataset, valid_dataset = self.load_dataset_from_directory()

        self.model.fit(
            train_dataset,
            epochs=self.config.params_epochs,
            validation_data=valid_dataset,
            callbacks=self.config.callbacks_list
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
