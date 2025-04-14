import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.augmentation import mixup_generator, cutmix_generator

class Training:
    def __init__(self, config: TrainingConfig):
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

    def train_valid_generator(self):
        """Sets up training and validation data generators, applies MixUp/CutMix if enabled."""
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bicubic"
        )

        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=30,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        base_train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # Compute train data size
        self.train_data_size = len(base_train_generator.filenames)

        # Apply MixUp or CutMix wrapper
        if self.config.use_mixup:
            print("🧪 Using MixUp augmentation...")
            self.train_generator = mixup_generator(base_train_generator, alpha=self.config.mixup_alpha)
        elif self.config.use_cutmix:
            print("🧪 Using CutMix augmentation...")
            self.train_generator = cutmix_generator(base_train_generator, alpha=self.config.cutmix_alpha)
        else:
            print("✅ Using standard data generator...")
            self.train_generator = base_train_generator

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

        if self.train_data_size is None:
            raise ValueError("⚠️ train_data_size not computed. Make sure `train_valid_generator()` is called before training.")

        self.steps_per_epoch = self.train_data_size // self.config.params_batch_size
        self.validation_steps = self.valid_generator.samples // self.config.params_batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=self.config.callbacks_list
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
