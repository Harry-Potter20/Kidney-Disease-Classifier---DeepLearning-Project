import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.augmentation import mixup_tf, cutmix_tf


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

    def load_dataset_from_directory(self):
        """Loads and prepares training and validation datasets from directories."""
        print("📂 Loading datasets...")

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

        # Normalize
        def normalize(x, y):
            return tf.cast(x, tf.float32) / 255.0, y

        train_dataset = train_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        valid_dataset = valid_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle
        train_dataset = train_dataset.shuffle(buffer_size=self.config.shuffle_buffer_size)

        # 🧪 Apply MixUp or CutMix
        if self.config.use_mixup:
            print("🧪 Applying MixUp augmentation...")
            train_dataset = train_dataset.map(
                lambda x, y: mixup_tf(x, y, alpha=self.config.mixup_alpha),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        elif self.config.use_cutmix:
            print("🧪 Applying CutMix augmentation...")
            train_dataset = train_dataset.map(
                lambda x, y: cutmix_tf(x, y, alpha=self.config.cutmix_alpha),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Cache datasets (if enabled)
        if self.config.cache_dataset:
            print("📦 Caching datasets...")
            train_dataset = train_dataset.cache()
            valid_dataset = valid_dataset.cache()

        # Prefetch for performance
        buffer_size = (
            tf.data.AUTOTUNE if self.config.prefetch_buffer_size == -1
            else self.config.prefetch_buffer_size
        )
        train_dataset = train_dataset.prefetch(buffer_size=buffer_size)
        valid_dataset = valid_dataset.prefetch(buffer_size=buffer_size)

        # Store dataset size for later use
        self.train_data_size = self._count_dataset(train_dataset)
        print(f"✅ Training dataset loaded. Size: {self.train_data_size} batches")

        return train_dataset, valid_dataset

    @staticmethod
    def _count_dataset(dataset):
        """Counts the number of batches in a tf.data.Dataset."""
        return sum(1 for _ in dataset)

    def calculate_class_weights(self, dataset):
        """Automatically calculates class weights based on the dataset labels."""
        # Extract the labels from the dataset
        labels = []
        for _, label in dataset:
            labels.extend(np.argmax(label.numpy(), axis=-1))  # Convert one-hot to class index
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

        # Return a dictionary with class indices as keys and class weights as values
        return {i: class_weights[i] for i in range(len(class_weights))}

    def train(self):
        """Runs the training pipeline: loads data, trains the model, saves it."""
        # Load data
        train_dataset, valid_dataset = self.load_dataset_from_directory()

        # Load base model
        self.get_base_model()

        # Handle class weights based on the boolean flag
        class_weight = None
        if self.config.use_class_weights:  # Check if we need to calculate class weights
            print("⚖️ Calculating class weights...")
            class_weight = self.calculate_class_weights(train_dataset)

        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=self.config.params_epochs,
            callbacks=self.config.callbacks_list,
            class_weight=class_weight
        )

        # Save the trained model
        self.model.save(self.config.trained_model_path)
        print(f"✅ Trained model saved to: {self.config.trained_model_path}")

        return history
