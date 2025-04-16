import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from pathlib import Path 
from cnnClassifier.entity.config_entity import BaseModelConfig
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts  # Import CosineDecayRestarts

class BaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Loads the base model (ResNet101V2) and saves it."""
        self.model = tf.keras.applications.ResNet101V2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def prepare_full_model(model, classes, freeze_all=True, freeze_till=None, learning_rate_schedule=None):
        """Prepares the model by adding custom layers and applying learning rate scheduling."""
        # Freeze layers if needed
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add custom top layers
        x = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dropout(0.5)(x)  # Dropout to reduce overfitting
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(0.001)
        )(x)
        x = tf.keras.layers.Dropout(0.5)(x)  # Another Dropout
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            kernel_regularizer=l2(0.001)
        )(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        # If a learning rate schedule is provided, use it; otherwise, use the default fixed learning rate
        full_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate_schedule),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """Update the base model to the full model with custom top layers and LR scheduler."""
        # Define CosineDecayRestarts learning rate scheduler
        learning_rate_schedule = CosineDecayRestarts(
            initial_learning_rate=self.config.params_learning_rate,
            first_decay_steps=self.config.first_decay_steps,
            t_mul=self.config.t_mul,
            m_mul=self.config.m_mul,
            alpha=self.config.cosine_decay_alpha,
        )


        # Prepare the full model with the learning rate schedule
        self.full_model = self.prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,  # You can change to False and set freeze_till if needed
            freeze_till=None,
            learning_rate_schedule=learning_rate_schedule
        )

        # Save the updated model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Helper function to save the model."""
        model.save(path)
