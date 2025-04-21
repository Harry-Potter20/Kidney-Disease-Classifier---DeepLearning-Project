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
        """Loads and saves the base model without custom top layers."""
        self.model = tf.keras.applications.ResNet101V2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(self.config.base_model_path, self.model)

    @staticmethod
    def prepare_full_model(model, config: BaseModelConfig, learning_rate_schedule=None):
        """Adds custom layers including BatchNorm and compiles the full model."""
        # Freeze layers if needed
        if config.params_freeze_till is not None and config.params_freeze_till > 0:
            for layer in model.layers[:-config.params_freeze_till]:
                layer.trainable = False
        else:
            for layer in model.layers:
                layer.trainable = False

        # Top layers
        x = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dropout(config.params_dropout_rate)(x)
        x = tf.keras.layers.Dense(
            config.params_dense_units,
            activation='relu',
            kernel_regularizer=l2(config.params_l2_regularization)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)  # ✅ Added BatchNormalization
        x = tf.keras.layers.Dropout(config.params_dropout_rate)(x)
        prediction = tf.keras.layers.Dense(
            units=config.params_classes,
            activation="softmax",
            kernel_regularizer=l2(config.params_l2_regularization)
        )(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        # Use cosine decay schedule if available
        learning_rate = learning_rate_schedule if learning_rate_schedule else config.params_learning_rate

        full_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """Applies custom layers and compiles model with or without learning rate schedule."""
        learning_rate_schedule = None

        if self.config.use_cosine_decay:
            learning_rate_schedule = CosineDecayRestarts(
                initial_learning_rate=self.config.params_learning_rate,
                first_decay_steps=self.config.first_decay_steps,
                t_mul=self.config.t_mul,
                m_mul=self.config.m_mul,
                alpha=self.config.cosine_decay_alpha,
            )

        self.full_model = self.prepare_full_model(
            model=self.model,
            config=self.config,
            learning_rate_schedule=learning_rate_schedule
        )

        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the model to the specified path."""
        model.save(path)