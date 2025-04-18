from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig, 
    BaseModelConfig, 
    TrainingConfig, 
    EvaluationConfig
)
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir, 
            source_URL=config.source_URL,
            local_data_file=config.local_data_file, 
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir), 
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path), 
            params_image_size=self.params.IMAGE_SIZE, 
            params_learning_rate=self.params.LEARNING_RATE, 
            params_include_top=self.params.INCLUDE_TOP, 
            params_weights=self.params.WEIGHTS, 
            params_classes=self.params.CLASSES,
            params_dense_units=self.params.DENSE_UNITS,
            params_dropout_rate=self.params.DROPOUT_RATE,
            params_l2_regularization=self.params.L2_REGULARIZATION,
            params_freeze_till=self.params.FREEZE_TILL, 
            use_cosine_decay=self.params.USE_COSINE_DECAY,
            cosine_decay_type=self.params.COSINE_DECAY_TYPE,
            first_decay_steps=self.params.FIRST_DECAY_STEPS,
            t_mul=self.params.T_MULTIPLIER,
            m_mul=self.params.M_MULTIPLIER,
            cosine_decay_alpha=self.params.COSINE_DECAY_ALPHA
        )
        return base_model_config

    def get_callbacks(self):
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=self.params.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
        ]
    
        if not self.params.USE_COSINE_DECAY:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_accuracy",
                    patience=self.params.REDUCE_LR_PATIENCE,
                    factor=self.params.REDUCE_LR_FACTOR,
                    verbose=1
                )
            )
        return callbacks

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        base_model = self.config.base_model
        params = self.params

        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )

        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir), 
            trained_model_path=Path(training.trained_model_path), 
            updated_base_model_path=Path(base_model.updated_base_model_path), 
            training_data=Path(training_data), 
            params_epochs=params.EPOCHS, 
            params_batch_size=params.BATCH_SIZE, 
            params_is_augmentation=params.AUGMENTATION, 
            params_image_size=params.IMAGE_SIZE, 
            early_stopping_patience=params.EARLY_STOPPING_PATIENCE,
            reduce_lr_patience=params.REDUCE_LR_PATIENCE,
            reduce_lr_factor=params.REDUCE_LR_FACTOR,

            # ✅ New strategies
            use_mixup=params.USE_MIXUP,
            mixup_alpha=params.MIXUP_ALPHA,
            use_cutmix=params.USE_CUTMIX,
            cutmix_alpha=params.CUTMIX_ALPHA,
            label_smoothing=params.LABEL_SMOOTHING,
            class_weights=params.CLASS_WEIGHTS, 
            use_class_weights=params.USE_CLASS_WEIGHTS,

            shuffle_buffer_size=params.SHUFFLE_BUFFER_SIZE,
            prefetch_buffer_size=params.PREFETCH_BUFFER_SIZE, # -1 for AUTOTUNE
            cache_dataset=params.get('CACHE_DATASET', True), # Default to False if missing


            # ✅ Callbacks list
            callbacks_list=self.get_callbacks()  # Add the callbacks here
        )
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5", 
            training_data="artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone", 
            mlflow_uri="https://dagshub.com/Harry-Potter20/Kidney-Disease-Classifier---DeepLearning-Project.mlflow",  
            all_params=self.params, 
            params_image_size=self.params.IMAGE_SIZE, 
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config

