from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import tensorflow as tf


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_dense_units: int
    params_dropout_rate: float
    params_l2_regularization: float
    params_freeze_till: int
    use_cosine_decay: bool
    cosine_decay_type: str                  # "restart" or "normal"
    first_decay_steps: int                  # in epochs
    t_mul: float
    m_mul: float
    cosine_decay_alpha: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    label_smoothing: float  # NEW
    use_class_weights: bool  # NEW
    use_mixup: bool          # NEW
    use_cutmix: bool
    mixup_alpha: float
    cutmix_alpha: float
    label_smoothing: float
    class_weights: bool         # NEW
    callbacks_list: Optional[List[tf.keras.callbacks.Callback]] = None 
    shuffle_buffer_size: int
    prefetch_buffer_size: int       # Use -1 for tf.data.AUTOTUNE
    cache_dataset: bool

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    prefetch_buffer_size: int       # Use -1 for tf.data.AUTOTUNE
