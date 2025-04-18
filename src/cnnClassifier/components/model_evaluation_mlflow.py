import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json

class Evaluation:
    """
    A class responsible for evaluating a trained TensorFlow Keras model
    using tf.data.Dataset for data loading and logging results to MLflow.
    """
    def __init__(self, config: EvaluationConfig):
        """
        Initializes the Evaluation class with an evaluation configuration.

        Args:
            config (EvaluationConfig): A configuration object containing
                                       evaluation parameters and paths.
        """
        self.config = config
        self.model = None
        self.valid_dataset = None  # Changed from valid_generator to hold tf.data.Dataset
        self.score = None

    def create_validation_dataset(self):
        """
        Loads the validation dataset directly from image files in a directory
        structure using tf.keras.utils.image_dataset_from_directory.
        Applies necessary preprocessing (rescaling) and performance optimizations.
        """
        print(f"Loading validation data from directory: {self.config.training_data}")

        # Use tf.keras.utils.image_dataset_from_directory which returns a tf.data.Dataset
        valid_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data, # Path to the root directory
            labels='inferred',               # Labels are inferred from subdirectory names
            label_mode='categorical',        # Labels are one-hot encoded vectors
            image_size=self.config.params_image_size[:-1], # Target image size (excluding channel)
            interpolation="bicubic",         # Resizing method
            batch_size=self.config.params_batch_size,   # Number of samples per batch
            shuffle=False,                   # Do not shuffle validation data
            seed=42,                         # Seed for reproducibility (if validation_split is used)
            validation_split=0.30,           # Same split as used in training
            subset="validation",             # Specify that this is the validation subset
        )

        # Apply rescaling (0-255 to 0.0-1.0) using a map function on the dataset
        # This replaces the rescale=1./255 from ImageDataGenerator
        valid_dataset = valid_dataset.map(lambda x, y: (x / 255.0, y),
                                          num_parallel_calls=tf.data.AUTOTUNE) # Process mapping in parallel

        # Apply performance optimizations
        valid_dataset = valid_dataset.cache() # Cache dataset elements in memory after first pass
        valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE) # Overlap data preprocessing and evaluation

        self.valid_dataset = valid_dataset # Store the created tf.data.Dataset object

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Loads a trained TensorFlow Keras model from the specified path."""
        print(f"Loading model from {path}")
        try:
            model = tf.keras.models.load_model(path)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e # Re-raise the exception

    def evaluation(self):
        """
        Loads the model and validation dataset, evaluates the model,
        and saves the evaluation score to a JSON file.
        """
        self.model = self.load_model(self.config.path_of_model)
        self.create_validation_dataset() # Use the new method to create tf.data.Dataset

        print("🚀 Starting model evaluation...")
        # Evaluate the model directly using the tf.data.Dataset
        # tf.keras.Model.evaluate accepts a tf.data.Dataset
        self.score = self.model.evaluate(self.valid_dataset, verbose=1)
        print(f"✅ Evaluation complete. Loss: {self.score[0]:.4f}, Accuracy: {self.score[1]:.4f}")

        self.save_score()

    def save_score(self):
        """Saves the evaluation scores (loss and accuracy) to a JSON file."""
        if self.score is not None:
            scores = {
                "loss": float(self.score[0]),
                "accuracy": float(self.score[1])
            }
            save_json(path=Path("scores.json"), data=scores)
            print(f"💾 Evaluation scores saved to scores.json")
        else:
            print("❌ No evaluation score available to save.")


    def log_into_mlflow(self):
        """
        Logs parameters, evaluation metrics, and the trained model to MLflow.
        """
        print("Logging results to MLflow...")
        try:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # Log hyperparameters if available in config
                if hasattr(self.config, 'all_params'):
                     mlflow.log_params(self.config.all_params)
                     print("📝 Logged parameters to MLflow.")
                else:
                     print("⚠️ 'all_params' not found in config, skipping parameter logging.")

                # Log evaluation metrics if available
                if self.score is not None:
                     mlflow.log_metrics({
                         "val_loss": float(self.score[0]),
                         "val_accuracy": float(self.score[1])
                     })
                     print("📈 Logged metrics (val_loss, val_accuracy) to MLflow.")
                else:
                     print("❌ No score found to log metrics.")

                # Log the model
                if self.model is not None:
                     print("📦 Logging model to MLflow...")
                     model_name = self.config.all_params.get("REGISTERED_MODEL_NAME", "ResNet101V2") if hasattr(self.config, 'all_params') else "ResNet101V2"
                     if tracking_url_type_store != "file":
                         # Log with registration name if not a file-based store (e.g., http, postgres)
                         mlflow.keras.log_model(
                             self.model,
                             "model",
                             registered_model_name=model_name
                         )
                         print(f"✅ Model logged and registered as '{model_name}' in MLflow.")
                     else:
                         # Log without registration name for file-based store
                         mlflow.keras.log_model(self.model, "model")
                         print("✅ Model logged to MLflow (file store).")
                else:
                     print("❌ No model found to log.")

            print("✅ MLflow logging complete.")

        except Exception as e:
            print(f"❌ Error during MLflow logging: {e}")
            # Optionally re-raise or handle the logging error