import os
import sys

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utils.main_utils.utils import (
    evaluate_models,
    load_numpy_array_data,
    load_object,
    save_object,
)
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.utils.ml_utils.model.estimator import NetworkModel


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)  # type: ignore

    def track_mlflow(self, best_model, classificationmetric, X_data):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            predictions = best_model.predict(X_data)

            signature = infer_signature(X_data, predictions)

            mlflow.sklearn.log_model(  # type: ignore
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_data[:5],
            )

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_features": ["sqrt", "log2"],
            },
            "Random Forest": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_features": ["sqrt", "log2", None],
                "n_estimators": [8, 16, 32, 128, 256],
            },
            "Gradient Boosting": {
                "loss": ["log_loss", "exponential"],
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "criterion": ["squared_error", "friedman_mse"],
                "max_features": ["sqrt", "log2"],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }
        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(
            y_true=y_train, y_pred=y_train_pred
        )

        # Track training metrics with MLflow
        self.track_mlflow(best_model, classification_train_metric, X_train)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(
            y_true=y_test, y_pred=y_test_pred
        )

        # Track test metrics with MLflow
        self.track_mlflow(best_model, classification_test_metric, x_test)

        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(
            self.model_trainer_config.trained_model_file_path, obj=Network_Model
        )
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)  # type: ignore
