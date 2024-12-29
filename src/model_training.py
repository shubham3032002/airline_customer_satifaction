import os
import sys
import json
import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import lightgbm as lgb
from src.logger import logger
from src.exception import MyException
from config.path_config import ENGINEERED_DATA_PATH, PARAMS_PATH, MODEL_SAVE_PATH


class ModelTraining:
    def __init__(self, data_path, params_path, model_save_path, experiment_name="model_training_experiment"):
        self.data_path = data_path
        self.params_path = params_path
        self.model_save_path = model_save_path
        self.exp_name = experiment_name
        self.best_model = None
        self.metrics = None

    def load_data(self):
        try:
            logger.info("Loading data for model training.")
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data path not found: {self.data_path}")

            data = pd.read_csv(self.data_path)
            if 'satisfaction' not in data.columns:
                raise ValueError("Target column 'satisfaction' is missing in the dataset.")
            
            logger.info("Data loaded successfully.")
            return data

        except Exception as e:
            raise MyException("Error while loading data", sys)

    def split_data(self, data):
        try:
            logger.info("Splitting data into training and testing sets.")
            X = data.drop(columns='satisfaction')
            y = data['satisfaction']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info("Data splitting completed.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise MyException("Error while splitting data", sys)

    def train_model(self, X_train, y_train, params):
        try:
            logger.info("Starting model training.")
            lgbm = lgb.LGBMClassifier()

            grid_search = GridSearchCV(lgbm, param_grid=params, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            logger.info("Model training completed.")
            self.best_model = grid_search.best_estimator_
            return grid_search.best_params_

        except Exception as e:
            raise MyException("Error while training model", sys)

    def evaluate_model(self, X_test, y_test):
        try:
            logger.info("Evaluating the model.")
            y_pred = self.best_model.predict(X_test)

            self.metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }
            
            cm = confusion_matrix(y_test, y_pred)
            self.metrics["confusion_matrix"] = cm.tolist()

            logger.info(f"Evaluation metrics: {self.metrics}")
            return self.metrics

        except Exception as e:
            raise MyException("Error while evaluating model", sys)

    def save_model(self):
        try:
            logger.info("Saving the trained model.")
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            joblib.dump(self.best_model, self.model_save_path)
            logger.info(f"Model saved successfully at {self.model_save_path}.")

        except Exception as e:
            raise MyException("Error while saving the model", sys)

    def run(self):
        try:
            logger.info("Starting the model training pipeline.")
            mlflow.set_experiment(self.exp_name)

            with mlflow.start_run():
                # Load data
                data = self.load_data()
                X_train, X_test, y_train, y_test = self.split_data(data)

                # Load hyperparameters
                with open(self.params_path, "r") as f:
                    params = json.load(f)
                logger.info(f"Loaded hyperparameters: {params}")
                mlflow.log_params({f"grid_{key}": value for key, value in params.items()})

                # Train model
                best_params = self.train_model(X_train, y_train, params)
                logger.info(f"Best hyperparameters: {best_params}")
                mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})

                # Evaluate model
                metrics = self.evaluate_model(X_test, y_test)
                for metric, value in metrics.items():
                    if metric != "confusion_matrix":
                        mlflow.log_metric(metric, value)
                    else:
                        mlflow.log_dict({"confusion_matrix": value}, "confusion_matrix.json")

                # Save model
                self.save_model()
                mlflow.sklearn.log_model(self.best_model, "model")
                mlflow.end_run(status="FINISHED")

        except Exception as e:
            logger.error("Error while running the model training pipeline.")
            mlflow.end_run(status="FAILED")
            raise MyException(e, sys)


if __name__ == "__main__":
    model_training = ModelTraining(
        data_path=ENGINEERED_DATA_PATH,
        params_path=PARAMS_PATH,
        model_save_path=MODEL_SAVE_PATH
    )
    model_training.run()
