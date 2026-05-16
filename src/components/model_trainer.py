import os
import sys

from dataclasses import dataclass

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# Optional advanced regressors
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # =========================
    # Convert Numeric to Grade
    # =========================
    def convert_numeric_to_grade(self, score):

        if score >= 4.75:
            return "AA"

        elif score >= 4.25:
            return "AB"

        elif score >= 3.75:
            return "BB"

        elif score >= 3.25:
            return "BC"

        elif score >= 2.75:
            return "CC"

        elif score >= 2.25:
            return "CD"

        elif score >= 1.75:
            return "DD"

        elif score >= 1.25:
            return "DF"

        else:
            return "FF"

    # =========================
    # Evaluate Models
    # =========================
    def evaluate_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        models,
    ):

        model_report = {}

        for model_name, model in models.items():

            logging.info(
                f"Training model: {model_name}"
            )

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(
                y_test,
                y_test_pred
            )

            mse = mean_squared_error(
                y_test,
                y_test_pred
            )

            rmse = mse ** 0.5

            r2 = r2_score(
                y_test,
                y_test_pred
            )

            train_r2 = r2_score(
                y_train,
                y_train_pred
            )

            # Cross Validation
            cv_mae_scores = cross_val_score(
                clone(model),
                X_train,
                y_train,
                cv=5,
                scoring="neg_mean_absolute_error",
            )

            cv_rmse_scores = cross_val_score(
                clone(model),
                X_train,
                y_train,
                cv=5,
                scoring="neg_root_mean_squared_error",
            )

            cv_r2_scores = cross_val_score(
                clone(model),
                X_train,
                y_train,
                cv=5,
                scoring="r2",
            )

            cv_mae = -cv_mae_scores.mean()

            cv_rmse = -cv_rmse_scores.mean()

            cv_r2 = cv_r2_scores.mean()

            model_report[model_name] = {

                "MAE": mae,

                "MSE": mse,

                "RMSE": rmse,

                "R2": r2,

                "Train_R2": train_r2,

                "CV_MAE": cv_mae,

                "CV_RMSE": cv_rmse,

                "CV_R2": cv_r2,
            }

        return model_report

    # =========================
    # Main Training Function
    # =========================
    def initiate_model_trainer(
        self,
        train_array,
        test_array,
    ):

        try:

            logging.info(
                "Splitting training and testing arrays"
            )

            # Split arrays
            X_train = train_array[:, :-1]

            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]

            y_test = test_array[:, -1]

            logging.info(
                f"X_train shape: {X_train.shape}"
            )

            logging.info(
                f"X_test shape: {X_test.shape}"
            )

            # =========================
            # Models
            # =========================
            models = {

                "Ridge Regression": Ridge(
                    alpha=1.0
                ),

                "Decision Tree Regressor":
                DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                ),

                "Random Forest Regressor":
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                ),

                "Gradient Boosting Regressor":
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                ),
            }

            # XGBoost
            if XGBRegressor is not None:

                models["XGBRegressor"] = XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                )

            # CatBoost
            if CatBoostRegressor is not None:

                models["CatBoostRegressor"] = CatBoostRegressor(
                    iterations=800,
                    learning_rate=0.05,
                    depth=8,
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                )

            # =========================
            # Evaluate Models
            # =========================
            model_report = self.evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            logging.info(
                "========== MODEL REPORT =========="
            )

            for model_name, metrics in model_report.items():

                logging.info(
                    f"----- {model_name} -----"
                )

                logging.info(
                    f"Train R2: "
                    f"{metrics['Train_R2']:.4f}"
                )

                logging.info(
                    f"Test R2: "
                    f"{metrics['R2']:.4f}"
                )

                logging.info(
                    f"MAE: "
                    f"{metrics['MAE']:.4f}"
                )

                logging.info(
                    f"RMSE: "
                    f"{metrics['RMSE']:.4f}"
                )

                logging.info(
                    f"CV MAE: "
                    f"{metrics['CV_MAE']:.4f}"
                )

                logging.info(
                    f"CV RMSE: "
                    f"{metrics['CV_RMSE']:.4f}"
                )

                logging.info(
                    f"CV R2: "
                    f"{metrics['CV_R2']:.4f}"
                )

            # =========================
            # Best Model Selection
            # =========================
            best_model_name = min(
                model_report,
                key=lambda x:
                model_report[x]["CV_RMSE"]
            )

            best_model = models[
                best_model_name
            ]

            logging.info(
                f"Best Model Selected: "
                f"{best_model_name}"
            )

            logging.info(
                f"Best Model CV_RMSE: "
                f"{model_report[best_model_name]['CV_RMSE']:.4f}"
            )

            # =========================
            # Retrain Best Model
            # =========================
            best_model.fit(
                X_train,
                y_train
            )

            # =========================
            # Sample Prediction
            # =========================
            final_predictions = best_model.predict(
                X_test
            )

            sample_prediction = final_predictions[0]

            predicted_grade = self.convert_numeric_to_grade(
                sample_prediction
            )

            logging.info(
                f"Sample Predicted Numeric Score: "
                f"{sample_prediction:.3f}"
            )

            logging.info(
                f"Sample Predicted Grade: "
                f"{predicted_grade}"
            )

            # =========================
            # Save Model
            # =========================
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info(
                "Best model saved successfully"
            )

            # =========================
            # Return Results
            # =========================
            return {

                "best_model_name":
                best_model_name,

                "best_model_metrics":
                model_report[
                    best_model_name
                ],

                "sample_prediction": {

                    "numeric_score":
                    round(
                        float(sample_prediction),
                        3
                    ),

                    "grade":
                    predicted_grade,
                }
            }

        except Exception as e:
            raise CustomException(e, sys)