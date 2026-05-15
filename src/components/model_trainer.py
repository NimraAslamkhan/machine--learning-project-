import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

# Optional advanced regressors (if installed)
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        """Return per-model metrics for regression."""
        model_report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, y_test_pred)

            # Cross-validation (helps avoid 'same score' feeling)
            # Use negative scores because sklearn maximizes.
            cv_mae = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
            cv_rmse = (-cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")).mean()

            model_report[model_name] = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "CV_MAE(mean)": cv_mae,
                "CV_RMSE(mean)": cv_rmse,
            }

        return model_report


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                # Simple baseline regressor
                "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                # Better-performing default ensembles for tabular data
                "Random Forest Regressor": RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            }

            if XGBRegressor is not None:
                models["XGBRegressor"] = XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    objective="reg:squarederror",
                )

            if CatBoostRegressor is not None:
                models["CatBoostRegressor"] = CatBoostRegressor(
                    iterations=800,
                    learning_rate=0.05,
                    depth=8,
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                )



            model_report: dict = self.evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            logging.info("Model Report (Regression) - lower MAE/RMSE is better; higher R2 is better")
            for model_name, metrics in model_report.items():
                logging.info(f"--- {model_name} ---")
                logging.info(
                    f"MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}, "
                    f"CV_MAE={metrics['CV_MAE(mean)']:.4f}, CV_RMSE={metrics['CV_RMSE(mean)']:.4f}"
                )

            # Pick best model by lowest CV_RMSE (more robust than single split)
            best_model_name = min(
                model_report.keys(),
                key=lambda k: model_report[k]["CV_RMSE(mean)"],
            )
            best_model = models[best_model_name]

            logging.info(f"Best Model Found: {best_model_name}")
            logging.info(f"Best Model CV_RMSE(mean): {model_report[best_model_name]['CV_RMSE(mean)']:.4f}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return model_report[best_model_name]


        except Exception as e:
            raise CustomException(e, sys)

