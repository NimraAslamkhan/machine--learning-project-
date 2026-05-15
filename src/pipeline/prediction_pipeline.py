import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor file not found at: {preprocessor_path}"
                )

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Avoid hard failures when user inputs categories not seen during training.
            # OneHotEncoder was fitted inside the preprocessing pipeline.
            try:
                for _, transformer, _cols in getattr(preprocessor, 'transformers_', []):
                    if hasattr(transformer, 'named_steps'):
                        for _step_name, step_obj in transformer.named_steps.items():
                            if hasattr(step_obj, 'handle_unknown'):
                                step_obj.set_params(handle_unknown='ignore')
                    elif hasattr(transformer, 'handle_unknown'):
                        transformer.set_params(handle_unknown='ignore')
            except Exception:
                # If we can't patch it, we'll still attempt transform normally.
                pass


            # Must match columns used during training pipeline transformation.
            expected_num_cols = ["Weekly_Study_Hours"]
            expected_cat_cols = [
                "Student_Age",
                "Sex",
                "High_School_Type",
                "Scholarship",
                "Additional_Work",
                "Attendance",
                "Reading",
                "Notes",
                "Listening_in_Class",
                "Sports_activity",
                "Transportation",
                "Project_work",
            ]
            expected_cols = expected_num_cols + expected_cat_cols

            missing = [c for c in expected_cols if c not in features.columns]
            if missing:
                raise KeyError(
                    "Missing required feature columns for prediction: "
                    + ", ".join(missing)
                )

            # Ensure correct column order for the preprocessor.
            features = features[expected_cols]

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


