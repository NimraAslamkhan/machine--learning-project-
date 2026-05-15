import sys
import os

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        try:

            numerical_columns = [
                "Weekly_Study_Hours",
            ]

            # These features contain string values like Yes/No/Always/Never in the dataset,
            # so they cannot be treated as numeric without mapping/encoding.
            categorical_additional_columns = [
                "Attendance",
                "Reading",
                "Notes",
                "Listening_in_Class",
                "Student_Age",
            ]






            categorical_columns = [
                "Sex",
                "High_School_Type",
                "Scholarship",
                "Additional_Work",
                "Sports_activity",
                "Transportation",
                "Project_work",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Grade"
            # Treat target as REGRESSION target by mapping letter grades to numbers.
            # Dataset contains values like 'AA', 'AB', etc., so direct to_numeric() fails.
            grade_map = {
                "AA": 5.0,
                "AB": 4.5,
                "BB": 4.0,
                "BC": 3.5,
                "CC": 3.0,
                "CD": 2.5,
                "DD": 2.0,
                "DF": 1.5,
                "FF": 1.0,
            }

            train_df[target_column_name] = train_df[target_column_name].map(grade_map)
            test_df[target_column_name] = test_df[target_column_name].map(grade_map)

            train_df = train_df.dropna(subset=[target_column_name])
            test_df = test_df.dropna(subset=[target_column_name])

            if train_df.shape[0] == 0 or test_df.shape[0] == 0:
                raise ValueError(
                    "No rows left after mapping 'Grade' to numeric values. "
                    "Update `grade_map` to cover all Grade values in your dataset."
                )



            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]


            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)