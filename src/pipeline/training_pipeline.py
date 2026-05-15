import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


# NOTE:
# This file historically contained a PredictPipeline implementation that directly called:
#   preprocessor.transform(features)
# With persisted artifacts + scikit-learn internal attribute changes, this can break.
# The active/robust implementation lives in src/pipeline/prediction_pipeline.py.
#
# app.py imports PredictPipeline from this module, so we re-export it.
from src.pipeline.prediction_pipeline import PredictPipeline


class CustomData:
    def __init__(
        self,
        Student_Age,
        Sex,
        Graduated_High_School_Type,
        Scholarship_Type,
        Additional_Work,
        Sports_Activity,
        Transportation,
        Weekly_Study_Hours,
        Attendance,
        Reading,
        Notes,
        Listening_in_Class,
        Project_Work
    ):

        self.Student_Age = Student_Age
        self.Sex = Sex
        self.Graduated_High_School_Type = Graduated_High_School_Type
        self.Scholarship_Type = Scholarship_Type
        self.Additional_Work = Additional_Work
        self.Sports_Activity = Sports_Activity
        self.Transportation = Transportation
        self.Weekly_Study_Hours = Weekly_Study_Hours
        self.Attendance = Attendance
        self.Reading = Reading
        self.Notes = Notes
        self.Listening_in_Class = Listening_in_Class
        self.Project_Work = Project_Work


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Student_Age": [self.Student_Age],
                "Sex": [self.Sex],
                "High_School_Type": [self.Graduated_High_School_Type],
                "Scholarship": [self.Scholarship_Type],
                "Additional_Work": [self.Additional_Work],
                "Sports_activity": [self.Sports_Activity],
                "Transportation": [self.Transportation],
                "Weekly_Study_Hours": [self.Weekly_Study_Hours],
                "Attendance": [self.Attendance],
                "Reading": [self.Reading],
                "Notes": [self.Notes],
                "Listening_in_Class": [self.Listening_in_Class],
                "Project_work": [self.Project_Work],
            }


            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)