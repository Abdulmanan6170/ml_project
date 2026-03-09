import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Takes input features as DataFrame and returns model predictions
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            legacy_preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")

            print("Loading model and preprocessor...")

            model = load_object(file_path=model_path)
            if os.path.exists(preprocessor_path):
                preprocessor = load_object(file_path=preprocessor_path)
            elif os.path.exists(legacy_preprocessor_path):
                # Backward compatibility for older artifact name.
                preprocessor = load_object(file_path=legacy_preprocessor_path)
            else:
                raise FileNotFoundError(
                    f"Preprocessor not found at '{preprocessor_path}' or '{legacy_preprocessor_path}'"
                )

            print("Transforming input data...")
            data_scaled = preprocessor.transform(features)

            print("Making prediction...")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Converts user input into a pandas DataFrame
        in the exact format expected by the model
        """
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
