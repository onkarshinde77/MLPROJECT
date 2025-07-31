import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging

from src.utils import load_object



class CustomData:
    def __init__(self,gender:str, race_ethnicity:str ,parental_level_of_education:str ,lunch:str ,test_preparation_course:str, reading_score:int ,writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    def get_data_in_dataframe(self):
        try:
            data_dict = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e,sys)

class PredictPipelines:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            logging.info("predict is begning")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            logging.info("object load")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info("object loaded complete")
            data_scale = preprocessor.transform(features)
            pred = model['best_model'].predict(data_scale)
            logging.info("predict completed")
            return pred
            
        except Exception as e:
            raise CustomException(e,sys)
        