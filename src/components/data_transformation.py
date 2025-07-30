import pandas as pd
import  numpy as np
import os
import sys
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer            # it is use to filling missing values
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preporcessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    # this function responsible for data tranformation
    def get_transform(self):
        try:
            num_col = ['reading_score', 'writing_score']
            cat_col = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                        'test_preparation_course']
            
            # make it in standard form
            num_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ])
            
            logging.info('Numerical columns standard scalling complete')
            logging.info('Categorical columns encoding complete')
            
            # Applies num_pipeline to num_col and cat_pipeline to cat_col simultaneously.
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_col),
                ('cat_pipeline',cat_pipeline,cat_col)
            ])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranform(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('Reading train and test data completed')
            
            logging.info("Obtaining preprocessing object")
            pre_processing_obj = self.get_transform()
            
            target_col = 'math_score'
            # numerical_col = ['reading_score', 'writing_score']
            
            train_input_data = train_data.drop(columns=[target_col],axis=1)
            train_target_data = train_data[target_col]
            
            test_input_data = test_data.drop(columns=[target_col],axis=1)
            test_target_data = test_data[target_col]
            
            logging.info('Applying preprocessing object on traning or test dataframe')
            
            input_trained_array = pre_processing_obj.fit_transform(train_input_data)
            input_test_array = pre_processing_obj.transform(test_input_data)
            
            # combine the input & target data
            train_target_array = np.array(train_target_data).reshape(-1, 1)
            test_target_array = np.array(test_target_data).reshape(-1, 1)

            train_arr = np.c_[input_trained_array, train_target_array]
            test_arr = np.c_[input_test_array, test_target_array]
            logging.info('saved preprocessing object')
            
            save_obj(
                file_path = self.data_transformation_config.preporcessor_obj_file_path,
                obj = pre_processing_obj
            )
            
            return (
                train_arr,test_arr,
                self.data_transformation_config.preporcessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            