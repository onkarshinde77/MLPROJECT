import sys
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from .data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# this class only store the file path of train,test,raw data
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path  : str = os.path.join('artifacts','data.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion method or component')
        try:
            df = pd.read_csv('Notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            logging.info('train test split initiated')
            train_test,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_test.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()
    
    transform = DataTransformation()
    train_arr,test_arr,path_ = transform.initiate_data_tranform(train_data,test_data)
    
    model = ModelTrainer()
    r2_score = model.initiate_model_trainer(train_array=train_arr,test_array=test_arr)
    print(r2_score)
    
            