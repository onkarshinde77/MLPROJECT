import os
import sys

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor , GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_obj ,evaluate_models


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training data and test data')
            # X_train,X_test,y_train,y_test= train_test_split(
            #     train_array[:,:-1],
            #     train_array[:,-1],
            #     test_array[:,:-1],
            #     test_array[:,-1],
            # )
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "GradientBoosting" : GradientBoostingRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(n_estimators=100), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
            
            model_report : dict = evaluate_models(X_train=X_train,X_test=X_test,
                                                 y_train=y_train,y_test=y_test,models=models)
            
            best_model_name = max(model_report,key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            # also write like this instead of both line
            # best_model, best_score = max(model_report.items(), key=lambda x: x[1])
            
            if best_model_score <0.6:
                raise CustomException('No Best Model Found',sys)
            logging.info('Best model found in training & test dataset')
            save_obj(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            return best_model_score
            
        except Exception as e:
            raise CustomException(e,sys)
    
    