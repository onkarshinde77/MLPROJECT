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
from src.utils import save_obj ,evaluate_models , finetune_hyperparameter


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

            params = {
            		"Linear Regression": {
                        # LinearRegression rarely needs param tuning; can keep empty or add 'fit_intercept'
                        'fit_intercept': [True, False]
                    },
                    "Lasso": {
                        'alpha': [0.001, 0.01, 0.1, 1, 10],
                        'selection': ['cyclic', 'random']
                    },
                    "Ridge": {
                        'alpha': [0.01, 0.1, 1, 10, 100],
                        'solver': ['svd', 'cholesky', 'lsqr']
                    },
                    "K-Neighbors Regressor": {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2]  # p=1: Manhattan, p=2: Euclidean distance
                    },
                    "Decision Tree": {
                        'max_depth': [None, 5, 10, 15],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    "AdaBoost Regressor": {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1]
                    },
                    "GradientBoosting": {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5, 7]
                    },
                    "Random Forest Regressor": {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    },
                    "XGBRegressor": {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5, 7]
                    },
                    "CatBoosting Regressor": {
                        'depth': [4, 6, 8],
                        'learning_rate': [0.01, 0.1],
                        'iterations': [500, 1000],
                        'l2_leaf_reg': [1, 3, 5]
                    }
            }
            
            logging.info("Hyperparameter finetune starting!")
            models:dict = finetune_hyperparameter(params=params,X_train=X_train,y_train=y_train)
            logging.info("Hyperparameter finetune completed!")
            model_report : dict = evaluate_models(X_train=X_train,X_test=X_test,
                                                 y_train=y_train,y_test=y_test,models=models)
            return None
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



