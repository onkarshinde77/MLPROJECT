import os
import sys

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_obj ,evaluate_models , finetune_hyperparameter

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training data and test data')
            
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            params = {
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False],
                    }
                },
                'Lasso': {
                    'model': Lasso(max_iter=5000, random_state=42),
                    'params': {
                        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                        'selection': ['cyclic', 'random'],
                        'max_iter': [1000, 5000]
                    }
                },
                'Ridge': {
                    'model': Ridge(max_iter=5000, random_state=42),
                    'params': {
                        'alpha': [0.01, 0.1, 1, 10, 100],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
                        'max_iter': [1000, 5000]
                    }
                },
                'KNeighborsRegressor': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9, 11],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2],  # 1=Manhattan, 2=Euclidean
                        'leaf_size': [20, 30, 40]
                    }
                },
                'DecisionTree': {
                    'model': DecisionTreeRegressor(random_state=42),
                    'params': {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  # sklearn >=1.0 uses squared_error instead of mse
                        'max_depth': [None, 5, 10, 20, 50],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 5],
                        'max_features': [None,'sqrt', 'log2']
                    }
                },
                'RandomForest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt', 'log2']
                    }
                },
                'AdaBoost': {
                    'model': AdaBoostRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.001, 0.01, 0.1, 1],
                        'loss': ['linear', 'square', 'exponential']
                    }
                },
                'GradientBoosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 150, 200],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.6, 0.8, 1.0],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['auto', 'sqrt', 'log2']
                    }
                },
                'XGBoost': {
                    'model': XGBRegressor(random_state=42, verbosity=0, objective='reg:squarederror'),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7, 9],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0],
                        'gamma': [0, 1, 5],
                        'reg_alpha': [0, 0.01, 0.1],
                        'reg_lambda': [1, 1.5, 2]
                    }
                },
                'CatBoost': {
                    'model': CatBoostRegressor(verbose=0, random_state=42),
                    'params': {
                        'iterations': [100, 200, 500],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'depth': [4, 6, 8, 10],
                        'l2_leaf_reg': [1, 3, 5, 7, 9],
                        'border_count': [32, 50, 100],
                        'thread_count': [4],  # adjust to your CPU cores
                        'random_strength': [1, 20, 50]
                    }
                }
            }

            logging.info("Hyperparameter finetune starting!")
            models:dict = finetune_hyperparameter(params=params,X_train=X_train,y_train=y_train)
            logging.info("Hyperparameter finetune completed!")
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



