import numpy as np
import pandas as pd
import sys
import os
import dill

from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV


def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
        
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,X_test,y_train,y_test,models):
    try:
        print(models)
        return None
        report={}
        for key,value in models.items():
            model = value
            model.fit(X_train,y_train)
            
            # y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # train_score = r2_score(y_train,y_train_pred)
            test_score = r2_score(y_test,y_test_pred)
            report[key] = test_score
        return report
    
    except Exception as e:
        CustomException(e,sys)

# this method only finetune each algorithm 
def finetune_hyperparameter(params, X_train, y_train, n_iter=20):
    best_models = {}
    for name, config in params.items():
        print(f"Tuning model: {name}")
        random_search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=n_iter,                    # Number of random combinations to try
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        random_search.fit(X_train, y_train)
        best_models[name] = {
            'best_model': random_search.best_estimator_,
            'best_score': -random_search.best_score_,
            'best_params': random_search.best_params_
        }
    return best_models

