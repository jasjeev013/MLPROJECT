import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException

import dill
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException("Error in Saving Object", e)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    model_report = {}
    for model_name, model in models.items():
        try:
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2
            logging.info(f"Model: {model_name} trained successfully")
        except Exception as e:
            logging.error(f"Error while training model: {model_name}")
            logging.error(e)
    return model_report