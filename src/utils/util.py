import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.expection.expection import CustomExpection
from src.logger1.logger import logging
import joblib

def save_obj(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            joblib.dump(obj,f)

    except Exception as e:
        raise CustomExpection(e,sys)
    
def model_evaluate(x_train,y_train,x_test,y_test,models,param):
    try:
        logging.info("start model evaluation")
        report={}
        for i in range(len(list(models))):
            logging.info(f"start wroking on model {i}")
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            #model.fit(x_train,y_train)


            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score
            logging.info(f"Finish wroking on model {i} ")

        return report
    except Exception as e:
        raise CustomExpection(e,sys)