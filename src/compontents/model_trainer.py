import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from src.utils.util import model_evaluate,save_obj
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.expection.expection import CustomExpection
from src.logger1.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig):
        self.model_trainer_cofig=model_trainer_config

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Start model Trainre")
            x_trian,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("done trian test split")

            models={"Random":RandomForestRegressor(),
                    "Decision":DecisionTreeRegressor(),
                    "Gradient Boosting":GradientBoostingRegressor(),
                    "k-Neighbour":KNeighborsRegressor(),
                    "CatBoosting":CatBoostRegressor(),
                    "AdaBoost":AdaBoostRegressor(),
                    "XGBRegressor":XGBRegressor(),
                    "Linear Regression":LinearRegression()
                    }
            
            params={
                "Decision": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "k-Neighbour":{},
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }




            logging.info("start model evaluation process")
            model_report:dict=model_evaluate(x_train=x_trian,y_train=y_train,
                                             x_test=x_test,y_test=y_test,
                                             models=models,param=params)
            ## to get best model name  and score for dict 
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.75:
                raise CustomExpection("No Best model found")
            
            logging.info("Best model found on both Traning and Testing dataset")

            save_obj(file_path=self.model_trainer_cofig.trained_model_file_path,
                     obj=best_model)
            
            predict=best_model.predict(x_test)
            r2=r2_score(y_test,predict)
            logging.info("done model tariner and got r2 socre")
            
            return r2

        except Exception as e:
            pass