import os
import sys
import numpy as np
import pandas as pd
from src.logger1.logger import logging
from src.expection.expection import CustomExpection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils.util import save_obj
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass


#@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str=os.path.join('artifacts/preprocessor.pkl')
    transform_train_data:str=os.path.join('artifacts/transform_train.csv')
    transform_test_data:str=os.path.join('artifacts/transform_test.csv')


class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig):
        self.data_transformation_config=data_transformation_config
    
    def get_data_transform_object(self):

        """
        performa data transofrmation on categorial and numrical  column 
        """
        try:
            numrical_column=["writing_score","reading_score"]

            categorial_column=["gender","race_ethnicity",
                               "parental_level_of_education","lunch",
                               "test_preparation_course"]
            
            logging.info(f"categorial columns {categorial_column}")
            logging.info(f"numrical column {numrical_column}")
            
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler(with_mean=False))
            ]
            )
            logging.info("Numrical transformation completed pipeline")

            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ]
            )

            logging.info("categorial encodeing completed pipeline")


            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numrical_column),
                ("cat_pipeline",cat_pipeline,categorial_column)
            ]
            )

            return preprocessor
        except Exception as e:
            raise CustomExpection(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("reading train and test spilt")

            logging.info("Obtaning prrproceessing object")

            preprocessing_obj=self.get_data_transform_object()

            target_column_name="math_score"
            numrical_columns=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)

            target_feature_test_df=test_df[target_column_name]
            target_feature_train_df=train_df[target_column_name]

            logging.info("applying preprocessor on trainign dataframe and test datafreame"
                         )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info(f"saved preprocessing object")
            save_obj(file_path=self.data_transformation_config.preprocessor_file_path,
                        obj=preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomExpection(e,sys)
