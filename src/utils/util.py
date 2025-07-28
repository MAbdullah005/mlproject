import os
import sys
import numpy as np
import pandas as pd
from src.expection.expection import CustomExpection
import joblib

def save_obj(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            joblib.dump(obj,f)

    except Exception as e:
        raise CustomExpection(e,sys)