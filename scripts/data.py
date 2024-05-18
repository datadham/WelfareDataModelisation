import pandas as pd
import numpy as np
import os 


def getdata(path_to_folder):
    diabetes_path = os.path.join(path_to_folder, 'diabetes_data.csv')
    stroke_path = os.path.join(path_to_folder, 'stroke_data.csv')
    hypertension_path = os.path.join(path_to_folder, 'hypertension_data.csv')
    
    diabetes = pd.read_csv(diabetes_path)
    stroke = pd.read_csv(stroke_path)
    hypertension = pd.read_csv(hypertension_path)
    
    return diabetes, stroke, hypertension