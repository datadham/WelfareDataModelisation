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

def interpret(diabetes):
    cluster_label = {0:'Young Energy',
        1:'Vitality and Balance',
        2:'Balanced Wisdom',
        3:'Vigilant Serenity'}
    
    diabetes['cluster_labels'] = diabetes['cluster_labels'].apply(lambda x: cluster_label[x])

    data = pd.read_csv('data/data.csv')

    umap_x = data['umap_x']
    umap_y = data['umap_y']

    return diabetes,umap_x,umap_y