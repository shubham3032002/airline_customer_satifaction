from config.path_config import *
import pandas as pd

def load_data(data_path):
    df=pd.read_csv(data_path)
    print(df.shape)
    print(df.head())
    
    
    
load_data(RAW_DATA_PATH)    
    
    