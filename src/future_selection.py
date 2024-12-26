import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from src.logger import logger
from src.exception import MyException
from config.path_config import *
import sys
from sklearn.preprocessing import LabelEncoder
from utils.common import *

class FeatureEngineering:
    def __init__(self):
        self.data_path =PROCESS_DATA_PATH
        self.df=None
        self.label_mapping={}
        
        
    def load_data(self):
        try:
            logger.info('data loading start')
            self.df=pd.read_csv(self.data_path)
            logger.info('data loadind complete')
        except Exception as e:
            logger.error(f"error while loading data {e}")
            raise MyException(e,sys)    
        
        
        
    def feature_construction(self):
        
        try:
           logger.info('Doing feature construction')    
           self.df['total Delay'] =self.df['Departure Delay in Minutes'] + self.df["Arrival Delay in Minutes"]
           self.df['delay ratio'] =self.df['total Delay']/ (self.df['Flight Distance'] + 1)
           logger.info("featur construction succesffuly")
        except Exception as e:
            logger.error(f"error while feature construction{e}")
            raise MyException(e,sys)
        
        
    def bin_age(self):
        try:
            logger.info("starting bin age of column")
            self.df['Age_group'] =pd.cut(self.df['Age'],bins=[0,18,30,50,100],labels=['child','youngster','adult','senior'])        
            logger.info("end bin age of column") 
        except Exception as e:
            logger.error(f"error while in bining {e}")
            raise MyException(e,sys)   
        
    def econding(self):
        
        try:
           columns_to_encode = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction', 'Age_group']
           logger.info("encoding starte")
           self.df,self.label_mapping=label_encode(self.df,columns_to_encode)
           
           for col,mapping in self.label_mapping.items():
               logger.info(f"mapping for {col}:{mapping}")
               
           logger.info("label mapping completed sucessful")
        
        except Exception as e:
            logger.error(f"error while encoding {e}")
            raise MyException(e,sys)       
  
  
    def feature_selection(self):
        try:
            
            logger.info("Starting Feature Selection")
            X = self.df.drop(columns='satisfaction')
            y = self.df['satisfaction']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            mutual_info = mutual_info_classif(X_train, y_train, discrete_features=True)

            mutual_info_df = pd.DataFrame({
                                    'Feature': X.columns,
                                    'Mutual Information': mutual_info
                                    }).sort_values(by='Mutual Information', ascending=False)
            
            logger.info(f"Mutual Information Table is : \n{mutual_info_df}")

            top_features = mutual_info_df.head(12)['Feature'].tolist()

            self.df = self.df[top_features + ['satisfaction']]
            logger.info(f"Top features : {top_features}")
            logger.info("Feature Selection Sucesfull")

        except Exception as e:
            logger.error(f"Error while feature slection {e}")
            raise MyException("Error while feature selection" , sys)     
        
        
    def save_data(self):
        try:
            logger.info("data saving started")
            os.makedirs(ENGINEERED_DIR,exist_ok=True)
            self.df.to_csv(ENGINEERED_DATA_PATH,index=False)
            logger.info(f'data save successfully in{ENGINEERED_DATA_PATH}')    
            
            
        except Exception as e:
            logger.error(f"error while saving data {e}")
            raise MyException(e,sys)     
        
        
    def run(self):
        try:
            logger.info("starting featureengineering pipeline")
            self.load_data()
            self.feature_construction()
            self.bin_age()
            self.econding()
            self.feature_selection()  
            self.save_data()
            logger.info("feature engineering pipeline completed")
            
        except Exception as e:
            logger.error(f"error while pipeline {e}")
            raise MyException(e,sys) 
        
        finally:
            logger.info('end of fe pipeline')
            
            
            
            
if __name__ =="__main__":
    fe=FeatureEngineering()
    fe.run()            
                  