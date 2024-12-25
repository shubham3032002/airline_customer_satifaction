import pandas as pd
from config.path_config import *
from src.logger import logger
from src.exception import MyException
import sys


class Datapreprocess:
    
    def __init__(self):
        self.train_path=TRAIN_DATA_PATH
        self.process_data_path=PROCESS_DATA_PATH
        
    def load_data(self):
        try:
          logger.info({"data Precossing started"})
          df=pd.read_csv(self.train_path)
          logger.info(f"Data read succesfull : Data shape : {df.shape}")
          return df
        except Exception as e:
            logger.error("Problem while Loading Data")
            raise MyException("Error while loading data : ",sys)
        
        
    def drop_unuse_col(self,df,column):
        try:
            logger.info(f"droping unused colums {column}")
            df = df.drop(columns = column, axis=1)
            logger.info(f"Columns dropped Sucesfully : Shape = {df.shape}")
            return df
        except Exception as e:
            logger.error("Problem while dropping columns")
            raise MyException("Error while sropping columns : ",sys)
              
    def handle_outliers(self, df , columns):
        try:
            logger.info(f"Handling outliers : Columns = {columns}")
            for column in columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info(f"Outliers handled  Sucesffuully : {df.shape}")
            return df
        
        except Exception as e:
            logger.error("Problem while Outlier handling")
            raise MyException("Error while outlier handling : ",sys)          
              
    def save_data(self,df):
        try:
            os.makedirs( PROCEESED_DIR , exist_ok=True)
            df.to_csv(self.process_data_path , index=False)
            logger.info("Processed data saved sucesfully")
        
        except Exception as e:
            logger.error("Problem while saving data")
            raise MyException("Error while saving data : ",sys)
        
        
        
    def run(self):
        try:
            logger.info("starting pipeline of data processing")
            df=self.load_data()
            df=self.drop_unuse_col(df,['MyUnknownColumn','id'])
            columns_to_handel= ['Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes', 'Checkin service']
            df = self.handle_outliers(df , columns_to_handel)
            self.save_data(df)
            logger.info("Data Proccesing Pipeline COmpleted Sucessfully")
        
        except MyException as ce:
            logger.error(f"Error ocuured in Data Processing Pipleine : {str(ce)}")    


if __name__=="__main__":
    processor = Datapreprocess()
    processor.run()
    