from src.data_ingestion import DataIngestion
from src.data_preprocessing import Datapreprocess
from src.future_selection import FeatureEngineering
from src.model_training import ModelTraining
from config.path_config import *
from src.logger import logger
from src.exception import MyException
import sys

if __name__=="__main__":
    try:
        # Data Ingestion
        ingestion = DataIngestion(raw_data_path=RAW_DATA_PATH,ingested_data_dir=INGESTED_DATA_DIR)
        ingestion.create_ingested_data_dir()
        ingestion.split_data(train_path=TRAIN_DATA_PATH,test_path=TEST_DATA_PATH)
        
        
        #Data processing
        processor = Datapreprocess()
        processor.run()
        
        #feature engineering
        fe=FeatureEngineering()
        fe.run() 
        
        #model training
        model_training = ModelTraining(
        data_path=ENGINEERED_DATA_PATH,
        params_path=PARAMS_PATH,
        model_save_path=MODEL_SAVE_PATH
        )
        model_training.run()
        
    except Exception as e:
            logger.error("Error while running the model training pipeline.")
           
            raise MyException(e, sys)    
        
        