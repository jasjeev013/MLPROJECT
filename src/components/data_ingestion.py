import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig,DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"raw.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingesion_config = DataIngestionConfig()
    
    def intiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:  
            df = pd.read_csv('src/notebook/data/stud.csv')
            logging.info("Data Ingestion Completed")
            
            os.makedirs(os.path.dirname(self.ingesion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingesion_config.raw_data_path,index=False,header=True)
            
            logging.info("Data Saved at: {}".format(self.ingesion_config.raw_data_path))
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingesion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingesion_config.test_data_path,index=False,header=True)
            
            logging.info("Data Split Completed")
            
            return (
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.intiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)


