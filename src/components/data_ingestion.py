import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    
    
# DataIngestion class handles data ingestion process
class DataIngestion:
    
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """_summary_

        Raises:
            CustomException: if path to data is not valid

        Returns:
            _type_: tuple contains path to train and test set
        """
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
            
        except Exception as e:
            logging.info("Exception: {}".format(e))
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    # print(train_data,test_data)
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))