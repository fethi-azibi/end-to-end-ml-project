import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    
    def initiate_model_trainer(self,train_array, test_array)-> float:
        """reponsible for training model and choosing the best model that has r2_score on test data

        Args:
            train_array (dataframe): train data
            test_array (dataframe): test data

        Raises:
            CustomException: if there is no best model or other error

        Returns:
            float: r2_score for the best model
        """
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            logging.info("model Initiating")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                 models=models, param=params)
            #get best model r2_score
            best_model_score = max(sorted(model_report.values()))
            #get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                logging.info("no best model found")
                raise CustomException("No best model found")
            
            logging.info("saving the best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_quare = r2_score(y_test,predicted)
            return r2_quare

            
            

        except Exception as e:
            # logging.info("Exception {}".format(e))
            CustomException(e, sys)

    
    