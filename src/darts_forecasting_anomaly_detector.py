import yaml
import pandas as pd
import darts
import darts.ad
import darts.models
from darts.models import RegressionModel
from darts import TimeSeries
import logging
import matplotlib.pyplot as plt

from base_anomaly_detector import BaseAnomalyDetector, read_yaml




class DartsForecastingAnomalyDetector(BaseAnomalyDetector):
    """
    Clase DartsForecastingAnomalyDetector para la detección de anomalías utilizando modelos de forecasting de Darts.
    """
    def __init__(self, config: dict, 
                 train_val_data: pd.DataFrame, 
                 test_data: pd.DataFrame = None, 
                 known_anomalies_data: pd.DataFrame = None,
                 data_name: str = None):
        super().__init__(config, train_val_data, test_data, known_anomalies_data, data_name)
        self.anomaly_model = None
        self.forecasting_model = None
        self.anomaly_scorer = None
        self.anomaly_detector = None
        self._initialize_models()


    def set_timestamps(self, train_start=None, train_end=None, val_start=None, val_end=None, test_start=None, test_end=None):
        super().set_timestamps(train_start, train_end, val_start, val_end, test_start, test_end)
        
        # Convert data to Darts TimeSeries
        self.train_series = TimeSeries.from_dataframe(self.get_train_interval_data())
        self.val_series = TimeSeries.from_dataframe(self.get_val_interval_data())
        if self.test_data is not None:
            self.test_series = TimeSeries.from_dataframe(self.get_test_interval_data())
        if self.known_anomalies_data is not None:
            self.known_anomalies_series = TimeSeries.from_dataframe(self.get_known_anomalies_interval_data())

    
   
        
        
    def _initialize_models(self):
        assert 'darts' in self.config, 'darts configuration not found in the config file'
        assert 'forecasting_model' in self.config['darts'], 'forecasting_model configuration not found in the config file' 
        assert 'anomaly_model' in self.config['darts'], 'anomaly_model configuration not found in the config file'
        assert 'anomaly_scorer' in self.config['darts'], 'anomaly_scorer configuration not found in the config file'
        assert 'anomaly_detector' in self.config['darts'], 'anomaly_detector configuration not found in the config file'

        model_class = self.config['darts']['forecasting_model']['class']
        model_parameters = self.config['darts']['forecasting_model']['parameters']
        forecasting_model_class = getattr(darts.models, model_class)
        self.forecasting_model = forecasting_model_class(**model_parameters)

        scorer_class = self.config['darts']['anomaly_scorer']['class']
        scorer_parameters = self.config['darts']['anomaly_scorer']['parameters']
        anomaly_scorer_class = getattr(darts.ad, scorer_class)
        self.anomaly_scorer = anomaly_scorer_class(**scorer_parameters)

        model_class = self.config['darts']['anomaly_model']['class'] 
        anomaly_model_class = getattr(darts.ad, model_class)
        self.anomaly_model = anomaly_model_class(model=self.forecasting_model, 
                                                 scorer=[self.anomaly_scorer])
        
        detector_class = self.config['darts']['anomaly_detector']['class']
        detector_parameters = self.config['darts']['anomaly_detector']['parameters']
        anomaly_detector_class = getattr(darts.ad, detector_class)
        self.anomaly_detector = anomaly_detector_class(**detector_parameters)

    def fit(self):
        try:
            model_name = self.config['darts']['forecasting_model']['parameters']['model_name']
            self.forecasting_model = self.forecasting_model.load_from_checkpoint(model_name=model_name, best=True)
        except:
            pass

        # self.forecasting_model.fit(series=self.train_series,
        #                            val_series=self.val_series)
        
        self.forecasting_model.fit(self.train_series)
        logging.info('Forecasting model fitted')

        anomaly_model_fit_parameters = self.config['darts']['anomaly_model']['fit_parameters']
        self.anomaly_model.fit(self.train_series, **anomaly_model_fit_parameters)
        logging.info('Anomaly detection model fitted')

    def get_val_scores(self, recompute=True, plot=False, verbose=False):
        if self.val_anomaly_scores is None or recompute:
            self.val_anomaly_scores, self.val_prediction_series = self.anomaly_model.score(self.val_series, 
                                                                                            return_model_prediction=True,
                                                                                            verbose=verbose)
            # guardar como pandas                                                                          
            self.val_anomaly_scores = self.val_anomaly_scores.to_dataframe()
            self.val_prediction_series = self.val_prediction_series.to_dataframe()

        if plot:
            self.plot_val_scores()
  
        logging.info('Anomaly scores computed on val data')
        return self.val_anomaly_scores
    
    def get_test_scores(self, recompute=True, plot=False, verbose=False):
        if self.test_anomaly_scores is None or recompute:
            self.test_anomaly_scores, self.test_prediction_series = self.anomaly_model.score(self.test_series, 
                                                                                            return_model_prediction=True,
                                                                                            verbose=verbose)
            
            # guardar como pandas                                                                          
            self.test_anomaly_scores = self.test_anomaly_scores.to_dataframe()
            self.test_prediction_series = self.test_prediction_series.to_dataframe()

        if plot:
            self.plot_test_scores()

        logging.info('Anomaly scores computed on test data')
        return self.test_anomaly_scores
    
   
    
    


