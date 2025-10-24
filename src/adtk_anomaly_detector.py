import adtk.detector
import pandas as pd
import logging
import matplotlib.pyplot as plt

from base_anomaly_detector import BaseAnomalyDetector, read_yaml
import adtk



class AdtkAnomalyDetector(BaseAnomalyDetector):
    """
    AdtkAnomalyDetector class
    This class implements an anomaly detector using the ADTK (Anomaly Detection Tool Kit) library.
    """
    def __init__(self, config: dict, 
                 train_val_data: pd.DataFrame, 
                 test_data: pd.DataFrame = None, 
                 known_anomalies_data: pd.DataFrame = None,
                 data_name: str = None):
        super().__init__(config, train_val_data, test_data, known_anomalies_data, data_name)
        
        self.anomaly_detector = None
        self._initialize_models()

    def _initialize_models(self):
        assert 'adtk_detector' in self.config, 'adtk configuration not found in the config file'

        if 'pipenet' in self.config['adtk_detector']:
            self._initialize_pipenet(self.config['adtk_detector']['pipenet'])
            pass
        else:
            detector_class = self.config['adtk_detector']['class']
            detector_parameters = self.config['adtk_detector']['parameters']

            adtk_detector_class = getattr(adtk.detector, detector_class)
            self.anomaly_detector = adtk_detector_class(**detector_parameters)
            
    def _initialize_pipenet(self, pipenet_config):
        raise NotImplementedError('Pipenet not implemented yet')

    def fit(self):
        self.anomaly_detector.fit(self.get_train_interval_data())
        logging.info('Anomaly detection model fitted')

    
    def get_val_scores(self, recompute=True, plot=False, verbose=False):
        if self.val_anomaly_scores is None or recompute:
            self.val_anomaly_scores = self.anomaly_detector.detect(self.get_val_interval_data()).astype(float)
        if plot:
            self.plot_val_scores()
        logging.info('Anomaly scores computed on val data')
        return self.val_anomaly_scores

    def get_test_scores(self, recompute=True, plot=False, verbose=False):
        if self.test_anomaly_scores is None or recompute:
            self.test_anomaly_scores = self.anomaly_detector.fit_detect(self.get_test_interval_data()).astype(float)
        if plot:
            self.plot_test_scores()
        logging.info('Anomaly scores computed on test data')
        return self.test_anomaly_scores

   
    