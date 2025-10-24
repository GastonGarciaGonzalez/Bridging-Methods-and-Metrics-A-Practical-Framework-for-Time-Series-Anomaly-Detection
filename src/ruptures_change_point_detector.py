import adtk.detector
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np

from base_anomaly_detector import BaseAnomalyDetector, read_yaml
import ruptures as rpt

def find_elbow(costs):
    """
    Find the "elbow" (knee) index in a sequence of cost values.
    Parameters
    ----------
    costs : array-like of shape (n,)
        Sequence of numeric values representing a cost or error for increasing
        model complexity (or similar). Values are interpreted as 2D points
        (i, costs[i]) where i is the index (0-based). Must contain at least two
        elements.
    Returns
    -------
    int
        The index (0-based) of the point with the maximum perpendicular distance
        to the straight line joining the first and last points (the elbow).
    Raises
    ------
    ValueError
        If `costs` has fewer than 2 elements or if the first and last points are
        identical (which would produce a zero-length baseline and make the elbow
        undefined).
    Notes
    -----
    This uses the "maximum distance to line" heuristic: compute the line between
    the first and last points in the (index, cost) plane and return the index of
    the point farthest from that line. Time complexity is O(n), where n = len(costs).
    Examples
    --------
    >>> find_elbow([10, 8, 6, 5, 4.8, 4.7, 4.6])
    2
    """

    all_coords = np.vstack((range(len(costs)), costs)).T
    first_point = all_coords[0]
    last_point = all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (len(costs), 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    elbow_index = np.argmax(dist_to_line)
    return elbow_index

class RupturesChangePointDetector(BaseAnomalyDetector):
    """
    RupturesChangePointDetector class
    This class implements a change-point detector using the ruptures library.
    http://dev.ipol.im/~truong/ruptures-docs/build/html/index.html
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
        assert 'ruptures_detector' in self.config, 'adtk configuration not found in the config file'

        detector_class_name = self.config['ruptures_detector']['class']
        detector_parameters = self.config['ruptures_detector']['parameters']

        ruptures_detector_class = getattr(rpt, detector_class_name)
        self.anomaly_detector = ruptures_detector_class(**detector_parameters)
        
    

    def fit(self):
        # Compute the median of the training data
        self.train_median = np.median(self.get_train_interval_data())
        
        # Ruptures does not require a separate training step. It operates directly on the signal
        # and fit_predict is used when computing scores.
        logging.info('Anomaly detection model fitted')


    def _compute_scores(self, data, max_n_bkps=20, debug=False):
        signal = data.values.squeeze()

        # Detect for different numbers of breakpoints and select the optimal one
        # The optimal number is chosen as the elbow of the cost curve
        costs = []
        for i in range(0, max_n_bkps + 1):
            my_bkps = self.anomaly_detector.fit_predict(signal, n_bkps=i)
            cost = self.anomaly_detector.cost.sum_of_costs(my_bkps)
            costs.append(cost)
            if debug:
                print(i, cost)

        elbow_index = find_elbow(costs)
        my_bkps = self.anomaly_detector.fit_predict(signal, n_bkps=elbow_index)
        if debug:
            print(f'Elbow index: {elbow_index}, Cost: {costs[elbow_index]}')
            plt.figure()
            plt.plot(costs)
            plt.show()

            fig, ax = rpt.show.display(signal, my_bkps, my_bkps, figsize=(8, 4))
            ax[0].set_title('Change Point Detection')

        # Compute scores for the detected intervals using the optimal number of breakpoints
        # Scores are the relative difference between the interval median and the training data median

        bb = [0] + my_bkps  # add start of the series
        medians = [np.median(signal[bb[i]:bb[i + 1]]) for i in range(len(bb) - 1)]
        relative_diffs = np.abs(medians - self.train_median) / self.train_median

        scores = np.zeros(len(signal))
        for i in range(len(bb) - 1):
            scores[bb[i]:bb[i + 1]] = relative_diffs[i]

        # Return scores as a DataFrame with the same index as the input data
        return pd.DataFrame(data=scores, index=data.index, columns=['anomaly_score'])


    
    def get_val_scores(self, recompute=True, plot=False, verbose=False):
        if self.val_anomaly_scores is None or recompute:
            self.val_anomaly_scores = self._compute_scores(self.get_val_interval_data(), debug=verbose)
        if plot:
            self.plot_val_scores()
        logging.info('Anomaly scores computed on val data')
        return self.val_anomaly_scores

    def get_test_scores(self, recompute=True, plot=False, verbose=False):
        if self.test_anomaly_scores is None or recompute:
            self.test_anomaly_scores = self._compute_scores(self.get_test_interval_data(), debug=verbose)
        if plot:
            self.plot_test_scores()
        logging.info('Anomaly scores computed on test data')
        return self.test_anomaly_scores

    