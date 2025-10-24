import os
import pandas as pd
import logging
import yaml
# from adtk.detector import SeasonalAD, ThresholdAD
# from adtk.data import validate_series
# from darts import TimeSeries

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def read_yaml(file_path):
    """
    Reads a YAML configuration file and returns its contents as a Python object.
    Args:
        file_path (str): The path to the YAML file to be read.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(file_path, 'r') as file:
        #config = yaml.safe_load(file)
        config = yaml.load(file, Loader=yaml.FullLoader) # needed to load tuples from yaml
                           
    return config


class BaseAnomalyDetector(ABC):
    def __init__(self, config: dict, 
                 train_val_data: pd.DataFrame, 
                 test_data: pd.DataFrame = None, 
                 known_anomalies_data: pd.DataFrame = None,
                 data_name: str = None):
        
        self.config = config
        self.train_val_data = train_val_data
        self.test_data = test_data
        self.known_anomalies_data = known_anomalies_data
        self.data_name = data_name


        self.val_prediction_series = None
        self.test_prediction_series = None

        self.val_anomaly_scores = None
        self.test_anomaly_scores = None
        self.test_anomaly_detection = None
        
        self.test_metric = None

        self.plotting_backend = 'plotly'

        self.train_val_ratio = 0.8  # para particion si no se especifica timestamps en el config (ver _initialize_data)
        # for partitioning if timestamps are not specified in the config (see _initialize_data)

        # initialization
        self._initialize_logging()
        self._initialize_data()
        self._initialize_models()

    def _initialize_logging(self):
        if 'logging' in self.config:
            logging_config = self.config['logging']
            os.makedirs(os.path.dirname(logging_config['file_path']), exist_ok=True)
            logging.basicConfig(filename=logging_config['file_path'], level=logging_config['level'])
            logging.info('Logging initialized')


    def _initialize_data(self):
        """
        Initialize data required by the anomaly detector.

        This method performs the following actions:
        1. Validates initialization data by calling `self._validate_data()`.
        2. Sets timestamps for training, validation and test datasets.
           - If 'timestamps' is present in the configuration (`self.config`), those values are used.
           - If 'timestamps' is not present, it assumes train/val and test are disjoint and splits
             the train/val portion according to `self.train_val_ratio`.

        Notes:
        - `self._validate_data()` is not fully implemented yet.
        - `self.set_timestamps()` performs basic correctness checks on the timestamps.
        """

        # validate that initialization data is correct
        self._validate_data()  # TODO: finish this function

        # Normalize all DatetimeIndex objects to UTC-aware timestamps to avoid
        # comparisons between tz-naive and tz-aware datetimes later on.
        # The nested helper intentionally keeps behavior simple for now.
        def _ensure_index_utc(idx: pd.DatetimeIndex):
            return idx

        self.train_val_data.index = _ensure_index_utc(self.train_val_data.index)
        if self.test_data is not None:
            self.test_data.index = _ensure_index_utc(self.test_data.index)
        if self.known_anomalies_data is not None:
            self.known_anomalies_data.index = _ensure_index_utc(self.known_anomalies_data.index)

        # Set timestamps. Use utc-aware parsing to keep consistency
        if 'timestamps' in self.config:
            self.set_timestamps(
                pd.to_datetime(self.config['timestamps']['train']['start'], utc=False),
                pd.to_datetime(self.config['timestamps']['train']['end'], utc=False),
                pd.to_datetime(self.config['timestamps']['val']['start'], utc=False),
                pd.to_datetime(self.config['timestamps']['val']['end'], utc=False),
                pd.to_datetime(self.config['timestamps']['test']['start'], utc=False),
                pd.to_datetime(self.config['timestamps']['test']['end'], utc=False),
            )
        else:
            # If timestamps are not provided in config, assume train/val and test are disjoint
            # and split train/val according to self.train_val_ratio
            len_train_val_data = self.train_val_data.shape[0]

            self.set_timestamps(
                self.train_val_data.index[0],
                self.train_val_data.index[int(len_train_val_data * self.train_val_ratio)],
                self.train_val_data.index[int(len_train_val_data * self.train_val_ratio) + 1],
                self.train_val_data.index[-1],
                self.test_data.index[0],
                self.test_data.index[-1],
            )


                
            

    def _validate_data(self):
        # TODO: finish this function

        # the index of the data series must be a DateTimeIndex
        assert self.train_val_data.index.__class__== pd.DatetimeIndex
        assert self.test_data.index.__class__ == pd.DatetimeIndex
        if self.known_anomalies_data is not None:
            assert self.known_anomalies_data.index.__class__ == pd.DatetimeIndex

        # series must not contain duplicate indices


        # series must have the correct columns


        

   
    def _validate_timestamps(self, train_start=None, train_end=None, val_start=None, val_end=None, test_start=None, test_end=None):
        """
        Validate and adjust timestamps for the training, validation and test sets.

        Parameters
        ----------
        train_start : str or datetime, optional
            Start date of the training set. If not provided, `self.train_start` is used.
        train_end : str or datetime, optional
            End date of the training set. If not provided, `self.train_end` is used.
        val_start : str or datetime, optional
            Start date of the validation set. If not provided, `self.val_start` is used.
        val_end : str or datetime, optional
            End date of the validation set. If not provided, `self.val_end` is used.
        test_start : str or datetime, optional
            Start date of the test set. If not provided, `self.test_start` is used.
        test_end : str or datetime, optional
            End date of the test set. If not provided, `self.test_end` is used.

        """

        aux_train_start = pd.to_datetime(train_start, utc=False) if train_start is not None else self.train_start
        aux_train_end = pd.to_datetime(train_end, utc=False) if train_end is not None else self.train_end
        aux_val_start = pd.to_datetime(val_start, utc=False) if val_start is not None else self.val_start
        aux_val_end = pd.to_datetime(val_end, utc=False) if val_end is not None else self.val_end
        aux_test_start = pd.to_datetime(test_start, utc=False) if test_start is not None else self.test_start
        aux_test_end = pd.to_datetime(test_end, utc=False) if test_end is not None else self.test_end

        assert aux_train_start < aux_train_end
        assert aux_val_start < aux_val_end
        assert aux_test_start < aux_test_end

        assert aux_train_end < aux_val_start
        assert aux_val_end < aux_test_start

        assert aux_train_start >= self.train_val_data.index[0]
        assert aux_val_end <= self.train_val_data.index[-1]

        assert aux_test_start >= self.test_data.index[0]
        assert aux_test_end <= self.test_data.index[-1]
               


    def plot_data(self, all_data=False):
        if self.plotting_backend=='matplotlib':
            self._plot_data_matplotlib(all_data)
        elif self.plotting_backend=='plotly':
            self._plot_data_plotly(all_data)
        else:
            raise ValueError('No a valid plotting backend')    


    def _plot_data_matplotlib(self, all_data=False):
        """
        Generate and display plots for training, validation, test and known anomalies data.

        Parameters
        ----------
        all_data : bool, optional
            If True, plots all available data (train/val, test and known anomalies).
            If False, plots the intervals defined by the configured timestamps.

        Details
        -------
        - When `all_data` is True the full series for train/val, test and known anomalies are plotted.
        - When `all_data` is False only the intervals defined by the timestamps are plotted.
        - Vertical lines are added to indicate train/val/test boundaries.
        - X-axis limits are aligned across subplots.

        Returns
        -------
        None
        """
        
        fig, ax = plt.subplots(nrows=3, figsize=(12,5) if all_data else (10,7), sharex=True)
        
        xlims = []
        if all_data:
            self.train_val_data.plot(ax=ax[0])
            xlims.append(ax[0].get_xlim())
            self.test_data.plot(ax=ax[1])
            xlims.append(ax[1].get_xlim())
            self.known_anomalies_data.plot(ax=ax[2])
            xlims.append(ax[2].get_xlim())
            ax[0].legend(['train_val_data'])
            ax[1].legend(['test_data'])
            ax[2].legend(['known_anomalies'])
        else:
            self.get_train_interval_data().plot(ax=ax[0])
            self.get_val_interval_data().plot(ax=ax[0])
            xlims.append(ax[0].get_xlim())
            if self.test_data is not None:
                self.get_test_interval_data().plot(ax=ax[1])
                xlims.append(ax[1].get_xlim())
            if self.known_anomalies_data is not None:
                self.get_known_anomalies_interval_data().plot(ax=ax[2])
                xlims.append(ax[2].get_xlim())
            ax[0].legend(['train_data', 'val_data'])
            if self.test_data is not None:
                ax[1].legend(['test_data'])
            if self.known_anomalies_data is not None:
                ax[2].legend(['known_anomalies'])

        for axis in ax:
            axis.axvline(self.train_start, color='r', linestyle='--')
            axis.axvline(self.train_end, color='r', linestyle='--')
            axis.axvline(self.val_start, color='g', linestyle='--')
            axis.axvline(self.val_end, color='g', linestyle='--')
            axis.axvline(self.test_start, color='m', linestyle='--')
            axis.axvline(self.test_end, color='m', linestyle='--')
        
        # Compute the union of x-limits
        min_xlim = min(xlim[0] for xlim in xlims)
        max_xlim = max(xlim[1] for xlim in xlims)
        for axis in ax:
            axis.set_xlim(min_xlim, max_xlim)
        
        fig.suptitle(f'{self.data_name}: {"Input data" if all_data else "Train, Val, Test and Known Anomalies"}')
        plt.show()
        
    def _plot_data_plotly(self, all_data: bool = False):
        """
        Interactive Plotly version of `plot_data`.

        Parameters
        ----------
        all_data : bool, optional
            If True, plots full series for train/val, test and known anomalies.
            If False, plots the configured intervals.

        Notes
        -----
        - Mirrors the layout and vertical timestamp indicators from `plot_data`.
        - Produces an interactive Plotly figure and calls `fig.show()`.
        """

        # Create 3-row subplot, shared x-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04,
                            subplot_titles=[f'{self.data_name}: Train/Val',
                                            f'{self.data_name}: Test',
                                            f'{self.data_name}: Known Anomalies'])

        x_ranges = []

        # Helper to add all columns of a dataframe as separate traces
        def _add_df_traces(df: pd.DataFrame, row: int, name_prefix: str = None):
            if df is None or df.empty:
                return
            for col in df.columns:
                trace_name = f'{name_prefix + " - " if name_prefix else ""}{col}'
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=trace_name), row=row, col=1)
            x_ranges.append((df.index.min(), df.index.max()))

        if all_data:
            # full series
            _add_df_traces(self.train_val_data, row=1, name_prefix='train_val')
            _add_df_traces(self.test_data, row=2, name_prefix='test')
            _add_df_traces(self.known_anomalies_data, row=3, name_prefix='known_anomalies')
        else:
            # intervals: train and val plotted on the first row (as separate traces)
            train_df = self.get_train_interval_data()
            val_df = self.get_val_interval_data()
            _add_df_traces(train_df, row=1, name_prefix='train')
            _add_df_traces(val_df, row=1, name_prefix='val')
            if self.test_data is not None:
                test_df = self.get_test_interval_data()
                _add_df_traces(test_df, row=2, name_prefix='test')
            if self.known_anomalies_data is not None:
                known_df = self.get_known_anomalies_interval_data()
                _add_df_traces(known_df, row=3, name_prefix='known_anomalies')

        # Add vertical lines for boundaries (span full paper height)
        shapes = []
        def _vline(ts, color):
            if ts is None:
                return
            xstr = pd.to_datetime(ts)
            shapes.append(dict(type='line', x0=xstr, x1=xstr, y0=0, y1=1, yref='paper',
                               line=dict(color=color, width=1, dash='dash')))

        _vline(self.train_start, 'red')
        _vline(self.train_end, 'red')
        _vline(self.val_start, 'green')
        _vline(self.val_end, 'green')
        _vline(self.test_start, 'magenta')
        _vline(self.test_end, 'magenta')

        if shapes:
            fig.update_layout(shapes=shapes)

        # Align x-axis ranges across subplots if we collected any
        if x_ranges:
            min_x = min(r[0] for r in x_ranges)
            max_x = max(r[1] for r in x_ranges)
            # format as strings for Plotly
            fig.update_xaxes(range=[min_x, max_x])

        fig.update_layout(height=700, showlegend=True, title_text=f'{self.data_name}: {"Input data" if all_data else "Train, Val, Test and Known Anomalies"}')
        fig.show()
        

    def get_timestamps(self, as_string=False, as_pd_dataframe=False):
        """
        Get timestamps for the training, validation and test periods.

        Parameters
        ----------
        as_string : bool
            If True, returns timestamps as formatted strings.
        as_pd_dataframe : bool
            If True, returns timestamps as a pandas DataFrame.

        Returns
        -------
        tuple or pandas.DataFrame
            Depending on the flags, returns a tuple of datetimes, a tuple of formatted strings, or a DataFrame.
        """

        if as_pd_dataframe:
            return pd.DataFrame({'train_start': [self.train_start], 'train_end': [self.train_end], 'val_start': [self.val_start], 'val_end': [self.val_end], 'test_start': [self.test_start], 'test_end': [self.test_end]})
        elif as_string:
            return self.train_start.strftime('%Y-%m-%d %H:%M:%S'), self.train_end.strftime('%Y-%m-%d %H:%M:%S'), self.val_start.strftime('%Y-%m-%d %H:%M:%S'), self.val_end.strftime('%Y-%m-%d %H:%M:%S'), self.test_start.strftime('%Y-%m-%d %H:%M:%S'), self.test_end.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return self.train_start, self.train_end, self.val_start, self.val_end, self.test_start, self.test_end

    def set_timestamps(self, train_start=None, train_end=None, val_start=None, val_end=None, test_start=None, test_end=None):
        #self._validate_timestamps(train_start, train_end, val_start, val_end, test_start, test_end)

        if train_start is not None:
            self.train_start = pd.to_datetime(train_start, utc=False)
        if train_end is not None:
            self.train_end = pd.to_datetime(train_end, utc=False)
        if val_start is not None:
            self.val_start = pd.to_datetime(val_start, utc=False)
        if val_end is not None:
            self.val_end = pd.to_datetime(val_end, utc=False)
        if test_start is not None:
            self.test_start = pd.to_datetime(test_start, utc=False)
        if test_end is not None:
            self.test_end = pd.to_datetime(test_end, utc=False)
        
        logging.info('Timestamps set')

    def get_train_interval_data(self):
        """
        Get training interval data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the training interval defined by `train_start` and `train_end`.
        """
        
        return self.train_val_data[self.train_start:self.train_end]
    
    def get_val_interval_data(self):
        """
        Get validation interval data.

        Returns
        -------
        pandas.DataFrame
            A subset of `train_val_data` from `val_start` to `val_end`.
        """
        return self.train_val_data[self.val_start:self.val_end]
    
    def get_test_interval_data(self):
        """
        Get test interval data.

        Returns
        -------
        pandas.DataFrame
            A subset of `test_data` from `test_start` to `test_end`.
        """
        return self.test_data[self.test_start:self.test_end]
    
    def get_known_anomalies_interval_data(self):
        """
        Get known-anomalies interval data.

        This returns known anomaly records within the interval defined by `test_start` and `test_end`.
        If no known anomalies are available, returns `None`.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame with known anomalies in the specified interval, or `None` if unavailable.
        """
        if self.known_anomalies_data is not None:
            return self.known_anomalies_data[self.test_start:self.test_end]
        else:
            return None


    def plot_val_scores(self):
        if self.plotting_backend=='matplotlib':
            self._plot_val_scores_matplotlib()
        elif self.plotting_backend=='plotly':
            self._plot_val_scores_plotly()
        else:
            raise ValueError('No a valid plotting backend')
        
    def plot_test_scores(self):
        if self.plotting_backend=='matplotlib':
            self._plot_test_scores_matplotlib()
        elif self.plotting_backend=='plotly':
            self._plot_test_scores_plotly()
        else:
            raise ValueError('No a valid plotting backend')    

    def _plot_val_scores_matplotlib(self):
        """
        Generate and display a plot of validation scores.

        This creates two subplots: validation data (and predictions if available) and validation anomaly scores.
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,5), sharex=True)
        self.get_val_interval_data().plot(ax=ax1)
        if self.val_prediction_series is not None:
            self.val_prediction_series.plot(ax=ax1)
            ax1.legend(['val_data', 'val_prediction'])
        else:
            ax1.legend(['val_data'])
        self.val_anomaly_scores.plot(ax=ax2)
        ax2.legend(['val_anomaly_scores'])
        fig.suptitle(f'{self.data_name}: Validation scores')
        plt.show()

    def _plot_test_scores_matplotlib(self):
        """
        Generate and display a plot of test scores.

        This creates two subplots: test data (and predictions if available) and test anomaly scores.
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,5), sharex=True)
        self.get_test_interval_data().plot(ax=ax1)
        if self.test_prediction_series is not None:
            self.test_prediction_series.plot(ax=ax1)
            ax1.legend(['test_data', 'test_prediction'])
        else:
            ax1.legend(['test_data'])
        self.test_anomaly_scores.plot(ax=ax2)
        ax2.legend(['test_anomaly_scores'])
        fig.suptitle(f'{self.data_name}: Test scores')
        plt.show()

    def _plot_val_scores_plotly(self):
        """
        Plotly interactive version of `plot_val_scores`.
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=[f'{self.data_name}: Validation data', f'{self.data_name}: Validation anomaly scores'])

        # Validation data
        val_df = self.get_val_interval_data()
        if val_df is not None and not val_df.empty:
            for col in val_df.columns:
                fig.add_trace(go.Scatter(x=val_df.index, y=val_df[col], mode='lines', name=f'val - {col}'), row=1, col=1)

        # Predictions (if available)
        if self.val_prediction_series is not None:
            if isinstance(self.val_prediction_series, pd.DataFrame):
                for col in self.val_prediction_series.columns:
                    fig.add_trace(go.Scatter(x=self.val_prediction_series.index, y=self.val_prediction_series[col], mode='lines', name=f'val_pred - {col}', line=dict(dash='dash')), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.val_prediction_series.index, y=self.val_prediction_series, mode='lines', name='val_prediction', line=dict(dash='dash')), row=1, col=1)

        # Validation anomaly scores
        if self.val_anomaly_scores is not None:
            if isinstance(self.val_anomaly_scores, pd.DataFrame):
                for col in self.val_anomaly_scores.columns:
                    fig.add_trace(go.Scatter(x=self.val_anomaly_scores.index, y=self.val_anomaly_scores[col], mode='lines', name=f'val_score - {col}'), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.val_anomaly_scores.index, y=self.val_anomaly_scores, mode='lines', name='val_anomaly_scores'), row=2, col=1)

        fig.update_layout(height=600, showlegend=True, title_text=f'{self.data_name}: Validation scores')
        fig.show()

    def _plot_test_scores_plotly(self):
        """
        Plotly interactive version of `plot_test_scores`.
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=[f'{self.data_name}: Test data', f'{self.data_name}: Test anomaly scores'])

        # Test data
        test_df = self.get_test_interval_data()
        if test_df is not None and not test_df.empty:
            for col in test_df.columns:
                fig.add_trace(go.Scatter(x=test_df.index, y=test_df[col], mode='lines', name=f'test - {col}'), row=1, col=1)

        # Test predictions (if available)
        if self.test_prediction_series is not None:
            if isinstance(self.test_prediction_series, pd.DataFrame):
                for col in self.test_prediction_series.columns:
                    fig.add_trace(go.Scatter(x=self.test_prediction_series.index, y=self.test_prediction_series[col], mode='lines', name=f'test_pred - {col}', line=dict(dash='dash')), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.test_prediction_series.index, y=self.test_prediction_series, mode='lines', name='test_prediction', line=dict(dash='dash')), row=1, col=1)

        # Test anomaly scores
        if self.test_anomaly_scores is not None:
            if isinstance(self.test_anomaly_scores, pd.DataFrame):
                for col in self.test_anomaly_scores.columns:
                    fig.add_trace(go.Scatter(x=self.test_anomaly_scores.index, y=self.test_anomaly_scores[col], mode='lines', name=f'test_score - {col}'), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.test_anomaly_scores.index, y=self.test_anomaly_scores, mode='lines', name='test_anomaly_scores'), row=2, col=1)

        fig.update_layout(height=600, showlegend=True, title_text=f'{self.data_name}: Test scores')
        fig.show()

    

    def get_test_anomaly_detection(self, thresholding: str = None, threshold_value: float = None, 
                                   plot: bool = False, verbose: bool = False, recompute: bool = True, 
                                   percentile: float = 0.95) -> pd.DataFrame:
        """
        Detect anomalies in test data using different thresholding methods.

        Parameters
        ----------
        thresholding : str, optional
            Thresholding method. One of 'relative_to_test_max', 'relative_to_val_max',
            'relative_to_val_percentile', 'test_percentile' or 'hard'.
        threshold_value : float, optional
            Threshold value to apply. Required for the supported methods.
        plot : bool, optional
            If True, plots test data, anomaly scores and detections.
        verbose : bool, optional
            If True, prints additional information during execution.
        recompute : bool, optional
            If True, recomputes validation and test scores before thresholding.
        percentile : float, optional
            Percentile to use when `relative_to_val_percentile` or 'test_percentile' is selected (default 0.95).

        Returns
        -------
        pd.DataFrame
            DataFrame with the binary anomaly detections for the test interval.

        Raises
        ------
        ValueError
            If an unsupported thresholding method is provided.
        """
        
        if recompute:
            _ = self.get_test_scores()        

        if thresholding == 'relative_to_test_max' and threshold_value is not None: 
            threshold = self.test_anomaly_scores.max().values[0] *  threshold_value
            self.test_anomaly_detection = (self.test_anomaly_scores >= threshold).astype(int)
        elif thresholding == 'relative_to_val_max' and threshold_value is not None:
            if recompute:
                _ = self.get_val_scores()
            threshold = self.val_anomaly_scores.max().values[0] * threshold_value
            self.test_anomaly_detection = (self.test_anomaly_scores >= threshold).astype(int)
        elif thresholding == 'relative_to_val_percentile' and threshold_value is not None:
            if recompute:
                _ = self.get_val_scores()
            threshold = self.val_anomaly_scores.quantile(percentile).values[0] * threshold_value
            self.test_anomaly_detection = (self.test_anomaly_scores >= threshold).astype(int)
        elif thresholding == 'test_percentile':
            threshold = self.test_anomaly_scores.quantile(percentile).values[0]
            self.test_anomaly_detection = (self.test_anomaly_scores >= threshold).astype(int)
        elif thresholding == 'hard' and threshold_value is not None:
            threshold = threshold_value
            self.test_anomaly_detection = (self.test_anomaly_scores >= threshold).astype(int)
        else:
            raise ValueError(f'Thresholding method not implemented: {thresholding}')

        if plot:
            if self.plotting_backend == 'mattplotlib':
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(12, 7), sharex=True)
                self.get_test_interval_data().plot(label='test_data', ax=ax1)
                ax1.legend(['test_data'])
                self.test_anomaly_scores.plot(label='test_anomaly_scores', ax=ax2)   
                ax2.legend(['test_anomaly_scores'])
                self.test_anomaly_detection.plot(label='test_anomaly_detection', ax=ax3)   
                ax3.legend(['test_anomaly_detection'])
                self.get_known_anomalies_interval_data().plot(label='known_anomalies_data', ax=ax4)
                ax4.legend(['known_anomalies_data'])
                fig.suptitle(f'{self.data_name}: Detected anomalies')
                plt.show()
            elif self.plotting_backend == 'plotly':
                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    subplot_titles=[
                        f'{self.data_name}: Test data',
                        f'{self.data_name}: Test anomaly scores',
                        f'{self.data_name}: Test anomaly detection',
                        f'{self.data_name}: Known anomalies'
                    ]
                )

                # Row 1: Test data
                test_df = self.get_test_interval_data()
                if test_df is not None and not test_df.empty:
                    for col in test_df.columns:
                        fig.add_trace(
                            go.Scatter(x=test_df.index, y=test_df[col], mode='lines', name=f'test - {col}'),
                            row=1, col=1
                        )

                # Row 2: Test anomaly scores
                if self.test_anomaly_scores is not None:
                    if isinstance(self.test_anomaly_scores, pd.DataFrame):
                        for col in self.test_anomaly_scores.columns:
                            fig.add_trace(
                                go.Scatter(x=self.test_anomaly_scores.index, y=self.test_anomaly_scores[col],
                                           mode='lines', name=f'test_score - {col}'),
                                row=2, col=1
                            )
                    else:
                        fig.add_trace(
                            go.Scatter(x=self.test_anomaly_scores.index, y=self.test_anomaly_scores,
                                       mode='lines', name='test_anomaly_scores'),
                            row=2, col=1
                        )

                # Row 3: Binary detection (use step-like line for clarity)
                if self.test_anomaly_detection is not None:
                    if isinstance(self.test_anomaly_detection, pd.DataFrame):
                        for col in self.test_anomaly_detection.columns:
                            fig.add_trace(
                                go.Scatter(x=self.test_anomaly_detection.index, y=self.test_anomaly_detection[col],
                                           mode='lines+markers', line=dict(shape='hv'), name=f'detection - {col}'),
                                row=3, col=1
                            )
                    else:
                        fig.add_trace(
                            go.Scatter(x=self.test_anomaly_detection.index, y=self.test_anomaly_detection,
                                       mode='lines+markers', line=dict(shape='hv'), name='test_anomaly_detection'),
                            row=3, col=1
                        )

                # Row 4: Known anomalies (as markers)
                known_df = self.get_known_anomalies_interval_data()
                if known_df is not None and not known_df.empty:
                    for col in known_df.columns:
                        fig.add_trace(
                            go.Scatter(x=known_df.index, y=known_df[col], mode='markers', marker=dict(symbol='x', size=8),
                                       name=f'known_anom - {col}'),
                            row=4, col=1
                        )

                # Vertical boundary lines (match other plotly usage)
                shapes = []
                def _vline(ts, color):
                    if ts is None:
                        return
                    xstr = pd.to_datetime(ts)
                    shapes.append(dict(type='line', x0=xstr, x1=xstr, y0=0, y1=1, yref='paper',
                                       line=dict(color=color, width=1, dash='dash')))

                _vline(self.train_start, 'red')
                _vline(self.train_end, 'red')
                _vline(self.val_start, 'green')
                _vline(self.val_end, 'green')
                _vline(self.test_start, 'magenta')
                _vline(self.test_end, 'magenta')

                if shapes:
                    fig.update_layout(shapes=shapes)

                fig.update_layout(height=900, showlegend=True,
                                  title_text=f'{self.data_name}: Detected anomalies (test interval)')
                fig.show()
            else:
                raise ValueError('No a valid plotting backend')
            
        return self.test_anomaly_detection
        

   
    @abstractmethod
    def _initialize_models(self):
        pass

    @abstractmethod
    def fit(self):
        pass    

    @abstractmethod
    def get_val_scores(self, recompute=True, plot=False, verbose=False):
        pass

    @abstractmethod
    def get_test_scores(self, recompute=True, plot=False, verbose=False):
        pass
    
   

    

   