"""
Performs lock-in amplification of a given dataset
"""
from scipy.signal.windows import hann
import numpy as np
import pandas as pd
pi, sin, sqrt, mean = np.pi, np.sin, np.sqrt, np.mean

class LIA:
    """
    Class for lock-in amplification of discrete-time measured data modulated sinusoidally.

    :param sampling_frequency: Frequency at which the data is sampled
    :param data: The data which you want to analyze. Assumed to be a 1xN array
    :param sync_points: 1xN array of points with a known phase of the modulation signal (i.e. zero crossings)
    """
    def __init__(self, sampling_frequency, data=None, sync_points=None):
        self.sampling_frequency = sampling_frequency
        self.data = data
        if isinstance(self.data, type(None)):
            self.data_length = 0
            self.sync_points = None
        elif isinstance(self.data, np.ndarray):
            self.data_length = len(data)
            self.sync_points = sync_points
        elif isinstance(self.data, pd.DataFrame):
            self.data_length = len(data)
            self.sync_points = np.nonzero(data['Sync'].values)[0]
        else:
            raise ValueError("data type {type(data)} not supported")

        if len(self.sync_points) == 0:
            raise ValueError("No Synchronization points detected. Please verify the signal generator is on and hooked up")

    def trimToSyncPoints(self):
        """
        Trims data to the first and last synchronization points we have
        """
        start_index = self.sync_points[0]
        end_index = min(self.sync_points[-1]+1, len(self.data)-1)
        self.data = self.data[start_index:end_index]
        self.data_length = len(self.data)

    def extractSignalFrequency(self):
        """
        Extracts the signal frequency from the set of synchronization points and the known sampling frequency
        """
        sample_periods = np.diff(self.sync_points)
        average_period = np.mean(sample_periods)
        average_sample_frequency = self.sampling_frequency / average_period
        return average_sample_frequency

    def Modulate(self, modulation_frequency, synchronization_phase=pi,
                 window='hann'):
        """
        :param modulation_frequency: The desired frequency at which to modulate the signal (this is the expected signal frequency)
        :param synchronization_phase: The phase of the synchronization points on a sin(x) signal, from 0-2pi

        Modulates the existing data with a sinusoid of known frequency.
        Returns data with the correct mean, but higher total signal power,
        by about a factor of 1.22. Recommended not to use directly except
        in boxcar mode for this reason.

        """
        times = np.arange(0, 1/self.sampling_frequency*self.data_length, 1/self.sampling_frequency)
        if window == 'hann':
            hann_window = hann(self.data_length)
            hann_mean = np.mean(hann_window)
            hann_normalized = hann_window / hann_mean
            window_data = hann_normalized
        elif window == 'box' or window == 'boxcar':
            window_data = 1
        else:
            raise ValueError('Window {window} not implemented. Choices ' +\
                             'are hann and boxcar')
        modulation_signal = sqrt(2)*sin(2*pi*modulation_frequency*times - synchronization_phase)
        if isinstance(self.data, np.ndarray):
            return window_data * self.data * modulation_signal
        else:
            return window_data * self.data.iloc[:,1] * modulation_signal

    def extractSignalAmplitude(self, modulation_frequency='auto', synchronization_phase=pi, mode='rms'):
        if modulation_frequency == 'auto':
            modulation_frequency = self.extractSignalFrequency()
        self.trimToSyncPoints()
        modulated_data = self.Modulate(
            modulation_frequency=modulation_frequency,
            synchronization_phase=synchronization_phase)
        average_signal = mean(modulated_data)
        if mode=='rms':
            return average_signal
        elif mode=='amplitude':
            return sqrt(2)*average_signal
        else:
            raise ValueError(f'Did not recognize mode of {mode}. Choose "rms" or "amplitude"')

