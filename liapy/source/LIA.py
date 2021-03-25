"""
Performs lock-in amplification of a given dataset
"""
from scipy.signal.windows import hann
from sciparse import column_from_unit, sampling_period, title_to_quantity
import numpy as np
import pandas as pd
from liapy import ureg
import pint
pi, sin, sqrt, mean = np.pi, np.sin, np.sqrt, np.mean

class LIA:
    """
    Class for lock-in amplification of discrete-time measured data modulated sinusoidally.

    :param data: The data which you want to analyze. Assumed to be a pandas DataFrame or a numpy array.
    :param sampling_frequency: Frequency at which the data is sampled. Only used if data type is numpy array.
    :param sync_indices: 1xN array of points with a known phase of the modulation signal (i.e. zero crossings)
    """
    def __init__(self, data, sampling_frequency=None, sync_indices=None):
        if isinstance(data, pd.DataFrame):
            sampling_frequency = \
                    (1 / sampling_period(data)).to(ureg.Hz)
            sync_indices = np.nonzero(data['Sync'].values)[0]
        elif isinstance(data, np.ndarray):
            sync_indices = sync_indices
        else:
            raise ValueError("data type {type(data)} not supported")

        if sync_indices is None:
            raise ValueError("No Synchronization points detected. Please verify the signal generator is on and hooked up")
        else:
            if len(sync_indices) == 0:
                raise ValueError("No Synchronization points detected. Please verify the signal generator is on and hooked up")

        self.sync_indices = sync_indices
        self.sampling_frequency = sampling_frequency
        self.data = self.trim_to_sync_indices(data, sync_indices)

    def trim_to_sync_indices(self, data, sync_indices):
        """
        Trims data so that we only have an integer number of periods
        in the data we are analyzing to minimize spectral leakage

        :param data: Data to trim. Must contain a "Sync" column.
        """
        start_index = sync_indices[0]
        end_index = min(sync_indices[-1]+1, len(data))
        trimmed_data = data[start_index:end_index].copy()

        # Redefine t=0
        if isinstance(trimmed_data, pd.DataFrame):
            zero_offset = trimmed_data.iloc[0,0]
            trimmed_data.iloc[:,0] -= zero_offset
            trimmed_data = trimmed_data.reset_index(drop=True)
        return trimmed_data

    def extract_signal_frequency(self, data, sync_indices):
        """
        Extracts the signal frequency from the set of synchronization points and the known sampling frequency.
        """
        points_per_period = np.diff(sync_indices)
        average_points_per_period = np.mean(points_per_period)
        average_sample_frequency = self.sampling_frequency / average_points_per_period
        return average_sample_frequency

    def modulate(
            self, data, modulation_frequency,
            synchronization_phase=pi,window='hann'):
        """
        Modulates data with a sinusoid of known frequency.
        Returns data with the correct mean, but higher total signal power,
        by about a factor of 1.22. Recommended not to use directly except
        in boxcar mode for this reason.

        :param data: The data which you want modulated
        :param modulation_frequency: The desired frequency at which to modulate the signal (this is the expected signal frequency)
        :param synchronization_phase: The phase of the synchronization points on a sin(x) signal, from 0-2pi
        """
        if window == 'hann':
            hann_window = hann(len(data))
            hann_mean = np.mean(hann_window)
            hann_normalized = hann_window / hann_mean
            window_data = hann_normalized
        elif window == 'box' or window == 'boxcar':
            window_data = 1
        else:
            raise ValueError('Window {window} not implemented. Choices ' +\
                             'are hann and boxcar')

        if isinstance(data, pd.DataFrame):
            times = column_from_unit(data, ureg.s)
            modulation_signal = sqrt(2)*sin(2*pi*modulation_frequency*times - synchronization_phase)

            # This compensates for the offset of our sample points compared to the maxima of the sinewave - they have less power than they *should* as continuous-time signals
            squared_mean = np.mean(np.square(modulation_signal))
            modulation_signal /= squared_mean
            new_data = data.copy()
            if isinstance(modulation_signal, pint.Quantity):
                modulation_signal = modulation_signal.magnitude
            new_data.iloc[:,1] *= modulation_signal
            return new_data
        else:
            times = np.arange(0, len(data), 1) * 1 / self.sampling_frequency
            modulation_signal = sqrt(2)*sin(2*pi*modulation_frequency*times - synchronization_phase)
            return window_data * data * modulation_signal

    def extract_signal_amplitude(
            self, data=None, modulation_frequency=None,
            sync_indices=None,
            synchronization_phase=pi, mode='rms'):
        """
        Main method used to extract the amplitude of a dataset

        :param data: Input data, defaults to data you loaded in when defining the object
        """
        if sync_indices is None:
            sync_indices = self.sync_indices
        if data is None:
            data = self.data
        else:
            new_data = data.copy()
            sync_indices = np.nonzero(new_data['Sync'].values)[0]
            trimmed_data = self.trim_to_sync_indices(
                    new_data, sync_indices)
            data = trimmed_data
        if modulation_frequency is None:
            modulation_frequency = \
                self.extract_signal_frequency(data, sync_indices)

        modulated_data = self.modulate(data=data,
            modulation_frequency=modulation_frequency,
            synchronization_phase=synchronization_phase)
#import matplotlib.pyplot as plt
#plt.plot(data.iloc[:,1])
#plt.plot(modulated_data.iloc[:,1])
#plt.show()
        average_signal = modulated_data.mean()
        if isinstance(data, pd.DataFrame):
            if len(average_signal) == 3:
                quantity = title_to_quantity(data.columns.values[1])
                average_signal = quantity * average_signal[1]
            else:
                raise ValueError('Actual dataFrame has more than 3 columns. Unsure which column to use for the signal extraction')
        if mode=='rms':
            return average_signal
        elif mode=='amplitude':
            return sqrt(2)*average_signal
        else:
            raise ValueError(f'Did not recognize mode of {mode}. Choose "rms" or "amplitude"')

