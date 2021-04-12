"""
Performs lock-in amplification of a given dataset.

THIS CODE RIGHT NOW IS SO FUCKED. I NEED TO FIGURE OUT HOW TO REFACTOR IT SO THAT IT ACTUALLY MAKES SENSE. RIGHT NOW IT'S WRONG, WRONG, WRONG. NO TRIMMING. THE TRIMMING JUST CONFUSES EVERYTHING. TRIMMING IS NOW GONE, BUT STUFF IS STILL FUCKED.
"""
from scipy.signal.windows import hann
from sciparse import column_from_unit, sampling_period, title_to_quantity
import numpy as np
import pandas as pd
from liapy import ureg
import pint
import matplotlib.pyplot as plt
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
            raise ValueError(f"data type {type(data)} not supported")

        if sync_indices is None:
            raise ValueError("No Synchronization points detected. Please verify the signal generator is on and hooked up")
        else:
            if len(sync_indices) == 0:
                raise ValueError("No Synchronization points detected. Please verify the signal generator is on and hooked up")

        self.sync_indices = sync_indices
        self.sampling_frequency = sampling_frequency
        self.data = data

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
            sync_phase_delay=pi,window='hann'):
        """
        Modulates data with a sinusoid of known frequency.
        Returns data with the correct mean, but higher total signal power,
        by about a factor of 1.22. Recommended not to use directly except
        in boxcar mode for this reason.

        :param data: The data which you want modulated
        :param modulation_frequency: The desired frequency at which to modulate the signal (this is the expected signal frequency)
        :param sync_phase_delay: The phase of the synchronization points on a sin(x) signal, from 0-2pi
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
            sync_indices = np.nonzero(data['Sync'].values)[0]
            times = column_from_unit(data, ureg.s)
            time_phase_delay = times[sync_indices[0]]* \
                modulation_frequency * 2 * np.pi
            total_phase_delay = time_phase_delay - sync_phase_delay
#if time_phase_delay > sync_phase_delay:
#total_phase_delay = time_phase_delay - sync_phase_delay
#else:
#total_phase_delay = sync_phase_delay - time_phase_delay

            modulation_signal = sqrt(2)*np.sin(2*pi*modulation_frequency*times - total_phase_delay)

            # This compensates for the offset of our sample points compared to the maxima of the sinewave - they have less power than they *should* as continuous-time signals
            squared_mean = np.mean(np.square(modulation_signal))
            modulation_signal /= squared_mean
            new_data = data.copy()
#plt.plot(modulation_signal)
#plt.plot(data.iloc[9:,1].values*30)
#plt.show()
            if isinstance(modulation_signal, pint.Quantity):
                modulation_signal = modulation_signal.magnitude
            new_data.iloc[:,1] *= modulation_signal
            return new_data
        else:
            # THIS DOES NOT CURRENTLY WORK. NEEDS TO BE FIXED.
             raise ValueError('numpy arrays not currently supported. Just mimic the pandas stuff.')

    def extract_signal_amplitude(
            self, data=None, modulation_frequency=None,
            sync_indices=None,
            sync_phase_delay=pi, mode='rms'):
        """
        Main method used to extract the amplitude of a dataset

        :param data: Input data, defaults to data you loaded in when defining the object
        """
        if sync_indices is None:
            sync_indices = self.sync_indices
        if data is None:
            data = self.data
        if modulation_frequency is None:
            modulation_frequency = \
                self.extract_signal_frequency(data, sync_indices)

        modulated_data = self.modulate(data=data,
            modulation_frequency=modulation_frequency,
            sync_phase_delay=sync_phase_delay)
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

