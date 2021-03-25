"""
Interestingly, this is off by a bit (about 0.6% when measuring 10 periods). This appears to converge to the correct answer when an increasing number of periods are included and an increasing number of points on the sinewave are sampled. If we want better accuracy than this we will need to perform some data interpolation i.e. (filtering)
"""
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from pandas.testing import assert_frame_equal
from matplotlib import pyplot as plt
from liapy import LIA, ureg
from sciparse import assert_equal_qt, assert_allclose_qt
import pytest

@pytest.fixture
def data1():
    data_length = 11
    sampling_frequency = 10*ureg.Hz
    signal_rms_amplitude = 1*ureg.V
    signal_frequency = 1*ureg.Hz
    samples_per_period = sampling_frequency / signal_frequency
    number_periods = 1
    phase_offset = np.pi
    number_sync_points = 2 # This is the rare pathalogical case

    sync_indices = [0, 10]
    sync_points = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    times = ureg.s * np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    sin_data = signal_rms_amplitude * np.sqrt(2) * np.sin(2 * np.pi * signal_frequency * times - np.pi)
    squared_mean = 0.9090909090909094 # ONLY VALID FOR 11 POINTS
    sin_norm = np.sqrt(2) / squared_mean * \
               np.sin(2*np.pi * signal_frequency * times - np.pi)
    data = pd.DataFrame({
            'Time (s)': times.to(ureg.s).magnitude,
            'Amplitud (V)': sin_data.to(ureg.V).magnitude,
            'Sync': sync_points
            })
    lia = LIA(data)
    return {
        'times': times,
        'sin_data': sin_data,
        'sin_norm': sin_norm.magnitude,
        'sync_indices': sync_indices,
        'data': data,
        'lia': lia,
        'sampling_frequency': sampling_frequency,
        'signal_frequency': signal_frequency,
        'data_length': data_length,
    }

@pytest.fixture
def data():
    data_length= 1000
    sampling_frequency = 9700.0*ureg.Hz
    signal_rms_amplitude = 36*ureg.mV
    signal_frequency = 105.4*ureg.Hz
    phase_delay = 0.34
    samples_per_period = sampling_frequency / signal_frequency

    number_periods = int(np.floor(data_length / (sampling_frequency / signal_frequency)))
    number_sync_points = number_periods + 1
    indices = np.arange(0, number_sync_points, 1)
    sync_indices = \
            ((1/2*sampling_frequency / signal_frequency * \
           (1 + 2*indices + phase_delay/np.pi)).astype(np.int)).magnitude

    times = np.arange(0, data_length, 1) * (1 / sampling_frequency)
    squared_mean = 0.999786189 # ONLY VALID FOR THIS DATA
    sin_data = signal_rms_amplitude * np.sqrt(2) * \
        np.sin(2*np.pi*signal_frequency* times - phase_delay)
    sin_norm = np.sqrt(2) / squared_mean * \
               np.sin(2*np.pi * signal_frequency * times - phase_delay)
    zero_column = np.zeros(len(sin_data), dtype=np.int)
    zero_column[sync_indices] = 1
    test_data = pd.DataFrame({
           'Time (s)': times.to(ureg.s).magnitude,
           'Voltage (V)': sin_data.to(ureg.V).magnitude,
           'Sync': zero_column})
    test_data_trimmed = pd.DataFrame({
            'Time (s)': times.to(ureg.s).magnitude[50:972] - times.to(ureg.s).magnitude[50],
           'Voltage (V)': sin_data.to(ureg.V).magnitude[50:972],
           'Sync': zero_column[50:972]})

    lia = LIA(test_data)
    return {
        'test_data': test_data,
        'test_data_trimmed': test_data_trimmed,
        'lia': lia,
        'sampling_frequency': sampling_frequency,
        'signal_frequency': signal_frequency,
        'data_length': data_length,
        'sin_norm': sin_norm.magnitude,
        'sync_phase': phase_delay,
        'sync_indices': sync_indices,
    }

def testLIASetup(data):
    assert_equal_qt(data['lia'].sampling_frequency, data['sampling_frequency'])
    desired_data = data['test_data_trimmed']
    actual_data = data['lia'].data
    assert_frame_equal(actual_data, desired_data)

def test_setup_no_sync(data):
    test_data = data['test_data']
    test_data['Sync'] = 0
    with pytest.raises(ValueError):
        lia = LIA(data=test_data)

def test_trim_to_sync_indices_simple(data1):
    desired_data = data1['data']
    actual_data = data1['lia'].trim_to_sync_indices(data1['data'], data1['sync_indices'])
    assert_frame_equal(actual_data, desired_data)

def test_trim_to_sync_indices_complex(data):
    actual_data = data['lia'].trim_to_sync_indices(data['test_data'], sync_indices=data['sync_indices'])
    desired_data = data['test_data_trimmed']
    assert_frame_equal(actual_data, desired_data)

def test_extract_signal_frequency_simple(data1):
    actual_frequency = data1['lia'].extract_signal_frequency(
            data1['data'], sync_indices=data1['sync_indices'])
    desired_frequency = data1['signal_frequency']
    assert_equal_qt(actual_frequency, desired_frequency)

def test_extract_signal_frequency(data):
    actual_frequency = data['lia'].extract_signal_frequency(data['test_data'], sync_indices=data['sync_indices'])
    desired_frequency = 105.32030401737242*ureg.Hz
    assert_equal_qt(actual_frequency, desired_frequency)

def test_modulate_simple(data1):
    data_desired = data1['data'].copy()
    data_desired.iloc[:,1] *= data1['sin_norm']
    data_actual = data1['lia'].modulate(data1['data'], data1['signal_frequency'], window='boxcar')
    assert_frame_equal(data_actual, data_desired)

def test_modulate_complex(data):
    data_desired = data['test_data'].copy()
    data_desired.iloc[:,1] *= data['sin_norm']
    data_actual = data['lia'].modulate(data['test_data'],
            data['signal_frequency'], window='boxcar',
            synchronization_phase=data['sync_phase'])
    assert_frame_equal(data_actual, data_desired)

def test_extract_amplitude_simple(data1):
    actual_amplitude = data1['lia'].extract_signal_amplitude()
    desired_amplitude = 1*ureg.V
    assert_allclose_qt(actual_amplitude, desired_amplitude)

def test_extract_amplitude(data):
    actual_amplitude = \
            data['lia'].extract_signal_amplitude(
                    modulation_frequency=105.4*ureg.Hz)
    desired_amplitude = 0.035916721832169034*ureg.V
    assert_equal_qt(actual_amplitude, desired_amplitude)
