"""
Interestingly, this is off by a bit (about 0.6% when measuring 10 periods). This appears to converge to the correct answer when an increasing number of periods are included and an increasing number of points on the sinewave are sampled. If we want better accuracy than this we will need to perform some data interpolation i.e. (filtering)
"""
# TODO: There seems to be a bug in modulate() with maximum-frequency signals, causing absurdly-high values.
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from pandas.testing import assert_frame_equal
from matplotlib import pyplot as plt
from liapy import LIA, ureg
from sciparse import assert_allclose_qt
import pytest
import os

@pytest.fixture
def file_path():
    file_location = os.path.realpath(__file__)
    directory_path = str(os.path.dirname(file_location))
    return directory_path

@pytest.fixture
def data1():
    data_length = 11
    sampling_frequency = 10*ureg.Hz
    signal_rms_amplitude = 1*ureg.V
    signal_frequency = 1*ureg.Hz
    samples_per_period = sampling_frequency / signal_frequency
    number_periods = 1
    phase_delay = np.pi/2
    mod_phase_delay =  -np.pi/2
    number_sync_points = 2 # This is the rare pathalogical case

    sync_indices = [0, 10]
    sync_points = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    times = ureg.s * np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    sin_data = signal_rms_amplitude * np.sqrt(2) * np.sin(2 * np.pi * signal_frequency * times - phase_delay)
    sin_norm = np.sqrt(2) * \
               np.sin(2*np.pi * signal_frequency * times - phase_delay)
    sin_norm -= np.mean(sin_norm)
    squared_mean = np.mean(np.square(sin_norm))
    sin_norm /= squared_mean
    data = pd.DataFrame({
            'Time (s)': times.to(ureg.s).magnitude,
            'Amplitude (V)': sin_data.to(ureg.V).magnitude,
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
        'sync_phase': phase_delay,
        'sync_phase_delay': mod_phase_delay,
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
    phases = 2*np.pi * signal_frequency * times
    delayed_phases = phases - phase_delay
    sin_data = signal_rms_amplitude * np.sqrt(2) * np.sin(delayed_phases)

    sin_norm = np.sqrt(2) / squared_mean * np.sin(delayed_phases)
    sin_norm -= np.mean(sin_norm)
    zero_column = np.zeros(len(sin_data), dtype=np.int)
    zero_column[sync_indices] = 1
    test_data = pd.DataFrame({
           'Time (s)': times.to(ureg.s).magnitude,
           'Voltage (V)': sin_data.to(ureg.V).magnitude,
           'Sync': zero_column})

    lia = LIA(test_data)
    return {
        'test_data': test_data,
        'lia': lia,
        'sampling_frequency': sampling_frequency,
        'signal_frequency': signal_frequency,
        'data_length': data_length,
        'sin_norm': sin_norm.magnitude,
        'sync_phase': np.pi,
        'sync_indices': sync_indices,
    }

def testLIASetup(data):
    assert_allclose_qt(data['lia'].sampling_frequency, data['sampling_frequency'], atol=1e-6)
    desired_data = data['test_data']
    actual_data = data['lia'].data
    assert_frame_equal(actual_data, desired_data, atol=1e-6)

def test_setup_no_sync(data):
    test_data = data['test_data']
    test_data['Sync'] = 0
    with pytest.raises(ValueError):
        lia = LIA(data=test_data)

def test_extract_signal_frequency_simple(data1):
    actual_frequency = data1['lia'].extract_signal_frequency(
            data1['data'], sync_indices=data1['sync_indices'])
    desired_frequency = data1['signal_frequency']
    assert_allclose_qt(actual_frequency, desired_frequency, atol=1e-6)

def test_extract_signal_frequency(data):
    actual_frequency = data['lia'].extract_signal_frequency(data['test_data'], sync_indices=data['sync_indices'])
    desired_frequency = 105.32030401737242*ureg.Hz
    assert_allclose_qt(actual_frequency, desired_frequency, atol=1e-6)

def test_modulate_simple_zero_phase():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4.0, 5, 6],
            'Val (V)':  [0, 1, 0, -1.0, 0, 1, 0],
            'Sync':     [1, 0, 0, 0, 1, 0, 0]})
    lia = LIA(test_data)
    actual_data = lia.modulate(test_data, 250*ureg.Hz,
            sync_phase_delay=0, window='boxcar')

    desired_data = pd.DataFrame({
            'time (ms)': np.array([0, 1.0, 2, 3, 4, 5, 6]),
            'Val (V)':  np.sqrt(2)/0.857142857 * np.array([0, 0.9, 0, 1.2, 0, 0.9, 0]),
            'Sync':     np.array([1, 0, 0, 0, 1, 0, 0])})
    assert_frame_equal(actual_data, desired_data, atol=1e-6)

def test_modulate_max_freq():
    """
    The problem here is that our modulation signal is just a bunch of zeros. It appears either our synchronization phase or our data is wrong. For some reason our phases are not what they should be.
    """

    test_data = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4],
            'Voltage (mV)': [1, -1, 1, -1, 1],
            'Sync': [0, 1, 0, 1, 0],
            })
    lia = LIA(test_data)
    desired_data = pd.DataFrame({
            'Time (ms)': np.array([0, 1, 2, 3, 4]),
            'Voltage (mV)': np.array([0.589256, 0.883883, 0.589256, 0.883883, 0.589256]),
            'Sync':     np.array([0, 1, 0, 1, 0])})
    actual_data = lia.modulate(data=test_data,
            modulation_frequency=500*ureg.Hz,
            sync_phase_delay=3/2*np.pi)
    assert_frame_equal(actual_data, desired_data)


def test_modulate_simple_pi_2():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4.0, 5, 6],
            'Val (V)':  [0, 1, 0, -1.0, 0, 1, 0],
            'Sync':     [0, 1, 0, 0, 0, 1, 0]})
    lia = LIA(test_data)
    actual_data = lia.modulate(test_data, 250*ureg.Hz,
            sync_phase_delay=np.pi/2, window='boxcar')
    desired_data = pd.DataFrame({
            'time (ms)': np.array([0, 1.0, 2, 3, 4, 5, 6]),
            'Val (V)':  np.sqrt(2)/0.857142857 * np.array([0, 0.9, 0, 1.2, 0, 0.9, 0]),
            'Sync':     np.array([0, 1, 0, 0, 0, 1, 0])})
    assert_frame_equal(actual_data, desired_data)

def test_modulate_simple_pi():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4.0, 5, 6, 7],
            'Val (V)':  [0, 1, 0, -1.0, 0, 1, 0, -1],
            'Sync':     [0, 0, 1, 0, 0, 0, 1, 0]})
    lia = LIA(test_data)
    actual_data = lia.modulate(test_data, 250*ureg.Hz,
            sync_phase_delay=np.pi, window='boxcar')
    desired_data = pd.DataFrame({
            'time (ms)': np.array([0, 1.0, 2, 3, 4, 5, 6, 7]),
            'Val (V)':  np.sqrt(2) * np.array([0, 1.0, 0, 1, 0, 1, 0, 1]),
            'Sync':     np.array([0, 0, 1, 0, 0, 0, 1, 0])})
    assert_frame_equal(actual_data, desired_data)

def test_modulate_complex(data1):
    data_desired = data1['data'].copy()
    data_desired.iloc[:,1] *= data1['sin_norm']
    data_actual = data1['lia'].modulate(data1['data'], data1['signal_frequency'], sync_phase_delay=data1['sync_phase_delay'], window='boxcar')
    assert_allclose(np.mean(data_actual)[1], 1.0, atol=1e-10)
    assert_frame_equal(data_actual, data_desired)

@pytest.mark.skip
def test_modulate_super_complex(data):
    data_desired = data['test_data'].copy()
    data_desired.iloc[:,1] *= data['sin_norm']
    data_actual = data['lia'].modulate(data['test_data'],
            data['signal_frequency'], window='boxcar',
            sync_phase_delay=data['sync_phase'])
    assert_equal(np.mean(data_actual)[1], 0.0358904219619713)
    assert_frame_equal(data_actual, data_desired)

def test_extract_amplitude_complex(data1):
    actual_amplitude = data1['lia'].extract_signal_amplitude(sync_phase_delay=data1['sync_phase_delay'])
    desired_amplitude = 1*ureg.V
    assert_allclose_qt(actual_amplitude, desired_amplitude)

def test_extract_amplitude_super_complex(data):
    actual_amplitude = \
            data['lia'].extract_signal_amplitude(
                    modulation_frequency=105.4*ureg.Hz,
                    sync_phase_delay=data['sync_phase'])
    desired_amplitude = 0.035890070877237404*ureg.V
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_pi_2():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4, 5],
            'val (V)':  [0, 1, 0, -1, 0, 1],
            'Sync':     [0, 1, 0, 0, 0, 1]})
    lia = LIA(test_data)
    desired_amplitude = 1*ureg.V / np.sqrt(2)
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=np.pi/2)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_minus_pi_2():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4, 5, 6, 7],
            'val (V)':  [0, 1, 0, -1, 0, 1, 0, -1],
            'Sync':     [0, 0, 0, 1, 0, 0, 0, 1]})
    lia = LIA(test_data)
    desired_amplitude = 1*ureg.V / np.sqrt(2)
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=-np.pi/2)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_pi():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4, 5, 6, 7],
            'val (V)':  [0, 1, 0, -1, 0, 1, 0, -1],
            'Sync':     [0, 0, 1, 0, 0, 0, 1, 0]})
    lia = LIA(test_data)
    desired_amplitude = 1*ureg.V / np.sqrt(2)
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=np.pi)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_zero_with_offset():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4, 5, 6, 7],
            'val (V)':  [1, 1, 1, 1, 1, 1, 1, 1],
            'Sync':     [0, 0, 1, 0, 0, 0, 1, 0]})
    lia = LIA(test_data)
    desired_amplitude = 0*ureg.V
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=np.pi)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_zero_with_large_offset():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4, 5, 6, 7],
            'val (V)':  1e6 * np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            'Sync':     [0, 0, 1, 0, 0, 0, 1, 0]})
    lia = LIA(test_data)
    desired_amplitude = 0*ureg.V
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=np.pi)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_minus_pi():
    test_data = pd.DataFrame({
            'time (ms)': [0, 1, 2, 3, 4, 5, 6, 7],
            'val (V)':  [0, 1, 0, -1, 0, 1, 0, -1],
            'Sync':     [0, 0, 1, 0, 0, 0, 1, 0]})
    lia = LIA(test_data)
    desired_amplitude = 1*ureg.V / np.sqrt(2)
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=-np.pi)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_minus_pi_2_offset():
    test_data = pd.DataFrame({
            'time (ms)': [1, 2, 3, 4, 5, 6, 7],
            'val (V)':  [1, 0, -1, 0, 1, 0, -1],
            'Sync':     [1, 0, 0, 0, 1, 0, 0]})
    lia = LIA(test_data)
    desired_amplitude = 1*ureg.V / np.sqrt(2)
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=np.pi/2)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_minus_pi_2_offset():
    test_data = pd.DataFrame({
            'time (ms)': [1, 2, 3, 4, 5, 6, 7],
            'val (V)':  [1, 0, -1, 0, 1, 0, -1],
            'Sync':     [1, 0, 0, 0, 1, 0, 0]})
    lia = LIA(test_data)
    desired_amplitude = 1*ureg.V / np.sqrt(2)
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=np.pi/2)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_with_large_dc():
    test_data = pd.DataFrame({
            'time (ms)': [1, 2, 3, 4, 5, 6],
            'val (V)':  10000 + np.array([1, 0, -1, 0, 1, 0]),
            'Sync':     [1, 0, 0, 0, 1, 0]})
    lia = LIA(test_data)
    desired_amplitude = 1*ureg.V / np.sqrt(2)
    actual_amplitude = lia.extract_signal_amplitude(sync_phase_delay=np.pi/2)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_amplitude_real_data(file_path):
    test_data = pd.read_csv(file_path + '/data/photovoltage_data.csv', skiprows=1)
    lia = LIA(test_data)
    actual_amplitude = lia.extract_signal_amplitude()
    desired_amplitude = (-0.0185371754 -4.60284137e-11) *ureg.mV
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)

def test_extract_amplitude_max_freq():
    test_data = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4],
            'Voltage (mV)': [1, -1, 1, -1, 1],
            'Sync': [0, 1, 0, 1, 0],
            })
    lia = LIA(test_data)
    desired_amplitude = 1 * ureg.mV
    actual_amplitude = lia.extract_signal_amplitude(
            mode='amplitude', sync_phase_delay = 3/2*np.pi)
    assert_allclose_qt(actual_amplitude, desired_amplitude, atol=1e-6)
