"""
Interestingly, this is off by a bit (about 0.6% when measuring 10 periods). This appears to converge to the correct answer when an increasing number of periods are included and an increasing number of points on the sinewave are sampled. If we want better accuracy than this we will need to perform some data interpolation i.e. (filtering)
"""
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from matplotlib import pyplot as plt
from liapy import LIA

@pytest.fixture
def numpy_data_complex():
    data_length= 1000
    sampling_frequency = 9700.0
    signal_rms_amplitude = 0.036
    signal_frequency = 105.4
    phase_delay = 0.34
    samples_per_period = sampling_frequency / signal_frequency

    number_periods = int(np.floor(data_length / (sampling_frequency / signal_frequency)))
    number_sync_indices = number_periods + 1
    indices = np.arange(0, number_sync_indices, 1)
    sync_indices = (1/2*sampling_frequency / signal_frequency * \
                       (1 + 2*indices + phase_delay/np.pi)).astype(np.int)

    times = np.arange(0, data_length*1/sampling_frequency, 1/sampling_frequency)
    test_data = signal_rms_amplitude * np.sqrt(2) * \
        np.sin(2*np.pi*signal_frequency* times - phase_delay)
    return {
        'sampling_frequency': sampling_frequency,
        'sync_indices':sync_indices,
        'test_data': test_data,}

@pytest.fixture
def lia_complex(numpy_data_complex):
    lia = LIA(sampling_frequency=numpy_data_complex['sampling_frequency'],
            data=numpy_data_complex['test_data'],
            sync_indices=numpy_data_complex['sync_indices'])
    return lia

def test_np_lia_setup(numpy_data_complex, lia_complex):
    assert_equal(lia_complex.sampling_frequency,
            numpy_data_complex['sampling_frequency'])
    assert_equal(numpy_data_complex['test_data'], lia_complex.data)
    assert_equal(1000, len(lia_complex.data))

def test_np_extract_frequency_complex(numpy_data_complex, lia_complex):
    actual_frequency = lia_complex.extract_signal_frequency(
            numpy_data_complex['test_data'],
            numpy_data_complex['sync_indices'])
    desired_frequency = 105.32030401737242
    assert_equal(actual_frequency, desired_frequency)

@pytest.mark.skip
def test_np_extract_amplitude_complex(lia_complex):
    signal_rms_amplitude = lia_complex.extract_signal_amplitude(sync_phase_delay=np.pi)
    assert_equal(isinstance(signal_rms_amplitude, float), True)
    breakpoint()
    assert_equal(signal_rms_amplitude, 0.035963590532051275) # We can't get exactly the right amplitude, but we can get pretty close.
