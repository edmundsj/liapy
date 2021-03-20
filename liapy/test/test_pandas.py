"""
Interestingly, this is off by a bit (about 0.6% when measuring 10 periods). This appears to converge to the correct answer when an increasing number of periods are included and an increasing number of points on the sinewave are sampled. If we want better accuracy than this we will need to perform some data interpolation i.e. (filtering)
"""
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
from pandas.testing import assert_frame_equal
from matplotlib import pyplot as plt
from liapy import LIA
import pytest

class TestLIAPandas(unittest.TestCase):
    def setUp(self):
        self.data_length= 1000
        self.sampling_frequency = 9700.0
        self.signal_rms_amplitude = 0.036
        self.signal_frequency = 105.4
        self.phase_delay = 0.34
        samples_per_period = self.sampling_frequency / self.signal_frequency

        number_periods = int(np.floor(self.data_length / (self.sampling_frequency / self.signal_frequency)))
        number_sync_points = number_periods + 1
        indices = np.arange(0, number_sync_points, 1)
        sync_indices = (1/2*self.sampling_frequency / self.signal_frequency * \
                           (1 + 2*indices + self.phase_delay/np.pi)).astype(np.int)

        times = np.arange(0, self.data_length*1/self.sampling_frequency, 1/self.sampling_frequency)
        sin_data = self.signal_rms_amplitude * np.sqrt(2) * \
            np.sin(2*np.pi*self.signal_frequency* times - self.phase_delay)
        zero_column = np.zeros(len(sin_data))
        zero_column[sync_indices] = 1
        self.test_data = pd.DataFrame({'Time (s)': times,
                                       'Voltage (V)': sin_data,
                                       'Sync': zero_column})
        self.lia = LIA(sampling_frequency=self.sampling_frequency,
                data=self.test_data)

    def testLIASetup(self):
        assert_equal(self.lia.sampling_frequency, self.sampling_frequency)
        assert_frame_equal(self.test_data, self.lia.data)
        assert_equal(self.data_length, self.lia.data_length)

    def test_setup_no_sync(self):
        test_data = self.test_data
        test_data['Sync'] = 0
        with pytest.raises(ValueError):
            lia = LIA(sampling_frequency=self.sampling_frequency,
                    data=test_data)

    def testLIaExtractSignalFrequency(self):
        assert_equal(self.lia.extractSignalFrequency(), 105.32030401737242)

    def testLIAExtractAmplitude(self):
        signal_rms_amplitude = self.lia.extractSignalAmplitude(self.signal_frequency)
        #plt.plot(self.test_data[self.sync_points[0]:self.sync_points[-1]])
        assert_equal(isinstance(signal_rms_amplitude, float), True)
        assert_equal(signal_rms_amplitude, 0.035917003718687314) # We can't get exactly the right amplitude, but we can get pretty close.


