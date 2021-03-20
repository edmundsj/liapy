.. _software_liapy:

Welcome to Lock-In Amplifier's documentation!
=============================================

Common Mistakes / Issues
--------------------------
When using the Modulate() function with hann windowing, the signal returned is
designed to have the same mean, but NOT the same signal power. The signal power
is higher by a factor of approximately :math:`2*\sqrt{0.375}`. This is because
the Hann window changes the mean and spectral power of the data differently,
and internally the LIA uses the *mean* of the data to compute the extracted
amplitude, not the DC component, as it is much more efficient.

.. autoclass:: LIA.LIA
    :members:
    :undoc-members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
