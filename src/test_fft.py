import numpy as np

from utils import plot


NUM_SAMPLES = 16


def test_fft():
    # Create some random samples.
    a = np.random.random((NUM_SAMPLES,))

    # Plot the random data
    plot(a)

    # Convert to frequency domain. If you want you could try the discrete cosine transformation np.fft.dct().
    # This produces only real numbers (not complex)
    a_freq = np.fft.fft(a)

    # Plot the frequencies. Note that we only plot the real part of the complex spectrum.
    # As you can the spectrum is symmetrically except for the very first sample.
    plot(a_freq.real)

    # We convert frequencies back to samples (frequency domain -> time domain).
    a_reversed = np.fft.ifft(a_freq)

    # The real part of the reversed samples should look like original samples
    plot(a_reversed.real)


def test_reverse_frequencies():
    # we create some artificial spectrum with only zeros and some numbers
    spectrum = np.zeros(1024)
    spectrum[1] = 1.0
    spectrum[10] = 1.0

    # Convert spectrum back to time domain and see what the spectrum looks like in time domain...
    samples = np.fft.ifft(spectrum)

    # ... by plotting it.
    plot(samples.real)


if __name__ == '__main__':
    print('uncomment one of the two functions at the very end of the code to start testing.')
    # test_fft()
    # test_reverse_frequencies()
