import numpy as np

from audio_effects import complement_half_spectrum
from utils import plot


NUM_SAMPLES = 32


def test_fft():
    # Create some random samples.
    # a = np.random.random((NUM_SAMPLES,))
    x = np.linspace(0, np.pi*2, NUM_SAMPLES)
    a = np.sin(x * 4) + np.cos(x * 3)

    # Plot the random data
    plot(a, title='Samples')

    # Convert to frequency domain. If you want you could try the discrete cosine transformation np.fft.dct().
    # This produces only real numbers (not complex)
    a_freq = np.fft.fft(a)

    # Plot the frequencies. Note that we only plot the real part of the complex spectrum.
    # As you can the spectrum is symmetrically except for the very first sample.
    plot(a_freq, title='Spektrum', legend=['real part', 'imaginary part'])

    # ------- Modify spectrum here -------
    half_spectrum = a_freq[:len(a_freq)//2 + 1]
    # half_spectrum[0] = 0
    full_spectrum = complement_half_spectrum(half_spectrum)
    # ------------------------------------

    # We convert frequencies back to samples (frequency domain -> time domain).
    a_reversed = np.fft.ifft(full_spectrum)

    # The real part of the reversed samples should look like original samples
    plot(a + a_reversed.real * 1j, title='Samples', legend=['Original', 'From Spectrum'])


def test_reverse_frequencies():
    # we create some artificial spectrum with only zeros and some numbers
    spectrum = np.zeros(1024)
    spectrum[1] = 1.0
    spectrum[10] = 1.0

    spectrum = np.linspace(0, 1, 21) ** 10
    plot(spectrum, title='Test Spectrum')

    # Convert spectrum back to time domain and see what the spectrum looks like in time domain...
    samples = np.fft.ifft(spectrum)

    # ... by plotting it.
    plot(samples, title='Samples (generiert von Spektrum)')


if __name__ == '__main__':
    # print('uncomment one of the two functions at the very end of the code to start testing.')
    test_fft()
    # test_reverse_frequencies()
