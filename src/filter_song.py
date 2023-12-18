import sys
from pathlib import Path

import numpy as np
import scipy

from utils import load_mono_audio, play_audio, set_area, plot


PLOT_SPECTRUM = True


def test_fft():
    # you probably have to change this path to some song that is located on your machine.
    path_to_song = Path('audio/Song.wav')
    if len(sys.argv) >= 2:
        # you can also define the audio with a command line argument
        path_to_song = sys.argv[1]

    # Load the song as mono audio (only load 5 seconds)
    samples = load_mono_audio(path_to_song, length=5)

    # Plot the sample data
    plot(samples, title='Samples')

    # Play original song
    # play_audio(samples)

    # Convert to frequency domain.
    # Note that frequencies with the fourier transformation are represented as complex numbers.
    spectrum = np.fft.fft(samples)
    # plot(spectrum, title='Spektrum (vollständig)')
    plot(np.abs(spectrum), title='Spektrum')

    # We set 90 percent of the frequencies to zero (the second argument in "set_area()").
    # You can try the different modes "start", "end", "mid", "border".
    # The fourier transformation creates a spectrum that is symmetrically.
    # Low frequencies are on the left side, in the middle are high frequencies and on the right side the mirrored low
    # frequencies again. If we want to suppress high frequencies we should set frequencies in the middle of the
    # spectrum to zero. To suppress low frequencies we should set the borders of the spectrum to zero.
    set_area(spectrum, 0.9, mode='mid')
    # set_area(spectrum, 0.04, mode='border')

    # Plot the spectrum (we only plot the real part of the complex spectrum).
    # You can also try to plot the imaginary part with "freq.imag".
    plot(np.abs(spectrum), title='Spektrum (nach Bearbeitung)')

    # Convert frequencies back to sample data in time domain.
    new_samples = np.fft.ifft(spectrum)

    # Play the edited audio data.
    # play_audio(new_samples.real)


def test_filter():
    # Get file path and load mono audio data
    path_to_song = Path('audio/Song.wav')
    if len(sys.argv) >= 2:
        path_to_song = sys.argv[1]
    samples = load_mono_audio(path_to_song, length=5)  # load 5 seconds

    # Play original
    # play_audio(samples)

    # Defines the number of samples in the filter. Changing the filter_width should change how the filter works.
    # An odd number of filter samples is recommended, so we have some central element
    filter_width = 121

    # You can choose between some different filters
    filter_type = 'gaussian'

    if filter_type == 'identity':
        audio_filter = np.zeros(filter_width)
        audio_filter[len(audio_filter)//2] = 1
    elif filter_type == 'box':
        # This creates an array containing <filter_width> samples each having the value 1.0 / <filter_width>
        audio_filter = np.ones(filter_width) / filter_width
    elif filter_type == 'gaussian':
        # Creates a filter by sampling a gaussian curve between -1.0 and 1.0
        audio_filter = gaussian_filter(filter_width, sigma=0.3)
    elif filter_type == 'gradient':
        # filter normally used for edge detection. This filter looks like this: [-1, ..., -1, 0, 1, ..., 1]
        audio_filter = np.zeros(filter_width)
        audio_filter[:filter_width//2] = -1 / filter_width
        audio_filter[filter_width//2+1:] = 1 / filter_width
    elif filter_type == 'DoG':
        # DoG stands for difference of gaussians. See https://www.desmos.com/calculator/za7pf8mx3k for more information.
        audio_filter = gaussian_filter(filter_width, sigma=0.25) - gaussian_filter(filter_width, sigma=0.3)
    elif filter_type == 'random':
        audio_filter = np.random.normal(size=filter_width)
    elif filter_type == 'custom':
        audio_filter = np.array([1/5, 1/5, 1/10, 0, -1/10, -1/5, -1/5])
    elif filter_type == 'spectrum':
        filter_spectrum = np.linspace(1, 0, filter_width) ** 2
        audio_filter = scipy.fft.idct(filter_spectrum)
    else:
        raise ValueError('Unknown filter type: {}'.format(filter_type))

    plot(audio_filter, title='Audio Filter')

    # apply filter to audio
    convolved_samples = np.convolve(samples, audio_filter, mode='same')

    # play the processed samples
    play_audio(convolved_samples)

    if PLOT_SPECTRUM:
        spectrum = np.fft.fft(samples)
        convolved_spectrum = np.fft.fft(convolved_samples)
        specs = np.array([
            np.abs(spectrum)[:len(spectrum)//2+1],
            np.abs(convolved_spectrum)[:len(convolved_spectrum)//2+1]
        ])
        plot(specs, zoom=4, title='Song Spektrum', legend=['Original', 'Gefiltert'])

    # plot spectrum of filter
    filter_spectrum = np.fft.fft(audio_filter)

    # only use the first half of the spectrum as it is mirrored
    filter_spectrum = np.abs(filter_spectrum[:len(filter_spectrum)//2+1])
    plot(np.abs(filter_spectrum), title='Filter Spektrum')


def gauss_curve(x, sigma=1.0, mean=0.0):
    # See https://de.wikipedia.org/wiki/Normalverteilung for more information
    return 1.0 / np.sqrt(2.0 * np.pi * sigma) * np.exp(-np.square(x - mean)/(2*sigma*sigma))


def gaussian_filter(filter_width, sigma=1.0, mean=0.0):
    # sample gaussian curve from -1.0 to 1.0 with filter_width points
    gauss_filter = np.linspace(-1, 1, filter_width)
    # calculate gaussian curve
    gauss_filter = gauss_curve(gauss_filter, sigma, mean)
    # normalize by dividing the sum
    return gauss_filter / np.sum(gauss_filter)


if __name__ == '__main__':
    # print('uncomment one of the two functions at the very end of the code to start testing.')
    # test_fft()
    test_filter()
