import sys
from pathlib import Path

import numpy as np

from utils import load_mono_audio, play_audio, set_area, plot


def test_fft():
    # you probably have to change this path to some song that is located on your machine.
    path_to_song = Path('../../audio/Song.wav')
    if len(sys.argv) >= 2:
        # you can also define the audio with a command line argument
        path_to_song = sys.argv[1]

    # Load the song as mono audio (only load 5 seconds)
    samples = load_mono_audio(path_to_song, length=5)

    # Plot the sample data
    plot(samples)

    # Play original song
    play_audio(samples)

    # Convert to frequency domain.
    # Note that frequencies with the fourier transformation are represented as complex numbers.
    spectrum = np.fft.fft(samples)

    plot(spectrum)

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
    plot(spectrum.real)

    # Convert frequencies back to sample data in time domain.
    new_samples = np.fft.ifft(spectrum)

    # Play the edited audio data.
    play_audio(new_samples)


def test_filter():
    # Get file path and load mono audio data
    path_to_song = Path('Song.wav')
    if len(sys.argv) >= 2:
        path_to_song = sys.argv[1]
    samples = load_mono_audio(path_to_song, length=5)  # load 5 seconds

    # Play original
    play_audio(samples)

    # Defines the number of samples in the filter. Changing the filter_width should change how the filter works.
    # An odd number of filter samples is recommended, so we have some central element
    filter_width = 21

    # You can choose between some different filters
    filter_type = 'simple_blur'

    if filter_type == 'simple_blur':
        # This creates an array containing <filter_width> samples each having the value 1.0 / <filter_width>
        audio_filter = np.ones(filter_width) / filter_width
    elif filter_type == 'gaussian_blur':
        # Creates a filter by sampling a gaussian curve between -1.0 and 1.0
        audio_filter = gaussian_filter(filter_width, sigma=0.3)
    elif filter_type == 'gradient':
        # filter normally used for edge detection. This filter looks like this: [-1, ..., -1, 0, 1, ..., 1]
        audio_filter = np.zeros(filter_width)
        audio_filter[:filter_width//2] = -1
        audio_filter[filter_width//2+1:] = 1
    elif filter_type == 'custom':
        audio_filter = np.array([-2, -5, -10, 34, -10, -5, -2])
    elif filter_type == 'DoG':
        # DoG stands for difference of gaussians. See https://www.desmos.com/calculator/za7pf8mx3k for more information.
        audio_filter = gaussian_filter(filter_width, sigma=0.25) - gaussian_filter(filter_width, sigma=0.3)
    else:
        raise ValueError('Unknown filter type: {}'.format(filter_type))

    # apply filter to audio
    convolved_samples = np.convolve(samples, audio_filter)

    # see how the processed samples sound
    play_audio(convolved_samples)


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
    test_fft()
    # test_filter()
