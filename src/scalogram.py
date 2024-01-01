import sys
from pathlib import Path

import numpy as np

from utils import load_mono_audio, plot, show_image

WINDOW_SIZE = 256
STRIDE = WINDOW_SIZE // 2

def test_scalogram():
    # you probably have to change this path to some song that is located on your machine.
    path_to_song = Path('audio/Song.wav')
    if len(sys.argv) >= 2:
        # you can also define the audio with a command line argument
        path_to_song = sys.argv[1]

    # Load the song as mono audio (only load 5 seconds)
    samples = load_mono_audio(path_to_song, length=10)

    # Plot the sample data
    # plot(samples, title='Samples')

    sliding_window = np.lib.stride_tricks.sliding_window_view(samples, WINDOW_SIZE)[::STRIDE]

    # create scalogram
    scalogram = np.fft.fft(sliding_window, axis=1)
    scalogram = np.abs(scalogram)

    scalogram = scalogram[:, WINDOW_SIZE//2+1:]

    # normalize
    scalogram = scalogram / (np.max(scalogram) + 0.000001)
    scalogram = scalogram ** 0.45

    # scale for better visibility
    scalogram = np.repeat(scalogram, 6, axis=1)

    show_image(scalogram.T, vmin=0, vmax=1, cmap='inferno')

if __name__ == '__main__':
    test_scalogram()
