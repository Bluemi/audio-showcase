import sys
from pathlib import Path

import numpy as np

from utils import load_mono_audio, play_audio, set_area, plot


DELAY_TIME = 0.001
DELAY_SAMPLES = int(44100 * DELAY_TIME)


def test_song():
    path_to_song = Path('../../audio/Song.wav')
    if len(sys.argv) >= 2:
        path_to_song = sys.argv[1]

    samples = load_mono_audio(path_to_song, length=5)
    play_audio(samples)


if __name__ == '__main__':
    test_song()
