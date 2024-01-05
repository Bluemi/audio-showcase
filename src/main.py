import sys
from pathlib import Path

import numpy as np

from audio_effects import pitch, stretch_audio, DFT_pshift, change_pitch, reverb, delay, flanger, distort, tremolo
from fdn_reverb import fdn_reverb
from utils import load_mono_audio, play_audio, seconds_to_samples, plot, pad_to_length

DELAY_TIME = 0.001
SAMPLERATE = 44100
DELAY_SAMPLES = int(SAMPLERATE * DELAY_TIME)


def test_song():
    path_to_song = Path('audio/Song2.wav')
    if len(sys.argv) >= 2:
        path_to_song = sys.argv[1]

    samples = load_mono_audio(path_to_song, length=10)

    new_samples = reverb(samples, length=2.0, wet_gain=2.2, dry_gain=1.0)
    # new_samples = delay(samples, length=0.25, strength=[0.5, 0.2, 0.1, 0.05])
    # new_samples = flanger(samples, min_delay=0.0, max_delay=0.002, duration=4.0)
    # new_samples = distort(samples, clip_level=0.01)
    # new_samples = tremolo(samples, min_amp=0.2, max_amp=1, duration=0.15)
    # new_samples = fdn_reverb(samples, gain_wet=2.2, gain_dry=1.0)
    # new_samples = stretch_audio(samples, value=2.4)

    # plot
    if np.mean(np.abs(samples)) < np.mean(np.abs(new_samples)):
        plot_data = np.array([pad_to_length(new_samples, len(samples)), samples])
        legend = ['Effekt', 'Original']
    else:
        plot_data = np.array([samples, pad_to_length(new_samples, len(samples))])
        legend = ['Original', 'Effekt']
    plot(plot_data, legend=legend)

    # play_audio(samples)
    play_audio(new_samples, normalize=True)


if __name__ == '__main__':
    test_song()
