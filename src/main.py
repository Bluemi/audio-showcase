import sys
from pathlib import Path

from audio_effects import pitch, stretch_audio, DFT_pshift, change_pitch, reverb, delay, flanger, distort, tremolo
from utils import load_mono_audio, play_audio, seconds_to_samples, plot

DELAY_TIME = 0.001
SAMPLERATE = 44100
DELAY_SAMPLES = int(SAMPLERATE * DELAY_TIME)


def test_song():
    path_to_song = Path('audio/Song2.wav')
    if len(sys.argv) >= 2:
        path_to_song = sys.argv[1]

    samples = load_mono_audio(path_to_song, length=5)

    # new_samples = reverb(samples, length=1.0, strength=1.0)
    # new_samples = delay(samples, length=0.15, strength=[0.5, 0.2, 0.1, 0.05])
    # new_samples = flanger(samples, min_delay=0.0, max_delay=0.002, duration=5.0)
    # new_samples = distort(samples, clip_level=0.01)
    new_samples = tremolo(samples, min_amp=0.2, max_amp=1, duration=0.15)

    plot(samples + new_samples * 1j)

    # play_audio(samples)
    play_audio(new_samples, normalize=False)


if __name__ == '__main__':
    test_song()
