import numpy as np

from utils import play_audio, plot

SAMPLERATE = 44100
LENGTH = 2
FREQUENCY = 440
RENDER_ZOOM = 440 / 5


def calc_freq(pitch):
    return 440 * 2 ** ((pitch - 69) / 12)


def main():
    x = np.linspace(0, LENGTH, SAMPLERATE * LENGTH, endpoint=True)
    samples = np.sin(x * FREQUENCY * 2 * np.pi)

    # samples = np.sign(samples)

    plot(samples, x, zoom=LENGTH * RENDER_ZOOM)

    play_audio(samples)


if __name__ == '__main__':
    main()
