#!/usr/bin/env python3
from pathlib import Path
from collections.abc import Sized

import numpy as np
import matplotlib.pyplot as plt
import pydub
from pydub.playback import play
import audiosegment


DEFAULT_SAMPLE_RATE = 44100


def set_area(data, percentage, mode, value=0.0):
    if mode == 'start':
        end_index = int(len(data) * percentage)
        data[:end_index].real = value
        if np.any(np.iscomplex(data)):
            data[:end_index].imag = value
    elif mode == 'end':
        start_index = int(len(data) * (1.0-percentage))
        data[start_index:].real = value
        if np.any(np.iscomplex(data)):
            data[start_index:].imag = value
    elif mode == 'mid':
        start_index = int(len(data) * (1.0-percentage) * 0.5)
        end_index = len(data) - start_index
        data[start_index:end_index].real = value
        if np.any(np.iscomplex(data)):
            data[start_index:end_index].imag = value
    elif mode == 'border':
        start_index = int(len(data) * percentage * 0.5)
        data[:start_index].real = value
        if np.any(np.iscomplex(data)):
            data[:start_index].imag = value
        end_index = len(data) - start_index
        data[end_index:].real = value
        if np.any(np.iscomplex(data)):
            data[end_index:].imag = value
    elif mode == 'smallest':
        indices = np.argsort(data)
        end_index = int(len(data) * percentage)
        data[indices[:end_index]] = value
    elif mode == 'biggest':
        indices = np.argsort(data)
        end_index = int(len(data) * percentage)
        data[indices[-end_index:]] = value
    else:
        raise ValueError('unknown mode: {}'.format(mode))


def play_audio(song: np.ndarray, volume=1.0, sample_rate=DEFAULT_SAMPLE_RATE, normalize=True):
    song = song - np.mean(song)
    if normalize:
        song = song * (volume / np.max(np.abs(song)))
    song = song.reshape((-1, 1))
    song = (song * 2**15).astype(np.int16)
    song = audiosegment.from_numpy_array(song, sample_rate)
    song = song.fade_in(100).fade_out(100)
    play(song)


def load_mono_audio(path: str or Path, length=3) -> np.ndarray:
    song = pydub.AudioSegment.from_wav(path)
    song = song[:1000*length]
    song = song.split_to_mono()[0]
    data = np.array(song.get_array_of_samples())

    return data / np.max(data)


def plot(y, x=None, zoom=None, title=None, legend=None):
    legend_kw_args1 = {}
    legend_kw_args2 = {}
    if legend is not None:
        if isinstance(legend, str):
            legend_kw_args1['label'] = legend
        elif isinstance(legend, list) or isinstance(legend, tuple):
            legend_kw_args1['label'] = legend[0]
            if len(legend) >= 2:
                legend_kw_args2['label'] = legend[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if x is None:
        x = np.arange(len(y))

    # only plot subarea of wave
    if zoom:
        size = x.shape[0] // int(zoom)
        x = x[:size]
        y = y[:size]

    # handle complex numbers
    if y.dtype == complex:
        ax.plot(x, y.real, **legend_kw_args1)
        ax.plot(x, y.imag, **legend_kw_args2)
    else:
        # plot
        ax.plot(x, y, **legend_kw_args1)

    # title
    if title:
        ax.set_title(title, fontdict={'fontsize': 32})

    if legend:
        ax.legend()
        plt.legend()

    # grid and spines
    ax.spines.left.set_position('zero')
    ax.spines.right.set_color('none')
    ax.spines.bottom.set_position('zero')
    ax.spines.top.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True)
    plt.show()


def seconds_to_samples(milliseconds, sample_rate=44100):
    return int(sample_rate * milliseconds)


def pad_to_length(samples, length):
    diff = len(samples) - length
    if diff < 0:
        samples = np.concatenate([samples, np.zeros(-diff)])
    elif diff > 0:
        samples = samples[:length]
    return samples


def complement_half_spectrum(half_spectrum):
    real_part = half_spectrum.real
    imag_part = half_spectrum.imag
    second_part = real_part[len(real_part)-2:0:-1] - imag_part[len(imag_part)-2:0:-1] * 1j
    return np.concatenate([half_spectrum, second_part])
