#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pydub
from pydub.playback import play
import audiosegment


DEFAULT_SAMPLE_RATE = 44100


def clamp(a, mini, maxi):
    return np.maximum(np.minimum(a, maxi), mini)


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


def samples_to_u16(samples, volume=1.0, normalize=True):
    song = samples - np.mean(samples)
    if normalize:
        song = song * (volume / np.max(np.abs(song)))
    song = song.reshape((-1, 1))
    return clamp((song * 2**15), -2**15, 2**15-1).astype(np.int16)


def play_audio(song: np.ndarray, volume=1.0, sample_rate=DEFAULT_SAMPLE_RATE, normalize=True):
    song = samples_to_u16(song, volume=volume, normalize=normalize)
    song = audiosegment.from_numpy_array(song, sample_rate)
    song = song.fade_in(100).fade_out(100)
    play(song)


def load_mono_audio(path: str or Path, length=3) -> np.ndarray:
    song = pydub.AudioSegment.from_wav(path)
    if length > 0:
        song = song[:1000*length]
    song = song.split_to_mono()[0]
    data = np.array(song.get_array_of_samples())

    return data / np.max(data)


def plot(y, x=None, zoom=None, title=None, legend=None, plot_style='line'):
    if len(y.shape) == 1:
        y = y.reshape((1, -1))
    num_plots = len(y)

    # handle complex numbers
    if y.dtype == complex:
        assert num_plots == 1
        y = np.array([y[0].real, y[0].imag])
        num_plots = 2

    legend_kw_args = []
    if legend is None:
        legend_kw_args = [{}] * num_plots
    else:
        if isinstance(legend, str):
            if num_plots != 1:
                raise ValueError(f'got only one legend for {num_plots} plots.')
            legend_kw_args.append({'label': legend})
        elif isinstance(legend, tuple) or isinstance(legend, list):
            if len(legend) != num_plots:
                raise ValueError(f'got only {len(legend)} legend for {num_plots} plots.')
            for l in legend:
                legend_kw_args.append({'label': l})
        else:
            raise TypeError('unknown type for legend')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if x is None:
        x = np.arange(len(y[0]))

    # only plot subarea of wave
    if zoom:
        size = x.shape[0] // int(zoom)
        x = x[:size]
        y = y[:, :size]

    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i, d in enumerate(zip(y, legend_kw_args)):
        data_line, legend_kw = d
        if plot_style == 'line':
            ax.plot(x, data_line, **legend_kw)
        elif plot_style == 'stem':
            ax.stem(x, data_line, linefmt=colors[i % len(colors)], markerfmt='.', **legend_kw)

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


def seconds_to_samples(seconds, sample_rate=44100):
    return int(sample_rate * seconds)


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


def show_image(image, **kwargs):
    plt.imshow(image, **kwargs)
    plt.show()
