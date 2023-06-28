import numpy as np
import scipy.signal

from utils import pad_to_length, complement_half_spectrum, load_mono_audio, seconds_to_samples, plot

REVERB_IMPULSE = None


def reverb(samples, length=1, strength=1):
    global REVERB_IMPULSE
    if REVERB_IMPULSE is None:
        REVERB_IMPULSE = load_mono_audio('audio/BalloonPop.wav')

    reverb_impulse = stretch_audio(REVERB_IMPULSE, length)
    new_samples = np.convolve(samples, reverb_impulse * strength)
    new_samples = new_samples / np.max(new_samples)  # normalize as the signal now should be much louder
    return new_samples


def delay(samples, length=1, strength=0.5):
    if isinstance(strength, float):
        strength = [strength]

    a_filter = np.zeros(int(length * 44100 * len(strength)))
    a_filter[0] = 1
    for i, s in enumerate(strength):
        index = min(int(44100 * (length * (i+1))), len(a_filter)-1)
        a_filter[index] = s

    new_samples = scipy.signal.fftconvolve(samples, a_filter)

    return new_samples



def flanger(samples, min_delay=0.0, max_delay=0.001, duration=1):
    delay_one_cycle = np.linspace(
        seconds_to_samples(min_delay),
        seconds_to_samples(max_delay),
        seconds_to_samples(duration/2)
    ).astype(int )
    delay_one_cycle = np.concatenate([delay_one_cycle, delay_one_cycle[::-1]])

    delay_samples = delay_one_cycle
    while len(delay_samples) < len(samples):
        delay_samples = np.concatenate([delay_samples, delay_one_cycle])

    delay_samples = delay_samples[:len(samples)]

    plot(delay_samples)

    indices = np.arange(len(samples)) + delay_samples
    indices = np.minimum(indices, len(samples)-1)

    return (samples + samples[indices]) / 2


def tremolo(samples, min_amp=0.8, max_amp=1.0, duration=0.2):
    amp_one_cycle = np.linspace(
        min_amp,
        max_amp,
        seconds_to_samples(duration/2)
    )
    amp_one_cycle = np.concatenate([amp_one_cycle, amp_one_cycle[::-1]])

    amp_samples = amp_one_cycle
    while len(amp_samples) < len(samples):
        amp_samples = np.concatenate([amp_samples, amp_one_cycle])

    amp_samples = amp_samples[:len(samples)]

    return samples * amp_samples


def distort(samples, clip_level=0.9):
    samples = np.minimum(samples, clip_level)
    samples = np.maximum(samples, -clip_level)
    return samples


def stretch_audio(samples, value):
    num_new_indices = int(len(samples) * value)
    indices = np.arange(num_new_indices)
    new_indices = np.round(indices / value).astype(int)
    new_indices = new_indices[new_indices < len(samples)]
    return samples[new_indices]


def change_pitch(samples, value):
    spectrum = np.fft.fft(samples)
    half_spectrum = spectrum[:len(spectrum)//2+1]
    new_spectrum = stretch_audio(half_spectrum, value)
    new_spectrum = pad_to_length(new_spectrum, len(spectrum)//2 + 1)
    new_spectrum = complement_half_spectrum(new_spectrum)
    return np.fft.ifft(new_spectrum)


def DFT_rescale(x, f):
    """
    Utility function that pitch shift a short segment `x`.
    """
    X = np.fft.fft(x)
    # separate even and odd lengths
    parity = (len(X) % 2 == 0)
    N = len(X) / 2 + 1 if parity else (len(X) + 1) / 2
    N = int(N)
    Y = np.zeros(N, dtype=complex)
    # work only in the first half of the DFT vector since input is real
    for n in range(0, N):
        # accumulate original frequency bins into rescaled bins
        ix = int(n * f)
        if ix < N:
            Y[ix] += X[n]
    # now rebuild a Hermitian-symmetric DFT
    Y = np.r_[Y, np.conj(Y[-2:0:-1])] if parity else np.r_[Y, np.conj(Y[-1:0:-1])]
    return np.real(np.fft.ifft(Y))


def win_taper(N, overlap):
    R = int(N * overlap / 2)
    r = np.arange(0, R) / float(R)
    win = np.r_[r, np.ones(N - 2*R), r[::-1]]
    stride = N - R - 1
    return win, stride


def DFT_pshift(x, f, G, overlap=0):
    """
    Function for pitch shifting an input signal by applying the above utility function on overlapping segments.
    """
    N = len(x)
    y = np.zeros(N)
    win, stride = win_taper(G, overlap)
    for n in np.arange(0, len(x) - G, stride):
        w = DFT_rescale(x[n:n+G] * win, f)
        y[n:n+G] += w * win
    return y


def pitch(samples, value):
    # TODO: does not work
    spectrum = np.fft.fft(samples)
    new_spectrum = np.concatenate([np.zeros(int(value*1000)), spectrum])
    new_samples = np.fft.ifft(new_spectrum)
    return new_samples
