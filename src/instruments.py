import numpy as np

from utils import plot, play_audio

SAMPLERATE = 44100
SECONDS = 3
FREQUENCY = 440


def test_instruments():
    # Create some data to convert to convert to samples
    x = np.linspace(0, SECONDS, SAMPLERATE*SECONDS)

    # Use sine function to create a sine wave with the given frequency.
    # You can use different types of waves here.
    wave_type = 'sawtooth'
    if wave_type == 'sin':
        y = np.sin(x * 2 * np.pi * FREQUENCY)
    elif wave_type == 'sawtooth':
        y = sawtooth(x * 2 * np.pi * FREQUENCY)
    elif wave_type == 'square':
        y = square(x * 2 * np.pi * FREQUENCY)
    elif wave_type == 'instrument':
        # You can try different instruments here
        y = piano(x * 2 * np.pi * FREQUENCY)
    else:
        raise ValueError('Unknown wave_type: {}'.format(wave_type))

    # Plot the generated wave. Only plot two periods of it.
    plot(y, x=x, zoom=440*SECONDS//2)

    # play the generated sound.
    play_audio(y, volume=0.3)


def sawtooth(x):
    result = np.zeros_like(x)
    # use 1000 iterations of overtones
    for i in range(1, 1000):
        result += 1/i * np.sin(i*x)
    return result


def square(x):
    result = np.zeros_like(x)
    # use 1000 iterations of overtones (only use every second overtone, see definition of square wave)
    for i in range(1, 1000, 2):
        result += 1/i * np.sin(i*x)
    return result


def piano(x):
    result = np.zeros_like(x)
    coefficients = [1, 0.3, 0.6, 0.57, 0.4]
    for i, c in enumerate(coefficients):
        result += c * np.sin((i+1)*x)
    return result


def piano2(x):
    result = np.zeros_like(x)
    coefficients = [1, 0.9, 0.9, 0.9, 0.5, 0.4, 0.56, 0.57, 0.5, 0.3, 0.56, 0.5]
    for i, c in enumerate(coefficients):
        result += c*(1/(i+1)) * np.sin((i+1)*x)
    return result


def horn(x):
    result = np.zeros_like(x)
    coefficients = [1, 0.9, 0.2, 0.17]
    for i, c in enumerate(coefficients):
        result += c * np.sin((i+1)*x)
    return result


def trumpet(x):
    result = np.zeros_like(x)
    coefficients = [1, 0.1, 1.0, 0.1, 1.0]
    for i, c in enumerate(coefficients):
        result += c * np.sin((i+1)*x)
    return result


# taken from http://www.lehrklaenge.de/PHP/Akustik/Obertonspektrum1.php
def trumpet2(x):
    result = np.zeros_like(x)
    coefficients = [1, 0.5, 0.75, 0.78, 0.9, 0.96, 0.96, 0.75, 0.78]
    for i, c in enumerate(coefficients):
        result += c * np.sin((i+1)*x)
    return result


def trombone(x):
    result = np.zeros_like(x)
    coefficients = [1, 1, 1]
    for i, c in enumerate(coefficients):
        result += c * np.sin((i+1)*x)
    return result


if __name__ == '__main__':
    test_instruments()
