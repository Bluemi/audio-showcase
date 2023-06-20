# Readme

Here are some scripts that can help understand discrete fourier transformations and filter
for audio data.
Be careful. Usually, calculation errors only hurt mentally. Here, they can also hurt physically.

## Getting started

In order to start these scripts you probably have to use a terminal with python installed. An IDE like pycharm can be helpful.
Needed libraries are listed in `requirements.txt`  and can be installed with `pip install -r requirements.txt`.

If you want to try load some audio data, you can download some audio-file (\*.wav) and put it into the `audio/` directory (ignored by git).

## Scripts
- `filter_song.py`:
  This script loads a song file, applies a filter to it and plays the resulting audio.
  The audio file should be named `Song.wav` and should be located in the same directory as the script.
  Alternatively, the file can be specified as a command line argument.

  There are two functions defined in this file: `test_fft()` and `test_filter()`.
  To execute those uncomment one of them at the very end of the file.

- `test_fft.py`:
  This file just generates some random data and performs a fourier transformation on it. It is a good
  starting point to see the discrete fourier transformation in action.

- `instruments.py`:
  This script creates different instruments (sine wave, square wave, piano) and play the audio.
  It is a good starting point to understand the principals of overtones.
