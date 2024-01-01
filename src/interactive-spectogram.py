import sys
from pathlib import Path

import numpy as np
import pygame as pg

from utils import load_mono_audio, samples_to_u16, seconds_to_samples

WINDOW_SIZE = 256 * 8
STRIDE = WINDOW_SIZE // 2
FADE_MS = 10


class Main:
    def __init__(self, samples, spectrogram):
        self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        self.spectrogram = spectrogram
        image = np.repeat(self.spectrogram, 3, axis=-1).reshape(*self.spectrogram.shape, 3)
        image = pg.surfarray.make_surface((image * 255).astype(int))
        scale_shape = (image.get_width(), pg.display.get_window_size()[1])
        self.spec_image = pg.transform.scale(image, scale_shape)

        self.running = True
        self.playing = False
        self.current_position = 0
        self.clock = pg.time.Clock()

        self.samples = samples_to_u16(samples)
        self.sound = self.get_current_sound()

    def get_current_sound(self):
        return pg.mixer.Sound(self.samples[seconds_to_samples(self.current_position):])

    def run(self):
        last_update = 0
        while self.running:
            self.handle_events()
            self.tick(last_update)
            self.render()
            last_update = self.clock.tick(60)

    def handle_events(self):
        events = pg.event.get()
        for event in events:
            self.handle_event(event)

    def handle_event(self, event):
        if event.type == pg.QUIT:
            self.running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self.running = False
            if event.key == pg.K_SPACE:
                if self.playing:
                    self.stop_sound()
                else:
                    self.start_sound()
            if event.key == pg.K_0:
                self.stop_sound()
                self.current_position = 0
                self.sound = self.get_current_sound()
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.current_position = event.pos[0] * (STRIDE / 44100)
                self.sound = self.get_current_sound()

    def stop_sound(self):
        self.playing = False
        self.sound.fadeout(FADE_MS)
        self.sound = self.get_current_sound()

    def start_sound(self):
        self.playing = True
        self.sound.play(fade_ms=FADE_MS)

    def tick(self, last_update):
        if self.playing:
            self.current_position += last_update / 1000

    def render(self):
        self.screen.fill((0, 0, 0))

        self.screen.blit(self.spec_image, (0, 0))

        render_pos = int(self.current_position * (44100 / STRIDE))
        pg.draw.line(self.screen, (80, 120, 200), (render_pos, 0), (render_pos, self.screen.get_height()))
        pg.display.flip()


def calc_spectrogram(samples):
    sliding_window = np.lib.stride_tricks.sliding_window_view(samples, WINDOW_SIZE)[::STRIDE]

    # create scalogram
    scalogram = np.fft.fft(sliding_window, axis=1)
    scalogram = np.abs(scalogram)

    scalogram = scalogram[:, WINDOW_SIZE//2+1:]

    # normalize
    scalogram = scalogram / (np.max(scalogram) + 0.000001)
    return scalogram ** 0.45


def main():
    pg.mixer.pre_init(44100, channels=1)
    pg.init()
    pg.key.set_repeat(130, 25)

    # load song
    path_to_song = Path('audio/Song.wav')
    if len(sys.argv) >= 2:
        path_to_song = sys.argv[1]

    samples = load_mono_audio(path_to_song, length=0)
    spectrogram = calc_spectrogram(samples)

    main_instance = Main(samples, spectrogram)
    main_instance.run()

    pg.quit()


if __name__ == '__main__':
    main()

