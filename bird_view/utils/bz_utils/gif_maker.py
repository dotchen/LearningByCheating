from pathlib import Path

import imageio


DEFAULT_DIR = str(Path.home().joinpath('debug'))
DEFAULT_PATH = 'test.gif'


class Dummy(object):
    images = dict()

    @classmethod
    def add(cls, key, image):
        if key not in cls.images:
            cls.images[key] = list()

        cls.images[key].append(image.copy())

    @classmethod
    def save(cls, key, save_dir=None, save_path=None, duration=0.1):
        save_dir = Path(save_dir or DEFAULT_DIR).resolve()
        save_path = save_path or DEFAULT_PATH

        save_dir.mkdir(exist_ok=True, parents=True)

        imageio.mimsave(
                str(save_dir.joinpath(save_path)), cls.images[key],
                'GIF', duration=duration)

        cls.clear(key)

    @classmethod
    def clear(cls, key=None):
        if key in cls.images:
            cls.images.pop(key)
        else:
            cls.images.clear()


add = Dummy.add
save = Dummy.save
clear = Dummy.clear
