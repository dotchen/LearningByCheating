from pathlib import Path

import imageio


DEFAULT_DIR = str(Path.home().joinpath('debug'))
DEFAULT_PATH = 'test.gif'


class Dummy(object):
    images = dict()

    @classmethod
    def add(cls, key, image):
        """
        Add an image to image to image to image

        Args:
            cls: (todo): write your description
            key: (todo): write your description
            image: (array): write your description
        """
        if key not in cls.images:
            cls.images[key] = list()

        cls.images[key].append(image.copy())

    @classmethod
    def save(cls, key, save_dir=None, save_path=None, duration=0.1):
        """
        Save an image to disk.

        Args:
            cls: (todo): write your description
            key: (str): write your description
            save_dir: (str): write your description
            save_path: (str): write your description
            duration: (float): write your description
        """
        save_dir = Path(save_dir or DEFAULT_DIR).resolve()
        save_path = save_path or DEFAULT_PATH

        save_dir.mkdir(exist_ok=True, parents=True)

        imageio.mimsave(
                str(save_dir.joinpath(save_path)), cls.images[key],
                'GIF', duration=duration)

        cls.clear(key)

    @classmethod
    def clear(cls, key=None):
        """
        Clears the images in - memory.

        Args:
            cls: (todo): write your description
            key: (str): write your description
        """
        if key in cls.images:
            cls.images.pop(key)
        else:
            cls.images.clear()


add = Dummy.add
save = Dummy.save
clear = Dummy.clear
