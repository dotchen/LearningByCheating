from collections import OrderedDict

from loguru import logger
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torchvision.utils as tv_utils


def _preprocess_image(x):
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            x = x.detach()

        if x.dim() == 3:
            if x.shape[0] == 3:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)

        # x = torch.nn.functional.interpolate(x, 128, mode='nearest')
        x = tv_utils.make_grid(x, padding=16, normalize=True, nrow=4)
        x = x.cpu().numpy()

    return x


def _format(**kwargs):
    result = list()

    for k, v in kwargs.items():
        if isinstance(v, float):
            result.append('%s: %.2f' % (k, v))
        else:
            result.append('%s: %s' % (k, v))

    return '\t'.join(result)


class Wrapper(object):
    def __init__(self, log):
        self.epoch = 0
        self._log = log
        self._writer = None
        self.scalars = OrderedDict()

        self.info = lambda **kwargs: self._log.info(_format(**kwargs))
        self.debug = self._log.debug

    def init(self, log_path):
        for i in self._log._handlers:
            self._log.remove(i)

        self._writer = SummaryWriter(log_path)
        self._log.add(
                '%s/log.txt' % log_path,
                format='{time:MM/DD/YY HH:mm:ss} {level}\t{message}')

    def scalar(self, **kwargs):
        for k, v in sorted(kwargs.items()):
            if k not in self.scalars:
                self.scalars[k] = list()

            self.scalars[k].append(v)

    def image(self, **kwargs):
        for k, v in sorted(kwargs.items()):
            self._writer.add_image(k, _preprocess_image(v), self.epoch)

    def end_epoch(self):
        for k, v in self.scalars.items():
            info = OrderedDict()
            info[k] = np.mean(v)
            info['std'] = float(np.std(v, dtype=np.float32))
            info['min'] = np.min(v)
            info['max'] = np.max(v)
            info['n'] = len(v)

            self.info(**info)
            self._writer.add_scalar(k, np.mean(v), self.epoch)

        self.epoch = self.epoch + 1
        self.scalars = OrderedDict()

        self.info(epoch=self.epoch)


log = Wrapper(logger)
