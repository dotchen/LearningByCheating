import numpy as np
import torch
import torchvision.transforms as transforms

import carla


class Agent(object):
    def __init__(self, model=None, **kwargs):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            model: (todo): write your description
        """
        assert model is not None

        if len(kwargs) > 0:
            print('Unused kwargs: %s' % kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.ToTensor()

        self.one_hot = torch.FloatTensor(torch.eye(4))

        self.model = model.to(self.device)
        self.model.eval()

        self.debug = dict()

    def postprocess(self, steer, throttle, brake):
        """
        Perform post - postprocess.

        Args:
            self: (todo): write your description
            steer: (todo): write your description
            throttle: (todo): write your description
            brake: (todo): write your description
        """
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False

        return control
