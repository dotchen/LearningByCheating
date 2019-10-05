import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms

import carla

from .resnet import get_resnet
from .common import select_branch, Normalize
from .agent import Agent


def BaselineBranch(p):
    return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p),

            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p),

            nn.Linear(256, 3))


class Baseline(nn.Module):
    def __init__(self, backbone='resnet18', dropout=0.5, **kwargs):
        super().__init__()

        conv, c = get_resnet(backbone, input_channel=3)

        self.conv = conv
        self.c = c
        self.global_avg_pool = nn.AvgPool2d((40, 96))

        self.rgb_transform = Normalize(
                mean=[0.31, 0.33, 0.36],
                std=[0.18, 0.18, 0.19],
            )

        self.speed_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

        self.joint = nn.Sequential(
            nn.Linear(c+128, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

        self.speed = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),

            nn.Linear(256, 1),
        )

        self.branches = nn.ModuleList([BaselineBranch(p=dropout) for i in range(4)])

    def forward(self, image, velocity, command):
        h = self.conv(self.rgb_transform(image))
        h = self.global_avg_pool(h).view(-1, self.c)
        v = self.speed_encoder(velocity[...,None])

        h = torch.cat([h, v], dim=1)
        h = self.joint(h)

        branch_outputs = [control(h) for control in self.branches]
        branch_outputs = torch.stack(branch_outputs, dim=1)

        control = select_branch(branch_outputs, command)
        speed = self.speed(h)

        return control, speed


class BaselineAgent(Agent):
    def run_step(self, observations):
        rgb = observations['rgb'].copy()
        speed = np.linalg.norm(observations['velocity'])
        command = self.one_hot[int(observations['command']) - 1]

        with torch.no_grad():
            _rgb = (self.transform(rgb)[None]).to(self.device)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = one_hot(torch.FloatTensor([command])).to(self.device)

            _control, _ = self.model(_rgb, _speed, _command)
            steer, throttle, brake = map(float, _control.cpu().numpy().squeeze())

        if not hasattr(self, 'hack'):
            self.hack = 0

        if self.hack < 20:
            speed = 2
            throttle = 0.5
            brake = 0

        self.hack += 1

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        return control
