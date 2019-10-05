import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from .resnet import get_resnet


CROP_SIZE = 192
MAP_SIZE = 320


def crop_birdview(birdview, dx=0, dy=0):
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview


def select_branch(branches, one_hot):
    shape = branches.size()

    for i, s in enumerate(shape[2:]):
        one_hot = torch.stack([one_hot for _ in range(s)], dim=i+2)

    return torch.sum(one_hot * branches, dim=1)


def signed_angle(u, v):
    theta = math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    if np.cross(u, v)[2] < 0:
        theta *= -1.0

    return theta


def project_point_to_circle(point, c, r):
    direction = point - c
    closest = c + (direction / np.linalg.norm(direction)) * r

    return closest


def make_arc(points, c, r):
    point_min = project_point_to_circle(points[0], c, r)
    point_max = project_point_to_circle(points[-1], c, r)

    theta_min = np.arctan2(point_min[1], point_min[0])
    theta_max = np.arctan2(point_max[1], point_max[0])

    # Probably a bug here.
    theta = np.linspace(theta_min, theta_max, 100)
    x1 = r * np.cos(theta) + c[0]
    x2 = r * np.sin(theta) + c[1]

    return np.stack([x1, x2], 1)


class ResnetBase(nn.Module):
    def __init__(self, backbone, input_channel=3, bias_first=True, pretrained=False):
        super().__init__()
        

        conv, c = get_resnet(
                backbone, input_channel=input_channel,
                bias_first=bias_first, pretrained=pretrained)

        self.conv = conv
        self.c = c

        self.backbone = backbone
        self.input_channel = input_channel
        self.bias_first = bias_first


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.mean = nn.Parameter(torch.FloatTensor(mean).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.FloatTensor(std).reshape(1, 3, 1, 1), requires_grad=False)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def forward(self, x):
        return (x - self.mean) / self.std


class NormalizeV2(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        
        self.mean = torch.FloatTensor(mean).reshape(1, 3, 1, 1).cuda()
        self.std = torch.FloatTensor(std).reshape(1, 3, 1, 1).cuda()

    def forward(self, x):
        return (x - self.mean) / self.std


class SpatialSoftmax(nn.Module):
    # Source: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super().__init__()

        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        weight = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(torch.autograd.Variable(self.pos_x)*weight, dim=1, keepdim=True)
        expected_y = torch.sum(torch.autograd.Variable(self.pos_y)*weight, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # feature_keypoints = expected_xy.view(-1, self.channel*2)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)

        return feature_keypoints


class SpatialSoftmaxBZ(torch.nn.Module):
    """
    IMPORTANT:
    i in [0, 1], where 0 is at the bottom, 1 is at the top
    j in [-1, 1]
    """
    def __init__(self, height, width):
        super().__init__()

        self.height = height
        self.width = width

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height),
            np.linspace(-1.0, 1.0, self.width)
        )

        self.pos_x = torch.from_numpy(pos_x).reshape(-1).float()
        self.pos_x = torch.nn.Parameter(self.pos_x, requires_grad=False)

        self.pos_y = torch.from_numpy(pos_y).reshape(-1).float()
        self.pos_y = torch.nn.Parameter(self.pos_y, requires_grad=False)

    def forward(self, feature):
        flattened = feature.view(feature.shape[0], feature.shape[1], -1)
        softmax = F.softmax(flattened, dim=-1)

        # This is not a bug.
        expected_x = torch.sum(self.pos_y * softmax, dim=-1)
        expected_x = (-expected_x + 1) / 2.0
        expected_y = torch.sum(self.pos_x * softmax, dim=-1)

        expected_xy = torch.stack([expected_x, expected_y], dim=2)

        return expected_xy


# tmp = SpatialSoftmax(48, 48)
#
# check = [(47, 0), (47, 24), (47, 47), (0, 24)]
#
# for i, j in check:
#     feature = np.zeros((48, 48))
#     feature[i,j] = 100
#     feature = torch.FloatTensor(feature).unsqueeze(0).unsqueeze(0)
#
#     print(i, j, tmp(feature))
