import math

import numpy as np

import torch
import torch.nn as nn

from . import common
from .agent import Agent
from .controller import CustomController, PIDController
from .controller import ls_circle


CROP_SIZE = 192
STEPS = 5
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5

        
class ImagePolicyModelSS(common.ResnetBase):
    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)
        
        self.c = {
                'resnet18': 512,
                'resnet34': 512,
                'resnet50': 2048
                }[backbone]
        self.warp = warp
        self.rgb_transform = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(self.c + 128),
            nn.ConvTranspose2d(self.c + 128,256,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(True),
        )
        
        if warp:
            ow,oh = 48,48
        else:
            ow,oh = 96,40 
        
        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,STEPS,1,1,0),
                common.SpatialSoftmax(ow,oh,STEPS),
            ) for i in range(4)
        ])
        
        self.all_branch = all_branch

    def forward(self, image, velocity, command):
        if self.warp:
            warped_image = tgm.warp_perspective(image, self.M, dsize=(192, 192))
            resized_image = resize_images(image)
            image = torch.cat([warped_image, resized_image], 1)
        
        
        image = self.rgb_transform(image)

        h = self.conv(image)
        b, c, kh, kw = h.size()
        
        # Late fusion for velocity
        velocity = velocity[...,None,None,None].repeat((1,128,kh,kw))
        
        h = torch.cat((h, velocity), dim=1)
        h = self.deconv(h)
        
        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)
        location_pred = common.select_branch(location_preds, command)

        if self.all_branch:
            return location_pred, location_preds
        
        return location_pred



class ImageAgent(Agent):
    def __init__(self, steer_points=None, pid=None, gap=5, camera_args={'x':384,'h':160,'fov':90,'world_y':1.4,'fixed_offset':4.0}, **kwargs):
        super().__init__(**kwargs)

        self.fixed_offset = float(camera_args['fixed_offset'])
        print ("Offset: ", self.fixed_offset)
        w = float(camera_args['w'])
        h = float(camera_args['h'])
        self.img_size = np.array([w,h])
        self.gap = gap
        
        if steer_points is None:
            steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

        if pid is None:
            pid = {
                "1" : {"Kp": 0.5, "Ki": 0.20, "Kd":0.0}, # Left
                "2" : {"Kp": 0.7, "Ki": 0.10, "Kd":0.0}, # Right
                "3" : {"Kp": 1.0, "Ki": 0.10, "Kd":0.0}, # Straight
                "4" : {"Kp": 1.0, "Ki": 0.50, "Kd":0.0}, # Follow
            }

        self.steer_points = steer_points
        self.turn_control = CustomController(pid)
        self.speed_control = PIDController(K_P=.8, K_I=.08, K_D=0.)
        
        self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0
        
        self.last_brake = -1

    def run_step(self, observations, teaching=False):
        rgb = observations['rgb'].copy()
        speed = np.linalg.norm(observations['velocity'])
        _cmd = int(observations['command'])
        command = self.one_hot[int(observations['command']) - 1]

        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            if self.model.all_branch:
                model_pred, _ = self.model(_rgb, _speed, _command)
            else:
                model_pred = self.model(_rgb, _speed, _command)

        model_pred = model_pred.squeeze().detach().cpu().numpy()
        
        pixel_pred = model_pred

        # Project back to world coordinate
        model_pred = (model_pred+1)*self.img_size/2

        world_pred = self.unproject(model_pred)

        targets = [(0, 0)]

        for i in range(STEPS):
            pixel_dx, pixel_dy = world_pred[i]
            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy])

            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        targets = np.array(targets)

        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (self.gap * DT)
        
        c, r = ls_circle(targets)
        n = self.steer_points.get(str(_cmd), 1)
        closest = common.project_point_to_circle(targets[n], c, r)
        
        acceleration = np.clip(target_speed - speed, 0.0, 1.0)

        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)

        steer = self.turn_control.run_step(alpha, _cmd)
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        # Slow or stop.
        
        if target_speed <= self.engine_brake_threshold:
            steer = 0.0
            throttle = 0.0
        
        if target_speed <= self.brake_threshold:
            brake = 1.0
            
        self.debug = {
                # 'curve': curve,
                'target_speed': target_speed,
                'target': closest,
                'locations_world': targets,
                'locations_pixel': model_pred.astype(int),
                }
        
        control = self.postprocess(steer, throttle, brake)
        if teaching:
            return control, pixel_pred
        else:
            return control

    def unproject(self, output, world_y=1.4, fov=90):

        cx, cy = self.img_size / 2
        
        w, h = self.img_size
        
        f = w /(2 * np.tan(fov * np.pi / 360))
        
        xt = (output[...,0:1] - cx) / f
        yt = (output[...,1:2] - cy) / f
        
        world_z = world_y / yt
        world_x = world_z * xt
        
        world_output = np.stack([world_x, world_z],axis=-1)
        
        if self.fixed_offset:
            world_output[...,1] -= self.fixed_offset
        
        world_output = world_output.squeeze()
        
        return world_output
