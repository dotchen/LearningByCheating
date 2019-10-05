from pathlib import Path

import torch
import lmdb
import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.image_utils import draw_msra_gaussian, gaussian_radius
from utils.carla_utils import visualize_birdview

import math
import random

PIXEL_OFFSET = 10


def world_to_pixel(
        x,y,ox,oy,ori_ox, ori_oy,
        pixels_per_meter=5, offset=(-80,160), size=320, angle_jitter=15):
    pixel_dx, pixel_dy = (x-ox)*pixels_per_meter, (y-oy)*pixels_per_meter

    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox

    pixel_x = 320-pixel_x

    return np.array([pixel_x, pixel_y]) + offset


class BirdViewDataset(Dataset):
    def __init__(
            self, dataset_path,
            img_size=320, crop_size=192, gap=5, n_step=5,
            crop_x_jitter=5, crop_y_jitter=5, angle_jitter=5,
            down_ratio=4, gaussian_radius=1.0, max_frames=None):

        # These typically don't change.
        self.img_size = img_size
        self.crop_size = crop_size
        self.down_ratio = down_ratio
        self.gap = gap
        self.n_step = n_step

        self.max_frames = max_frames

        self.crop_x_jitter = crop_x_jitter
        self.crop_y_jitter = crop_y_jitter
        self.angle_jitter = angle_jitter

        self.gaussian_radius = gaussian_radius

        self._name_map = {}
        self.file_map = {}
        self.idx_map = {}

        self.bird_view_transform = transforms.ToTensor()

        n_episodes = 0

        for full_path in sorted(glob.glob('%s/**' % dataset_path), reverse=True):
            txn = lmdb.open(
                    full_path,
                    max_readers=1, readonly=True,
                    lock=False, readahead=False, meminit=False).begin(write=False)

            n = int(txn.get('len'.encode())) - self.gap * self.n_step
            offset = len(self._name_map)

            for i in range(n):
                if max_frames and len(self) >= max_frames:
                    break

                self._name_map[offset+i] = full_path
                self.file_map[offset+i] = txn
                self.idx_map[offset+i] = i

            n_episodes += 1

            if max_frames and len(self) >= max_frames:
                break

        print('%s: %d frames, %d episodes.' % (dataset_path, len(self), n_episodes))

    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, idx):
        lmdb_txn = self.file_map[idx]
        index = self.idx_map[idx]

        bird_view = np.frombuffer(lmdb_txn.get(('birdview_%04d'%index).encode()), np.uint8).reshape(320,320,7)
        measurement = np.frombuffer(lmdb_txn.get(('measurements_%04d'%index).encode()), np.float32)
        rgb_image = None

        ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, steer, throttle, brake, manual, gear  = measurement
        speed = np.linalg.norm([vx,vy,vz])

        oangle = np.arctan2(ori_oy, ori_ox)
        delta_angle = np.random.randint(-self.angle_jitter,self.angle_jitter+1)
        dx = np.random.randint(-self.crop_x_jitter,self.crop_x_jitter+1)
        dy = np.random.randint(0,self.crop_y_jitter+1) - PIXEL_OFFSET

        o_camx = ox + ori_ox*2
        o_camy = oy + ori_oy*2

        pixel_ox = 160
        pixel_oy = 260

        bird_view = cv2.warpAffine(
                bird_view,
                cv2.getRotationMatrix2D((pixel_ox,pixel_oy), delta_angle, 1.0),
                bird_view.shape[1::-1], flags=cv2.INTER_LINEAR)

        # random cropping
        center_x, center_y = 160, 260-self.crop_size//2
        bird_view = bird_view[
                dy+center_y-self.crop_size//2:dy+center_y+self.crop_size//2,
                dx+center_x-self.crop_size//2:dx+center_x+self.crop_size//2]

        angle = np.arctan2(ori_oy, ori_ox) + np.deg2rad(delta_angle)
        ori_ox, ori_oy = np.cos(angle), np.sin(angle)

        locations = []
        orientations = []

        for dt in range(self.gap, self.gap*(self.n_step+1), self.gap):
            lmdb_txn = self.file_map[idx]
            index =self.idx_map[idx]+dt

            f_measurement = np.frombuffer(lmdb_txn.get(("measurements_%04d"%index).encode()), np.float32)
            x, y, z, ori_x, ori_y = f_measurement[:5]

            pixel_y, pixel_x = world_to_pixel(x,y,ox,oy,ori_ox,ori_oy,size=self.img_size)
            pixel_x = pixel_x - (self.img_size-self.crop_size)//2
            pixel_y = self.crop_size - (self.img_size-pixel_y)+70

            pixel_x -= dx
            pixel_y -= dy

            angle = np.arctan2(ori_y, ori_x) - np.arctan2(ori_oy, ori_ox)
            ori_dx, ori_dy = np.cos(angle), np.sin(angle)

            locations.append([pixel_x, pixel_y])
            orientations.append([ori_dx, ori_dy])

        bird_view = self.bird_view_transform(bird_view)

        # Create mask
        output_size = self.crop_size // self.down_ratio
        heatmap_mask = np.zeros((self.n_step, output_size, output_size), dtype=np.float32)
        regression_offset = np.zeros((self.n_step,2), np.float32)
        indices = np.zeros((self.n_step), dtype=np.int64)

        for i, (pixel_x, pixel_y) in enumerate(locations):
            center = np.array(
                    [pixel_x / self.down_ratio, pixel_y / self.down_ratio],
                    dtype=np.float32)
            center = np.clip(center, 0, output_size-1)
            center_int = np.rint(center)

            draw_msra_gaussian(heatmap_mask[i], center_int, self.gaussian_radius)
            regression_offset[i] = center - center_int
            indices[i] = center_int[1] * output_size + center_int[0]

        return bird_view, np.array(locations), cmd, speed



class BiasedBirdViewDataset(BirdViewDataset):
    def __init__(self, dataset_path, left_ratio=0.25, right_ratio=0.25, straight_ratio=0.25, **kwargs):
        super().__init__(dataset_path, **kwargs)
        
        print ("Doing biased: %.2f,%.2f,%.2f"%(left_ratio, right_ratio, straight_ratio))
        
        self._choices = [1,2,3,4]
        self._weights = [left_ratio,right_ratio,straight_ratio,1-left_ratio-right_ratio-straight_ratio]
        # Separately save data on different cmd
        self.cmd_map = { i : set([]) for i in range(1,5)}
        
        for idx in range(len(self.file_map)):
            lmdb_txn = self.file_map[idx]
            index = self.idx_map[idx]
            
            measurement = np.frombuffer(lmdb_txn.get(('measurements_%04d'%index).encode()), np.float32)
            ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, steer, throttle, brake, manual, gear = measurement
            speed = np.linalg.norm([vx,vy,vz])
            
            if cmd != 4 and speed > 1.0:
                self.cmd_map[cmd].add(idx)
            else:
                self.cmd_map[4].add(idx)
            
        for cmd, nums in self.cmd_map.items():
            print (cmd, len(nums))

    def __getitem__(self, idx):
        cmd = np.random.choice(self._choices, p=self._weights)
        [_idx] = random.sample(self.cmd_map[cmd], 1)
        return super(BiasedBirdViewDataset, self).__getitem__(_idx)



def load_birdview_data(
        dataset_dir,
        batch_size=32, num_workers=0, shuffle=True,
        crop_x_jitter=0, crop_y_jitter=0, angle_jitter=0, n_step=5, gap=5,
        max_frames=None, cmd_biased=False):
    if cmd_biased:
        dataset_cls = BiasedBirdViewDataset
    else:
        dataset_cls = BirdViewDataset

    dataset = dataset_cls(
        dataset_path,
        crop_x_jitter=crop_x_jitter,
        crop_y_jitter=crop_y_jitter,
        angle_jitter=angle_jitter,
        n_step=n_step,
        gap=gap,
        data_ratio=data_ratio,
    )

    return DataLoader(
            dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, drop_last=True, pin_memory=True)


class Wrap(Dataset):
    def __init__(self, data, batch_size, samples):
        self.data = data
        self.batch_size = batch_size
        self.samples = samples

    def __len__(self):
        return self.batch_size * self.samples

    def __getitem__(self, i):
        return self.data[np.random.randint(len(self.data))]


def _dataloader(data, batch_size, num_workers):
    return DataLoader(
            data, batch_size=batch_size, num_workers=num_workers,
            shuffle=True, drop_last=True, pin_memory=True)


def get_birdview(
        dataset_dir,
        batch_size=32, num_workers=8, shuffle=True,
        crop_x_jitter=0, crop_y_jitter=0, angle_jitter=0, n_step=5, gap=5,
        max_frames=None, cmd_biased=False):

    def make_dataset(dir_name, is_train):
        _dataset_dir = str(Path(dataset_dir) / dir_name)
        _samples = 1000 if is_train else 10
        _crop_x_jitter = crop_x_jitter if is_train else 0
        _crop_y_jitter = crop_y_jitter if is_train else 0
        _angle_jitter = angle_jitter if is_train else 0
        _max_frames = max_frames if is_train else None
        _num_workers = num_workers if is_train else 0

        if is_train and cmd_biased:
            dataset_cls = BiasedBirdViewDataset
        else:
            dataset_cls = BirdViewDataset

        data = dataset_cls(
                _dataset_dir, gap=gap, n_step=n_step,
                crop_x_jitter=_crop_x_jitter, crop_y_jitter=_crop_y_jitter,
                angle_jitter=_angle_jitter,
                max_frames=_max_frames)
        data = Wrap(data, batch_size, _samples)
        data = _dataloader(data, batch_size, _num_workers)

        return data

    train = make_dataset('train', True)
    val = make_dataset('val', False)

    return train, val
