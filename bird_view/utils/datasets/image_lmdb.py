from pathlib import Path

import torch
import lmdb
import os
import glob
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import math
import random

import augmenter

PIXEL_OFFSET = 10
PIXELS_PER_METER = 5
    
def world_to_pixel(x,y,ox,oy,ori_ox, ori_oy, offset=(-80,160), size=320, angle_jitter=15):
    pixel_dx, pixel_dy = (x-ox)*PIXELS_PER_METER, (y-oy)*PIXELS_PER_METER
    
    pixel_x = pixel_dx*ori_ox+pixel_dy*ori_oy
    pixel_y = -pixel_dx*ori_oy+pixel_dy*ori_ox
    
    pixel_x = 320-pixel_x
    
    return np.array([pixel_x, pixel_y]) + offset
    

def project_to_image(pixel_x, pixel_y, tran=[0.,0.,0.], rot=[0.,0.,0.], fov=90, w=384, h=160, camera_world_z=1.4, crop_size=192):
    # Apply fixed offset tp pixel_y
    pixel_y -= 2*PIXELS_PER_METER
    
    pixel_y = crop_size - pixel_y
    pixel_x = pixel_x - crop_size/2
    
    world_x = pixel_x / PIXELS_PER_METER
    world_y = pixel_y / PIXELS_PER_METER
    
    xyz = np.zeros((1,3))
    xyz[0,0] = world_x
    xyz[0,1] = camera_world_z
    xyz[0,2] = world_y

    f = w /(2 * np.tan(fov * np.pi / 360))
    A = np.array([
        [f, 0., w/2],
        [0, f, h/2],
        [0., 0., 1.]
    ])
    image_xy, _ = cv2.projectPoints(xyz, np.array(tran), np.array(rot), A, None)
    image_xy[...,0] = np.clip(image_xy[...,0], 0, w)
    image_xy[...,1] = np.clip(image_xy[...,1], 0, h)

    return image_xy[0,0]
    
class ImageDataset(Dataset):
    def __init__(self, 
        dataset_path,
        rgb_shape=(160,384,3),
        img_size=320,
        crop_size=192,
        gap=5, 
        n_step=5,
        gaussian_radius=1.,
        down_ratio=4,
        # rgb_mean=[0.29813555, 0.31239682, 0.33620676],
        # rgb_std=[0.0668446, 0.06680295, 0.07329721],
        augment_strategy=None,
        batch_read_number=819200,
        batch_aug=1,
    ):
        self._name_map = {}
        
        self.file_map = {}
        self.idx_map = {}

        self.bird_view_transform = transforms.ToTensor()
        self.rgb_transform = transforms.ToTensor()
        
        self.rgb_shape = rgb_shape
        self.img_size = img_size
        self.crop_size = crop_size
        
        self.gap = gap
        self.n_step = n_step
        self.down_ratio = down_ratio
        self.batch_aug = batch_aug
        
        self.gaussian_radius = gaussian_radius
        
        print ("augment with ", augment_strategy)
        if augment_strategy is not None and augment_strategy != 'None':
            self.augmenter = getattr(augmenter, augment_strategy)
        else:
            self.augmenter = None

        count = 0
        for full_path in glob.glob('%s/**'%dataset_path):
            # hdf5_file = h5py.File(full_path, 'r', libver='latest', swmr=True)
            lmdb_file = lmdb.open(full_path,
                 max_readers=1,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 meminit=False
            )
            
            txn = lmdb_file.begin(write=False)
            
            N = int(txn.get('len'.encode())) - self.gap*self.n_step
            
            for _ in range(N):
                self._name_map[_+count] = full_path
                self.file_map[_+count] = txn
                self.idx_map[_+count] = _
                
            count += N
        
        print ("Finished loading %s. Length: %d"%(dataset_path, count))
        self.batch_read_number = batch_read_number
        
    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, idx):

        lmdb_txn = self.file_map[idx]
        index = self.idx_map[idx]
        
        bird_view = np.frombuffer(lmdb_txn.get(('birdview_%04d'%index).encode()), np.uint8).reshape(320,320,7)
        measurement = np.frombuffer(lmdb_txn.get(('measurements_%04d'%index).encode()), np.float32)
        rgb_image = np.fromstring(lmdb_txn.get(('rgb_%04d'%index).encode()), np.uint8).reshape(160,384,3)

        if self.augmenter:
            rgb_images = [self.augmenter(self.batch_read_number).augment_image(rgb_image) for i in range(self.batch_aug)]
        else:
            rgb_images = [rgb_image for i in range(self.batch_aug)]
            
        if self.batch_aug == 1:
            rgb_images = rgb_images[0]
                            
        ox, oy, oz, ori_ox, ori_oy, vx, vy, vz, ax, ay, az, cmd, steer, throttle, brake, manual, gear  = measurement
        speed = np.linalg.norm([vx,vy,vz])
        
        oangle = np.arctan2(ori_oy, ori_ox)
        delta_angle = 0
        dx = 0
        dy = -PIXEL_OFFSET
            
        pixel_ox = 160
        pixel_oy = 260
        
        rot_mat = cv2.getRotationMatrix2D((pixel_ox,pixel_oy), delta_angle, 1.0)
        bird_view = cv2.warpAffine(bird_view, rot_mat, bird_view.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        # random cropping
        center_x, center_y = 160, 260-self.crop_size//2
        
            
        bird_view = bird_view[dy+center_y-self.crop_size//2:dy+center_y+self.crop_size//2,dx+center_x-self.crop_size//2:dx+center_x+self.crop_size//2]
        
        angle = np.arctan2(ori_oy, ori_ox) + np.deg2rad(delta_angle)
        ori_ox, ori_oy = np.cos(angle), np.sin(angle)
        
        locations = []

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
            
            # Coordinate transform
            
            locations.append([pixel_x, pixel_y])
        
        if self.batch_aug == 1:
            rgb_images = self.rgb_transform(rgb_images)
        else:
            # if len()
            #     import pdb; pdb.set_trace()
            rgb_images = torch.stack([self.rgb_transform(img) for img in rgb_images])
        bird_view = self.bird_view_transform(bird_view)
        
        # Create mask
        # output_h = self.rgb_shape[0] // self.down_ratio
        # output_w = self.rgb_shape[1] // self.down_ratio
        # heatmap_mask = np.zeros((self.n_step, output_h, output_w), dtype=np.float32)
        # regression_offset = np.zeros((self.n_step,2), np.float32)
        # indices = np.zeros((self.n_step), dtype=np.int64)
        
        # image_locations = []

        # for i, (pixel_x, pixel_y) in enumerate(locations):
        #     image_pixel_x, image_pixel_y = project_to_image(pixel_x, pixel_y)
            
        #     image_locations.append([image_pixel_x, image_pixel_y])

        #     center = np.array([image_pixel_x / self.down_ratio, image_pixel_y / self.down_ratio], dtype=np.float32)
        #     center = np.clip(center, (0,0), (output_w-1, output_h-1))
            
        #     center_int = np.rint(center)
            
        #     # draw_msra_gaussian(heatmap_mask[i], center_int, self.gaussian_radius)
        #     regression_offset[i] = center - center_int
            # indices[i] = center_int[1] * output_w + center_int[0]
            
        self.batch_read_number += 1
       
        return rgb_images, bird_view, np.array(locations), cmd, speed

        
def load_image_data(dataset_path, 
        batch_size=32, 
        num_workers=8,
        shuffle=True, 
        n_step=5,
        gap=10,
        augment=None,
        **kwargs
        # rgb_mean=[0.29813555, 0.31239682, 0.33620676],
        # rgb_std=[0.0668446, 0.06680295, 0.07329721],
    ):

    dataset = ImageDataset(
        dataset_path,
        n_step=n_step,
        gap=gap,
        augment_strategy=augment,
        **kwargs,
        # rgb_mean=rgb_mean,
        # rgb_std=rgb_std,
    )
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=True, pin_memory=True)
    

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


def get_image(
        dataset_dir,
        batch_size=32, num_workers=0, shuffle=True, augment=None,
        n_step=5, gap=5, batch_aug=1):

    # import pdb; pdb.set_trace()

    def make_dataset(dir_name, is_train):
        _dataset_dir = str(Path(dataset_dir) / dir_name)
        _samples = 1000 if is_train else 10
        _num_workers = num_workers if is_train else 0
        _batch_aug = batch_aug if is_train else 1
        _augment = augment if is_train else None

        data = ImageDataset(
                _dataset_dir, gap=gap, n_step=n_step, augment_strategy=_augment, batch_aug=_batch_aug)
        data = Wrap(data, batch_size, _samples)
        data = _dataloader(data, batch_size, _num_workers)

        return data

    train = make_dataset('train', True)
    val = make_dataset('val', False)

    return train, val
    
    
if __name__ == '__main__':
    batch_size = 256
    import tqdm
    dataset = ImageDataset('/raid0/dian/carla_0.9.6_data/train')
    loader = _dataloader(dataset, batch_size=batch_size, num_workers=16)
    mean = []
    for rgb_img, bird_view, locations, cmd, speed in tqdm.tqdm(loader):
        mean.append(rgb_img.mean(dim=(0,2,3)).numpy())

    print ("Mean: ", np.mean(mean, axis=0))
    print ("Std: ", np.std(mean, axis=0)*np.sqrt(batch_size))
