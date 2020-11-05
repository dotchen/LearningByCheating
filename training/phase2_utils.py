import torch
import numpy as np
import random
import augmenter
from torchvision import transforms
import torchvision.transforms.functional as TF

import sys
import glob
try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
except IndexError as e:
    pass

import utils.carla_utils as cu
from models.image import ImagePolicyModelSS
from models.birdview import BirdViewPolicyModelSS

CROP_SIZE = 192
PIXELS_PER_METER = 5


def repeat(a, repeats, dim=0):
    """
    Substitute for numpy's repeat function. Taken from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)]))
    else:
        order_index = torch.LongTensor(
            torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)


def get_weight(learner_points, teacher_points):
    """
    Calculate the weight of the weight.

    Args:
        learner_points: (str): write your description
        teacher_points: (str): write your description
    """
    decay = torch.FloatTensor([0.7**i for i in range(5)]).to(learner_points.device)
    xy_bias = torch.FloatTensor([0.7,0.3]).to(learner_points.device)
    loss_weight = torch.mean((torch.abs(learner_points - teacher_points)*xy_bias).sum(dim=-1)*decay, dim=-1)
    x_weight = torch.max(
        torch.mean(teacher_points[...,0],dim=-1),
        torch.mean(teacher_points[...,0]*-1.4,dim=-1),
    )
    
    return loss_weight

def weighted_random_choice(weights):
    """
    Return a random weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weighted weights.

    Args:
        weights: (array): write your description
    """
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t, random.uniform(0,s))

def get_optimizer(parameters, lr=1e-4):
    """
    Get the optimizer.

    Args:
        parameters: (todo): write your description
        lr: (str): write your description
    """
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    return optimizer

def load_image_model(backbone, ckpt, device='cuda'):
    """
    Loads an image from disk

    Args:
        backbone: (str): write your description
        ckpt: (str): write your description
        device: (str): write your description
    """
    net = ImagePolicyModelSS(
        backbone,
        all_branch=True
    ).to(device)
    
    net.load_state_dict(torch.load(ckpt))
    return net
    
def _log_visuals(rgb_image, birdview, speed, command, loss, pred_locations, _pred_locations, _teac_locations, size=16):
    """
    Log_visualization.

    Args:
        rgb_image: (todo): write your description
        birdview: (array): write your description
        speed: (array): write your description
        command: (list): write your description
        loss: (todo): write your description
        pred_locations: (todo): write your description
        _pred_locations: (todo): write your description
        _teac_locations: (todo): write your description
        size: (int): write your description
    """
    import cv2
    import numpy as np
    import utils.carla_utils as cu

    WHITE = [255, 255, 255]
    BLUE = [0, 0, 255]
    RED = [255, 0, 0]
    _numpy = lambda x: x.detach().cpu().numpy().copy()

    images = list()

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()
        canvas = np.uint8(_numpy(birdview[i]).transpose(1, 2, 0) * 255).copy()
        canvas = cu.visualize_birdview(canvas)
        rgb = np.uint8(_numpy(rgb_image[i]).transpose(1, 2, 0) * 255).copy()
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

        def _write(text, i, j):
            """
            Write text to the t.

            Args:
                text: (str): write your description
                i: (todo): write your description
                j: (todo): write your description
            """
            cv2.putText(
                    canvas, text, (cols[j], rows[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

        def _dot(_canvas, i, j, color, radius=2):
            """
            Dot product ( i j )

            Args:
                _canvas: (array): write your description
                i: (array): write your description
                j: (array): write your description
                color: (str): write your description
                radius: (array): write your description
            """
            x, y = int(j), int(i)
            _canvas[x-radius:x+radius+1, y-radius:y+radius+1] = color
        
        def _stick_together(a, b):
            """
            Concat ( b )

            Args:
                a: (str): write your description
                b: (str): write your description
            """
            h = min(a.shape[0], b.shape[0])
    
            r1 = h / a.shape[0]
            r2 = h / b.shape[0]
    
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
    
            return np.concatenate([a, b], 1)
        
        _command = {
                1: 'LEFT', 2: 'RIGHT',
                3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item()+1, '???')

        _dot(canvas, 0, 0, WHITE)

        for x, y in (_teac_locations[i] + 1) * (0.5 * CROP_SIZE): _dot(canvas, x, y, BLUE)
        for x, y in _pred_locations[i]: _dot(rgb, x, y, RED)
        for x, y in pred_locations[i]: _dot(canvas, x, y, RED)

        _write('Command: %s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)
        
        
        images.append((loss[i].item(), _stick_together(rgb, canvas)))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]
    
def load_birdview_model(backbone, ckpt, device='cuda'):
    """
    Loads a preview model object.

    Args:
        backbone: (todo): write your description
        ckpt: (todo): write your description
        device: (str): write your description
    """
    teacher_net = BirdViewPolicyModelSS(backbone, all_branch=True).to(device)
    teacher_net.load_state_dict(torch.load(ckpt))
    
    return teacher_net
    
class CoordConverter():
    def __init__(self, w=384, h=160, fov=90, world_y=1.4, fixed_offset=4.0, device='cuda'):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            w: (int): write your description
            h: (int): write your description
            fov: (todo): write your description
            world_y: (todo): write your description
            fixed_offset: (int): write your description
            device: (todo): write your description
        """
        self._img_size = torch.FloatTensor([w,h]).to(device)
        
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
        print ("Fixed offset", fixed_offset)
    
    def __call__(self, camera_locations):
        """
        Calculate the camera.

        Args:
            self: (todo): write your description
            camera_locations: (str): write your description
        """
        if isinstance(camera_locations, torch.Tensor):
            camera_locations = (camera_locations + 1) * self._img_size/2
        else:
            camera_locations = (camera_locations + 1) * self._img_size.cpu().numpy()/2
        
        w, h = self._img_size
        w = int(w)
        h = int(h)
        
        cx, cy = w/2, h/2

        f = w /(2 * np.tan(self._fov * np.pi / 360))
    
        xt = (camera_locations[...,0] - cx) / f
        yt = (camera_locations[...,1] - cy) / f

        world_z = self._world_y / yt
        world_x = world_z * xt
        
        if isinstance(camera_locations, torch.Tensor):
            map_output = torch.stack([world_x, world_z],dim=-1)
        else:
            map_output = np.stack([world_x,world_z],axis=-1)
    
        map_output *= PIXELS_PER_METER
        map_output[...,1] = CROP_SIZE - map_output[...,1]
        map_output[...,0] += CROP_SIZE/2
        map_output[...,1] += self._fixed_offset*PIXELS_PER_METER
        
        return map_output

class LocationLoss(torch.nn.Module):
    def forward(self, pred_locations, teac_locations):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            pred_locations: (todo): write your description
            teac_locations: (todo): write your description
        """
        pred_locations = pred_locations/(0.5*CROP_SIZE) - 1
        
        return torch.mean(torch.abs(pred_locations - teac_locations), dim=(1,2,3))

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, buffer_limit=100000, augment=None, sampling=True, aug_fix_iter=1000000, batch_aug=4):
        """
        Initialize the sampling.

        Args:
            self: (todo): write your description
            buffer_limit: (int): write your description
            augment: (str): write your description
            sampling: (todo): write your description
            aug_fix_iter: (str): write your description
            batch_aug: (todo): write your description
        """
        self.buffer_limit = buffer_limit
        self._data = []
        self._weights = []
        self.rgb_transform = transforms.ToTensor()
        
        self.birdview_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if augment and augment != 'None':
            self.augmenter = getattr(augmenter, augment)
        else:
            self.augmenter = None
            
        self.normalized = False
        self._sampling = sampling
        self.aug_fix_iter = aug_fix_iter
        self.batch_aug = batch_aug
            
    def __len__(self):
        """
        Returns the length of the data.

        Args:
            self: (todo): write your description
        """
        return len(self._data)

    def __getitem__(self, _idx):
        """
        Return an image item.

        Args:
            self: (todo): write your description
            _idx: (str): write your description
        """
        if self._sampling and self.normalized:
            while True:
                idx = weighted_random_choice(self._weights)
                if idx < len(self._data):
                    break
                print ("waaat")
        else:
            idx = _idx
            
        rgb_img, cmd, speed, target, birdview_img = self._data[idx]
        if self.augmenter:
            rgb_imgs = [self.augmenter(self.aug_fix_iter).augment_image(rgb_img) for i in range(self.batch_aug)]
        else:
            rgb_imgs = [rgb_img for i in range(self.batch_aug)]

        rgb_imgs = [self.rgb_transform(img) for img in rgb_imgs]
        if self.batch_aug == 1:
            rgb_imgs = rgb_imgs[0]
        else:
            rgb_imgs = torch.stack(rgb_imgs)

        birdview_img = self.birdview_transform(birdview_img)

        return idx, rgb_imgs, cmd, speed, target, birdview_img
        
    def update_weights(self, idxes, losses):
        """
        Update the weights with the weights.

        Args:
            self: (todo): write your description
            idxes: (array): write your description
            losses: (todo): write your description
        """
        idxes = idxes.numpy()
        losses = losses.detach().cpu().numpy()
        for idx, loss in zip(idxes, losses):
            if idx > len(self._data):
                continue

            self._new_weights[idx] = loss
            
    def init_new_weights(self):
        """
        Initialize new weights.

        Args:
            self: (todo): write your description
        """
        self._new_weights = self._weights.copy()
            
    def normalize_weights(self):
        """
        Normalize the weights.

        Args:
            self: (array): write your description
        """
        self._weights = self._new_weights
        self.normalized = True

    def add_data(self, rgb_img, cmd, speed, target, birdview_img, weight):
        """
        Add an rgb image.

        Args:
            self: (todo): write your description
            rgb_img: (str): write your description
            cmd: (str): write your description
            speed: (int): write your description
            target: (str): write your description
            birdview_img: (str): write your description
            weight: (str): write your description
        """
        self.normalized = False
        self._data.append((rgb_img, cmd, speed, target, birdview_img))
        self._weights.append(weight)
            
        if len(self._data) > self.buffer_limit:
            # Pop the one with lowest loss
            idx = np.argsort(self._weights)[0]
            self._data.pop(idx)
            self._weights.pop(idx)
            
            
    def remove_data(self, idx):
        """
        Removes the element from the given index.

        Args:
            self: (todo): write your description
            idx: (int): write your description
        """
        self._weights.pop(idx)
        self._data.pop(idx)
            
    def get_highest_k(self, k):
        """
        Return the k k k k k k k k ( k.

        Args:
            self: (todo): write your description
            k: (todo): write your description
        """
        top_idxes = np.argsort(self._weights)[-k:]
        rgb_images = []
        bird_views = []
        targets = []
        cmds = []
        speeds = []
        
        for idx in top_idxes:
            if idx < len(self._data):
                rgb_img, cmd, speed, target, birdview_img = self._data[idx]
                rgb_images.append(TF.to_tensor(np.ascontiguousarray(rgb_img)))
                bird_views.append(TF.to_tensor(birdview_img))
                cmds.append(cmd)
                speeds.append(speed)
                targets.append(target)
        
        return torch.stack(rgb_images), torch.stack(bird_views), torch.FloatTensor(cmds), torch.FloatTensor(speeds), torch.FloatTensor(targets)
