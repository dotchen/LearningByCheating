import torch

class SummaryWriter:
	def __init__(self, *args, **kwargs):
     """
     Initialize the module.

     Args:
         self: (todo): write your description
     """
		print("tensorboardX not found. You need to install it to use the SummaryWriter.")
		print("try: pip3 install tensorboardX")
		raise ImportError
		
class UnNormalize(object):
    def __init__(self, mean=[0.2929, 0.3123, 0.3292], std=[0.0762, 0.0726, 0.0801]):
        """
        Initialize standard deviation.

        Args:
            self: (todo): write your description
            mean: (float): write your description
            std: (array): write your description
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        new_tensor = tensor.new(*tensor.size())
        new_tensor[:, 0, :, :] = tensor[:, 0, :, :] * self.std[0] + self.mean[0]
        new_tensor[:, 1, :, :] = tensor[:, 1, :, :] * self.std[1] + self.mean[1]
        new_tensor[:, 2, :, :] = tensor[:, 2, :, :] * self.std[2] + self.mean[2]
        
        return new_tensor

try:
	from tensorboardX import SummaryWriter
except ImportError:
	pass

def one_hot(x, num_digits=4, start=1):
    """
    One - hot encoding of x.

    Args:
        x: (todo): write your description
        num_digits: (int): write your description
        start: (todo): write your description
    """
    N = x.size()[0]
    x = x.long()[:,None]-start
    x = torch.clamp(x, 0, num_digits-1)
    y = torch.FloatTensor(N, num_digits)
    y.zero_()
    y.scatter_(1, x, 1)
    return y
    
def viz_image_pred(rgb, pred_locations, gt_locations, dot_radius=2, unnormalizer=None):
    """
    Viz_image_predations

    Args:
        rgb: (todo): write your description
        pred_locations: (todo): write your description
        gt_locations: (todo): write your description
        dot_radius: (float): write your description
        unnormalizer: (bool): write your description
    """
    if unnormalizer:
        rgb_viz = unnormalizer(rgb.clone())
    else:
        rgb_viz = rgb.clone()
    for i, step_locations in enumerate(gt_locations.int()):
        for x, y in step_locations:
            rgb_viz[i,0,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            rgb_viz[i,1,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            rgb_viz[i,2,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 1
            
    for i, step_locations in enumerate(pred_locations.int()):
        for x, y in step_locations:
            rgb_viz[i,0,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 1
            rgb_viz[i,1,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            rgb_viz[i,2,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            
    return rgb_viz
    
def viz_birdview_pred(bird_view_viz, pred_locations, gt_locations, dot_radius=2):
    """
    Calculate the vizview position from the position.

    Args:
        bird_view_viz: (todo): write your description
        pred_locations: (todo): write your description
        gt_locations: (todo): write your description
        dot_radius: (float): write your description
    """
    for i, step_locations in enumerate(gt_locations.int()):
        for x, y in step_locations:
            bird_view_viz[i,0,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            bird_view_viz[i,1,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            bird_view_viz[i,2,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 1
            
    for i, step_locations in enumerate(pred_locations.int()):
        for x, y in step_locations:
            bird_view_viz[i,0,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 1
            bird_view_viz[i,1,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            bird_view_viz[i,2,y-dot_radius:y+dot_radius+1,x-dot_radius:x+dot_radius+1] = 0
            
    return bird_view_viz