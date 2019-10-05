import bz_utils.video_maker as video_maker


import time
import numpy as np


tmp = np.zeros((256, 128, 3), dtype=np.uint8)
video_maker.init()


for i in range(256):
    tmp[:,:,0] += 1
    video_maker.add(tmp)

    if i == 100:
        break
