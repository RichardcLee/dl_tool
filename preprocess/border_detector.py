# richard lee
# 2020.11.1

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import conv2d


# 按某轴寻找第一个非零元
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


# 输入为[0,255] 灰度 label，计算其 border
def get_a_border1(im_path, out_path, scale=255):
    im = Image.open(im_path)
    pixels = np.array(im)
    border = np.zeros_like(pixels)

    invalid_val = -1
    top2bottom = first_nonzero(pixels, axis=0, invalid_val=-1)
    left2right = first_nonzero(pixels, axis=1, invalid_val=-1)
    bottom2top = first_nonzero(pixels[::-1, :], axis=0, invalid_val=-1)
    right2left = first_nonzero(pixels[:, ::-1], axis=1, invalid_val=-1)

    for c, r in enumerate(top2bottom):
        if r != invalid_val:
            border[r][c] = scale

    for c, r in enumerate(bottom2top):
        if r != invalid_val:
            border[-(r+1)][c] = scale

    for r, c in enumerate(left2right):
        if c != invalid_val:
            border[r][c] = scale

    for r, c in enumerate(right2left):
        if c != invalid_val:
            border[r][-(c+1)] = scale

    border_im = Image.fromarray(border)
    border_im.show()
    border_im.save(out_path)


# 输入为[0,255] 灰度 label，计算其 border
def get_a_border(im_path, out_path):
    im = Image.open(im_path)
    pixels_np = np.array(im)
    pixels = torch.FloatTensor(pixels_np).unsqueeze(0).unsqueeze(0)
    kernel = torch.FloatTensor(np.ones((3, 3))).unsqueeze(0).unsqueeze(0)

    filtering = conv2d(pixels, kernel, padding=1).squeeze(0).squeeze(0).numpy()
    border = np.zeros_like(filtering)

    mask = (filtering != 0) & (filtering != 255*9)
    border[mask] = 255
    border[pixels_np == 0] = 0

    border_im = Image.fromarray(border)
    # border_im.show()
    border_im.save(out_path)

