import matplotlib.pyplot as plt
import math
import numpy as np
import mrcfile
import os
from PIL import Image

def load_mrc(filename):
    data = mrcfile.read(filename)
    return data


def contrast_stretch_py(img, output_path=None, resize=100):
    # Convert to numpy for percentile-based stretch
    arr = np.array(img).astype(np.float32)
    p1, p99 = np.percentile(arr, (1, 98))  # same idea as -contrast-stretch 1%x2%
    stretched = np.clip((arr - p1) / (p99 - p1) * 255.0, 0, 255).astype(np.uint8)
    img_stretched = Image.fromarray(stretched)

    # Resize if needed
    if resize != 100:
        w, h = img_stretched.size
        new_size = (int(w * resize / 100), int(h * resize / 100))
        img_stretched = img_stretched.resize(new_size, Image.LANCZOS)

    # Save
    # if output_path is None:
    #     output_path = input_path
    # img_stretched.save(output_path)
    return img_stretched


def mrc2png(data, output_dims=(512, 512), outname=None, contrast_stretch=False):
    if math.fabs(data.max() - data.min()) > np.finfo(float).eps:
        rescaled = (255.0 * (data - data.min()) / (data.max() - data.min())).astype(
            np.uint8
        )
    elif math.fabs(data.max()) > np.finfo(float).eps:
        rescaled = ( 255.0 * data / data.max() ).astype(np.uint8)
    else:
        rescaled = np.abs(data).astype(np.uint8)
    
    # rescaled = np.flipud(rescaled)
               
    # flip to match mrc writeout
    from PIL import Image

    im = Image.fromarray(rescaled[::-1, :])
    if contrast_stretch:
        im = contrast_stretch_py(im)
    # im = Image.fromarray(rescaled)
    im = im.resize(output_dims, Image.LANCZOS)
    
    if outname is not None:
        im.save(outname)
    return im