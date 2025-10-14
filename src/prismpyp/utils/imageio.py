import matplotlib.pyplot as plt
import math
import numpy as np
import mrcfile

def load_mrc(filename):
    data = mrcfile.read(filename)
    return data

def mrc2png(data, outname=None):
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
    # im = Image.fromarray(rescaled)
    
    if outname is not None:
        im.save(outname)
    return im