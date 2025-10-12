import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from mrc_dataset import MRCDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Filter by brightspots')
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing the MRC files')
    parser.add_argument('--nextpyp_preproc_dir', type=str,
                        help='NextPYP preprocessing directory')
    parser.add_argument('--metadata_path', type=str,
                        help='Path to the metadata file')
    parser.add_argument('--output_path', type=str,
                        help='Directory to save the filtered images')
    parser.add_argument('--link_type', type=str, choices=['hard', 'soft'], default='soft',
                        help='Type of link to create: hard or soft (Default: soft)')
    parser.add_argument('--threshold', type=float, default=5,
                        help='Resolution, in A, for the Fourier domain high-pass filter')
    parser.add_argument('--iciness_threshold', type=float, default=0.2,
                        help='Threshold for determining if an image is crystalline (lower == more conservative)')
    return parser.parse_args()


def pixel_to_frequency(img, pixel_size):
    if type(img) == Image.Image:
        img = np.array(img)
    if img.ndim == 3:
        h, w, _ = img.shape
    elif img.ndim == 2:
        h, w = img.shape
        
    f_h = np.fft.fftfreq(h, d=pixel_size)
    f_w = np.fft.fftfreq(w, d=pixel_size)
    return f_h, f_w


def create_hpf_mask(f_h, f_w, cutoff):
    fx, fy = np.meshgrid(f_h, f_w)  # Create 2D frequency grids
    fx_shifted = np.fft.fftshift(fx)  # Shift the zero frequency component to the center
    fy_shifted = np.fft.fftshift(fy)

    # Compute the radial frequency (spatial frequency magnitude)
    freq_r = np.sqrt(fx_shifted**2 + fy_shifted**2)  # Radial frequency map (1/Å)

    cutoff_freq = 1 / cutoff
    mask = freq_r >= cutoff_freq
    return mask

    
def main(args):
    metadata = pd.read_csv(args.metadata_path)

    augmentation = [
        transforms.Resize(512),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]

    train_dataset = MRCDataset(
        mrc_dir=args.data_dir, 
        transform=transforms.Compose(augmentation),
        webp_dir=os.path.join(args.nextpyp_preproc_dir, 'webp'),
        # is_fft=args.use_fft,
        metadata_path=args.metadata_path
    )
    
    pixel_size = train_dataset.pixel_size
    
    # Check that the threshold is within the Nyquist resolution
    nyquist = 2 * pixel_size
    if args.threshold < nyquist:
        raise ValueError(f"Threshold {args.threshold} is above the Nyquist resolution {nyquist}")
    
    # Make output subfolders: crystalline, vitreous
    crystalline_dir = os.path.join(args.output_path, 'icy')
    vitreous_dir = os.path.join(args.output_path, 'vitreous')
    os.makedirs(crystalline_dir, exist_ok=True)
    os.makedirs(vitreous_dir, exist_ok=True)
    
    iciness_dict = {}
    
    for i in tqdm(range(len(train_dataset)), desc="Processing images"):
        filename = train_dataset.file_paths[i]
        basename = os.path.basename(filename)
        
        power_spectrum = train_dataset.get_fft_image(i)
        power_spectrum = np.mean(np.array(power_spectrum), axis=2)
        f_h, f_w = pixel_to_frequency(power_spectrum, pixel_size)
        mask = create_hpf_mask(f_h, f_w, cutoff=args.threshold)
        masked_image = mask * power_spectrum
        
        num_white_pixels = np.count_nonzero(masked_image > 250)
        num_masked_black_pixels = np.count_nonzero(mask == 0)
        iciness = num_white_pixels/num_masked_black_pixels
        
        iciness_dict[basename] = iciness
        
        if iciness > args.iciness_threshold:
            dest_dir = crystalline_dir
        else:
            dest_dir = vitreous_dir
        
        if args.link_type == 'hard':
            os.link(filename, os.path.join(dest_dir, basename))
        else:
            os.symlink(filename, os.path.join(dest_dir, basename))
    
    # Convert iciness_dict to DataFrame and save to .csv
    iciness_df = pd.DataFrame(iciness_dict.items(), columns=['filename', 'iciness'])
    iciness_df.to_csv(os.path.join(args.output_path, 'iciness.csv'), index=False)
    
    print("Finished processing images")
    
    # Get number of images in each category
    num_crystalline = len(os.listdir(crystalline_dir))
    num_vitreous = len(os.listdir(vitreous_dir))
    
    print(f"Number of crystalline images: {num_crystalline}")
    print(f"Number of vitreous images: {num_vitreous}")
    
    return
    
if __name__ == "__main__":
    args = parse_args()
    main(args)