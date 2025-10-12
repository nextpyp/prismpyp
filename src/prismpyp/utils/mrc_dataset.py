import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import numpy as np
import mrcfile
from glob import glob
# import fft
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MRCDataset(Dataset):
    def __init__(self, mrc_dir, transform, webp_dir, metadata_path, is_fft=False, pixel_size=1.0):
        """
        Args:
            mrc_dir (str): List of .mrc files.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.mrc_dir = mrc_dir
        self.transform = transform
        self.is_fft = is_fft
        self.file_paths = []
        self.real_images = {}
        self.fft_images = {}
        self.webp_dir = webp_dir
        self.metadata = pd.read_csv(metadata_path)
        self.scaler = StandardScaler() # z-score norm
        # self.labels = []

        with open(mrc_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.file_paths.append(line.strip())
                self.real_images[line.strip()] = None
                self.fft_images[line.strip()] = None

        self.pixel_size = pixel_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the .mrc file and convert to Fourier representation
        img_path = self.file_paths[idx]
        basename = img_path # os.path.basename(img_path)
        
        if self.is_fft:
            path_to_powerspectrum = os.path.join(self.webp_dir, basename + '_ctffit.webp')
            powerspectrum = Image.open(path_to_powerspectrum)
            img = self.crop_and_mirror_image(powerspectrum)
        else:
            path_to_mg = os.path.join(self.webp_dir, basename + '.webp')
            img = Image.open(path_to_mg)
            
        if self.transform:
            views = self.transform(img)
        else:
            views = img
        
        return views
        
    def get_real_image(self, idx):
        img_path = self.file_paths[idx]
        basename = img_path #os.path.basename(img_path)
        path_to_mg = os.path.join(self.webp_dir, basename + '.webp')
        mrc_data = Image.open(path_to_mg)
        return mrc_data
    
    def get_fft_image(self, idx):
        img_path = self.file_paths[idx]
        basename = img_path #os.path.basename(img_path)
        path_to_mg = os.path.join(self.webp_dir, basename + '_ctffit.webp')
        ctf_img = Image.open(path_to_mg)
        fft_img = self.crop_and_mirror_image(ctf_img)
        return fft_img
    
    def get_ice_thickness(self, idx):
        img_path = self.file_paths[idx]
        basename = img_path #os.path.basename(img_path)[:-4]
        row = self.metadata[self.metadata['micrograph_name'] == basename]
        ret = row['rel_ice_thickness'].values[0]
        return self.scaler.fit_transform(np.array([ret]).reshape(-1, 1))[0][0]
    
    def get_ctf_fit(self, idx):
        img_path = self.file_paths[idx]
        basename = img_path # os.path.basename(img_path)[:-4]
        row = self.metadata[self.metadata['micrograph_name'] == basename]
        ret = row['ctf_fit'].values[0]
        return self.scaler.fit_transform(np.array([ret]).reshape(-1, 1))[0][0]
    
    def get_est_resolution(self, idx):
        img_path = self.file_paths[idx]
        basename = img_path # os.path.basename(img_path)[:-4]
        row = self.metadata[self.metadata['micrograph_name'] == basename]
        ret = row['est_resolution'].values[0]
        return self.scaler.fit_transform(np.array([ret]).reshape(-1, 1))[0][0]

    def get_defocus(self, idx):
        img_path = self.file_paths[idx]
        basename = img_path #os.path.basename(img_path)[:-4]
        row = self.metadata[self.metadata['micrograph_name'] == basename]
        ret = row['mean_defocus'].values[0]
        return ret
                               
    # Normalize an image to have pixel values between 0 and 255
    # Use the 1st and 99th percentile to avoid extreme outlier values    
    def soft_normalize(self, img):
        p1, p99 = np.percentile(img, (1, 99))
        normalized = np.clip((img - p1) / (p99 - p1), 0, 1)
        normalized *= 255
        normalized = normalized.astype(np.uint8)
        return normalized
    
    def get_nyquist(self, pixel_size):
        # Check if the pixel size is below the Nyquist frequency
        # Returns the Nyquist pixel size, *NOT* the Nyquist frequency
        nyquist = 2 * pixel_size
        return nyquist
    
    def crop_and_mirror_image(self, image1):
        image1_cropped = image1.crop((0, 0, image1.width, image1.height // 2))
        image1_flipped_lr = image1_cropped.transpose(Image.FLIP_LEFT_RIGHT)
        image1_flipped_tb = image1_flipped_lr.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Create a new image with combined width and the maximum height
        new_width = max(image1_cropped.size[0], image1_flipped_tb.size[0])
        new_height = image1_cropped.size[1] + image1_flipped_tb.size[1]

        # Create a new blank image (white background)
        new_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))

        # Paste the first image at the left side of the new image
        new_image.paste(image1_cropped, (0, 0))

        # Paste the second image at the bottom half of the new image
        new_image.paste(image1_flipped_tb, (0, image1_cropped.size[1]))
        
        return new_image
    
    def get_index_from_filename(self, filename):
        for idx, path in enumerate(self.file_paths):
            if filename in path:
                return idx
        return None
    
class WeakSupMRCDataset(Dataset):
    def __init__(self, mrc_dir, transform, webp_dir, metadata_path, valid_indices, labels, is_fft=False):
        """
        Args:
            mrc_dir (str): Directory containing .mrc images.
            transform (callable, optional): Optional transform to be applied to the images.
            valid_indices (list): List of indices to filter the metadata.
        """
        self.mrc_dir = mrc_dir
        self.transform = transform
        self.is_fft = is_fft
        self.file_paths = []
        self.webp_dir = webp_dir
        self.valid_indices = valid_indices
        self.filtered_labels = np.array(labels)[valid_indices]
        
        # Use valid_indices to filter metadata
        self.filtered_metadata = pd.read_csv(metadata_path)[valid_indices]

        all_file_paths = glob(os.path.join(mrc_dir, '*.mrc'))
        self.filtered_file_paths = np.array(all_file_paths)[valid_indices]
        with mrcfile.open(self.filtered_file_paths[0], permissive=True) as mrc:
            self.pixel_size = mrc.voxel_size.x
        

    def __len__(self):
        return len(self.filtered_file_paths)

    def __getitem__(self, idx):
        # Load the .mrc file and convert to Fourier representation
        img_path = self.filtered_file_paths[idx]
        basename = os.path.basename(img_path)
        
        if self.is_fft:
            path_to_powerspectrum = os.path.join(self.webp_dir, basename.replace('.mrc', '_ctffit.webp'))
            powerspectrum = Image.open(path_to_powerspectrum)
            img = self.crop_and_mirror_image(powerspectrum)
        else:
            path_to_mg = os.path.join(self.webp_dir, basename.replace('.mrc', '.webp'))
            img = Image.open(path_to_mg)
            
        if self.transform:
            views = self.transform(img)
        else:
            views = img
        
        label = self.filtered_labels[idx]
        
        return views, label
    
    def crop_and_mirror_image(self, image1):
        image1_cropped = image1.crop((0, 0, image1.width, image1.height // 2))
        image1_flipped_lr = image1_cropped.transpose(Image.FLIP_LEFT_RIGHT)
        image1_flipped_tb = image1_flipped_lr.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Create a new image with combined width and the maximum height
        new_width = max(image1_cropped.size[0], image1_flipped_tb.size[0])
        new_height = image1_cropped.size[1] + image1_flipped_tb.size[1]

        # Create a new blank image (white background)
        new_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))

        # Paste the first image at the left side of the new image
        new_image.paste(image1_cropped, (0, 0))

        # Paste the second image at the bottom half of the new image
        new_image.paste(image1_flipped_tb, (0, image1_cropped.size[1]))
        
        return new_image
    
    def get_real_image(self, idx):
        img_path = self.filtered_file_paths[idx]
        basename = os.path.basename(img_path)
        path_to_mg = os.path.join(self.webp_dir, basename.replace('.mrc', '.webp'))
        mrc_data = Image.open(path_to_mg)
        return mrc_data
    
    def get_fft_image(self, idx):
        img_path = self.filtered_file_paths[idx]
        basename = os.path.basename(img_path)
        path_to_mg = os.path.join(self.webp_dir, basename.replace('.mrc', '_ctffit.webp'))
        ctf_img = Image.open(path_to_mg)
        fft_img = self.crop_and_mirror_image(ctf_img)
        return fft_img