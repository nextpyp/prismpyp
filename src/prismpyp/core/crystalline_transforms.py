import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Sobel filter for edge detection
# Operates on tensors, so needs to come after the ToTensor transform
class SobelFilter(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        ## Sobel X and Y kernels (for edge detection)
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        sobel_x = sobel_x.to(device)
        sobel_y = sobel_y.to(device)

        # Expand to match 3 input channels → (out_channels=3, in_channels=3, kernel_size=3x3)
        self.sobel_x = nn.Parameter(sobel_x.expand(3, 3, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y.expand(3, 3, 3, 3), requires_grad=False)

    def forward(self, img):
        if torch.cuda.is_available():
            img = img.to('cuda')
        
        """ Applies Sobel filtering to RGB or batch of RGB images """
        has_batch = False
        if img.ndim == 3 and self.sobel_x.ndim == 4:  # Single RGB image, shape: (C, H, W)
            has_batch = False
            img = img.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
        
        grad_x = F.conv2d(img, self.sobel_x, padding=1, stride=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1, stride=1)
        sobel = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # Magnitude of gradients
        sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())  # Normalize to [0,1]
        if has_batch:
            return sobel
        else:
            ret = sobel.squeeze(0)
            return ret  # Remove batch dimension if necessary

# Contrast adjustment
# Operates on PIL images, so needs to come before the ToTensor transform
class LocalContrast(torch.nn.Module):
    def __init__(self):
        super(LocalContrast, self).__init__()
    
    def local_contrast(self, img):
        return transforms.functional.autocontrast(img)
    
    def forward(self, img):
        img = transforms.functional.equalize(img)
        img = self.local_contrast(img)
        return img

class HistogramEqualization(torch.nn.Module):
    def __init__(self):
        super(HistogramEqualization, self).__init__()
    
    def forward(self, img):
        if isinstance(img, Image.Image):  # Ensure the input is a PIL image
            return transforms.functional.equalize(img)
        return img
    
class HPF(torch.nn.Module):
    def __init__(self, pixel_size, cutoff, prob):
        super(HPF, self).__init__()
        self.pixel_size = pixel_size
        self.cutoff = cutoff
        self.prob = prob
    
    def pixel_to_frequency(self, img):
        if torch.cuda.is_available():
            img = img.to('cuda')
            
        if isinstance(img, Image.Image):
            img = transforms.functional.to_tensor(img)
        if img.ndimension() == 3:
            _, h, w = img.shape
        elif img.ndimension() == 2:
            h, w = img.shape

        f_h = torch.fft.fftfreq(h, d=self.pixel_size)
        f_w = torch.fft.fftfreq(w, d=self.pixel_size)
        return f_h, f_w

    # Create a high-pass filter mask based on resolution
    def create_hpf_mask(self, f_h, f_w):
        if torch.cuda.is_available():
            f_h = f_h.to('cuda')
            f_w = f_w.to('cuda')
        
        fx, fy = torch.meshgrid(f_h, f_w, indexing='ij')  # Create 2D frequency grids
        fx_shifted = torch.fft.fftshift(fx)  # Shift the zero frequency component to the center
        fy_shifted = torch.fft.fftshift(fy)

        # Compute the radial frequency (spatial frequency magnitude)
        freq_r = torch.sqrt(fx_shifted**2 + fy_shifted**2)  # Radial frequency map (1/Å)

        cutoff_freq = 1 / self.cutoff
        mask = freq_r >= cutoff_freq
        return mask
    
    def forward(self, img):
        was_pil = False # if the input was a PIL, make sure to also return a PIL
        if isinstance(img, Image.Image):
            was_pil = True
            img = transforms.functional.to_tensor(img)
            
        if torch.cuda.is_available():
            img = img.to('cuda')
            
        random_number = random.random()  # Generate a random number between 0 and 1
        if random_number > self.prob:
            if was_pil:
                img = transforms.functional.to_pil_image(img)
            return img
        
        f_h, f_w = self.pixel_to_frequency(img)
        hpf_mask = self.create_hpf_mask(f_h, f_w)
        
        # if isinstance(img, Image.Image):
        #     img = transforms.functional.to_tensor(img)
        
        if img.ndimension() == 3:
            hpf_mask = hpf_mask.unsqueeze(0).repeat(img.shape[0], 1, 1)
        masked = img * hpf_mask
        
        if was_pil:
            masked = transforms.functional.to_pil_image(masked)
        return masked

class SharpenTransform(torch.nn.Module):
    """Custom PyTorch transform for sharpening images."""
    def __init__(self, prob=0.5):
        super(SharpenTransform, self).__init__()
        self.prob = prob
        
    def forward(self, img):
        random_number = random.random()  # Generate a random number between 0 and 1
        if random_number > 1 - self.prob:
            return img
        
        return transforms.functional.adjust_sharpness(img, sharpness_factor=3.0)  # Adjust strength as needed

class CLAHETransform(torch.nn.Module):
    """Applies CLAHE to a PIL image."""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(5, 5), prob=0.5):
        super(CLAHETransform, self).__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.prob = prob
        
    def forward(self, img):
        random_number = random.random()  # Generate a random number between 0 and 1
        if random_number > 1 - self.prob:
            return img
        
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be a PIL image")

        # Convert PIL image to NumPy array
        img_np = np.array(img)

        if len(img_np.shape) == 2:  # Grayscale image
            img_np = self.clahe.apply(img_np)

        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB image
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)  # Convert to LAB
            img_np[:, :, 0] = self.clahe.apply(img_np[:, :, 0])  # Apply CLAHE to L channel
            img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)  # Convert back to RGB

        return Image.fromarray(img_np)
    
class RandomCLAHEOrSharpen(torch.nn.Module):
    def __init__(self, prob_clahe=0.3, prob_sharpen=0.3):
        super(RandomCLAHEOrSharpen, self).__init__()
        self.prob_clahe = prob_clahe
        self.prob_sharpen = prob_sharpen
        self.clahe = CLAHETransform(prob=1)
        self.sharpen = SharpenTransform(prob=1)
    
    def forward(self, img):
        random_number = random.random()  # Generate a random number between 0 and 1
        if random_number < self.prob_clahe:
            return self.clahe(img)
        elif random_number < self.prob_clahe + self.prob_sharpen:
            return self.sharpen(img)
        return img  # If neither transformation is applied, return the original image
    
class RandomRadialMask(nn.Module):
    def __init__(self, prob=0.5, mask_value=0.0):
        super().__init__()
        self.prob = prob
        self.mask_value = mask_value

    def forward(self, img):
        if random.random() > self.prob:
            return img

        c, h, w = img.shape
        assert c in (1, 3), f"Expected 1 or 3 channels, got {c}."

        device = img.device
        # Only generate coords once per call
        yy = torch.arange(h, device=device).view(-1, 1)
        xx = torch.arange(w, device=device).view(1, -1)
        center_y, center_x = h // 2, w // 2

        r = torch.sqrt((yy - center_y).float()**2 + (xx - center_x).float()**2)

        r_min = random.uniform(0.2, 0.7) * r.max()
        r_width = random.uniform(0.05, 0.2) * r.max()
        r_max = r_min + r_width

        mask = (r >= r_min) & (r <= r_max)  # (h, w) bool

        # Use inplace masked_fill_
        img = img.clone()  # safer to clone before in-place ops
        img[:, mask] = self.mask_value

        return img
    
class RandomRadialStretch(nn.Module):
    def __init__(self, prob=0.5, max_scale=0.2):
        super().__init__()
        self.prob = prob
        self.max_scale = max_scale

    def forward(self, img):
        if random.random() > self.prob:
            return img

        c, h, w = img.shape
        assert c in (1, 3), f"Expected 1 or 3 channels, got {c}."

        device = img.device
        yy = torch.arange(h, device=device).view(-1, 1)
        xx = torch.arange(w, device=device).view(1, -1)
        center_y, center_x = h // 2, w // 2

        dy = (yy - center_y).float()
        dx = (xx - center_x).float()

        r = torch.sqrt(dy**2 + dx**2)
        theta = torch.atan2(dy, dx)

        r_max = r.max()
        stretch_factor = random.uniform(1 - self.max_scale, 1 + self.max_scale)

        r_new = torch.clamp(r * stretch_factor, 0, r_max)

        x_new = center_x + r_new * torch.cos(theta)
        y_new = center_y + r_new * torch.sin(theta)

        # Normalize to [-1, 1]
        x_new = 2 * (x_new / (w - 1)) - 1
        y_new = 2 * (y_new / (h - 1)) - 1

        grid = torch.stack((x_new, y_new), dim=-1).unsqueeze(0)  # (1, h, w, 2)

        img = img.unsqueeze(0)  # (1, c, h, w)
        img_stretched = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        img_stretched = img_stretched.squeeze(0)

        return img_stretched
