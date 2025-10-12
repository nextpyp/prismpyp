import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from numpy.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift

import time
def timeit(func):
    """
    A decorator to time the execution of the decorated function (e.g., forward).
    """
    def wrapper(self, *args, **kwargs):
        start_time = time.time()  # Record the start time
        
        # Execute the original function
        result = func(self, *args, **kwargs)
        
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        
        # Get the class name of the object calling the method
        class_name = self.__class__.__name__
        
        # Print the timing information with the class name and method name
        print(f"{class_name}.{func.__name__} took {elapsed_time:.4f} seconds")
        
        return result
    
    return wrapper

class FFT2Center(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # @timeit
    def forward(self, img):
        """
        Batched 2D discrete Fourier transform reordered with origin at center.
        Assumes input is a batch of images with shape [batch_size, channels, height, width].
        """
        # Perform FFT and fftshift in a batch-aware manner
        # Input should be of shape (batch_size, channels, height, width)
        if torch.cuda.is_available():
            img = img.to('cuda')

        # print("[FFT forward] img is on device:", img.device)
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img), dim=(-2, -1)), dim=(-2, -1))

class IFFT2Center(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # @timeit
    def forward(self, img):
        """
        Batched 2D inverse discrete Fourier transform with origin at center.
        Assumes input is a batch of images with shape [batch_size, channels, height, width].
        """
        if torch.cuda.is_available():
            img = img.to('cuda')

        # print("[IFFT forward] img is on device:", img.device)
        # Get the image dimensions for scaling factor (height, width)
        img_height, img_width = img.shape[-2], img.shape[-1]
        img_dim = img_height * img_width

        # Perform IFFT and ifftshift in a batch-aware manner
        fft_img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(img), dim=(-2, -1)), dim=(-2, -1))

        # Scale the result by the image size (batch operation)
        return fft_img.real / img_dim
    
class FFTNCenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        """N-dimensional discrete Fourier transform reordered with origin at center."""
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(img)))
    
class IFFTNCenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        """N-dimensional inverse discrete Fourier transform with origin at center."""
        return torch.abs(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(img))))
    
class HT2Center(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def fft2_center(self, img: torch.Tensor) -> torch.Tensor:
        """2-dimensional discrete Fourier transform reordered with origin at center."""
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))

    def forward(self, img):
        """2-dimensional discrete Hartley transform reordered with origin at center."""
        img = self.fft2_center(img)
        return img.real - img.imag

class HTNCenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def fftn_center(self, img: torch.Tensor) -> torch.Tensor:
        """N-dimensional discrete Fourier transform reordered with origin at center."""
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(img)))
    
    def forward(self, img):
        """N-dimensional discrete Hartley transform reordered with origin at center."""
        img = self.fftn_center(img)
        return img.real - img.imag

class IHT2Center(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def fft2_center(self, img: torch.Tensor) -> torch.Tensor:
        """2-dimensional discrete Fourier transform reordered with origin at center."""
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))

    def forward(self, img):
        """2-dimensional inverse discrete Hartley transform with origin at center."""
        img = self.fft2_center(img)
        img /= img.shape[-1] * img.shape[-2]
        return img.real - img.imag

class IHTNCenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def fftn_center(self, img: torch.Tensor) -> torch.Tensor:
        """N-dimensional discrete Fourier transform reordered with origin at center."""
        return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(img)))

    def forward(self, img):
        """N-dimensional inverse discrete Hartley transform with origin at center."""
        img = self.fftn_center(img)
        img /= torch.prod(torch.tensor(img.shape))
        return img.real - img.imag

class FakeCTFModulation(torch.nn.Module):
    def __init__(self, strength=0.3, frequency_range=(5, 20), random_phase=True):
        """
        Applies a fake, differentiable CTF-like modulation to an image.
        
        Parameters:
            strength (float): Amplitude of the modulation. Equivalent to amplitude contrast (0 --> no effect, 1 --> contrast inversion)
            frequency_range (tuple): (min_freq, max_freq) for sampling radial frequency Equivalent to min (low) and max (high) defocus.
            random_phase (bool): If True, randomly offsets the phase per call.
        """
        super().__init__()
        self.strength = strength
        self.frequency_range = frequency_range
        self.random_phase = random_phase

    def forward(self, img):
        """
        img: Float tensor of shape (C, H, W) or (B, C, H, W), values ∈ [0, 1].
        Returns: Tensor of same shape.
        """
        if img.dim() != 3:
            raise ValueError("Input must be a 3D tensor of shape (C, H, W)")

        C, H, W = img.shape
        device = img.device

        # Create radial distance map
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)  # (H, W)

        # Random frequency and phase
        freq = torch.empty(1).uniform_(*self.frequency_range).item()
        phase = torch.rand(1).item() * 2 * torch.pi if self.random_phase else 0.0

        # Build modulation mask
        modulation = 1.0 + self.strength * torch.sin(2 * torch.pi * freq * r + phase)  # (H, W)
        
        # Edge tapering to prevent artifacts (using cosine fade)
        fade = torch.cos(torch.clamp(r, 0, 1) * torch.pi / 2)  # Smooth fade at edges
        modulation *= fade  # Apply fade to modulation

        # Ensure modulation stays in a reasonable range (clip if necessary)
        modulation = torch.clamp(modulation, 0.0, 1.0)  # Clipping to [0, 1]

        # Expand modulation to all channels (C, H, W)
        modulation = modulation.unsqueeze(0).expand(C, -1, -1)

        # Apply modulation to image and clip output to [0, 1]
        modulated_img = img * modulation
        modulated_img = torch.clamp(modulated_img, 0.0, 1.0)  # Clip final result

        return modulated_img

class CTF2D(torch.nn.Module):
    def __init__(self, apix, defocus, voltage, amplitude_contrast, cs, oversample):
        super().__init__()
        self.apix = torch.tensor(apix, dtype=torch.float32)
        self.defocus = torch.tensor(defocus, dtype=torch.float32)
        self.voltage = torch.tensor(voltage, dtype=torch.float32)
        self.amplitude_contrast = torch.tensor(amplitude_contrast, dtype=torch.float32)
        self.cs = torch.tensor(cs, dtype=torch.float32)
        self.oversample = torch.tensor(oversample, dtype=torch.float32)

    def wave_length(self):
        """Return the wavelength for the given voltage."""
        return 12.2639 / torch.sqrt(self.voltage * 1000.0 + 0.97845 * self.voltage * self.voltage)

    def ctf_2d(self, img, df1):
        """
        Computes the CTF for a single image in Fourier space.
        img: Tensor of shape (3, H, W)
        """
        _, x_dim, y_dim = img.shape

        nyquist = 1.0 / (2 * self.apix)
        ds_x = nyquist / (x_dim // 2 * self.oversample)
        ds_y = nyquist / (y_dim // 2 * self.oversample)

        sx = torch.arange(-x_dim * self.oversample // 2, x_dim * self.oversample // 2,
                        device=img.device, dtype=torch.float32) * ds_x
        sy = torch.arange(-y_dim * self.oversample // 2, y_dim * self.oversample // 2,
                        device=img.device, dtype=torch.float32) * ds_y
        sx, sy = torch.meshgrid(sx, sy, indexing='ij')

        theta = -torch.atan2(sy, sx)
        s2 = sx ** 2 + sy ** 2
        defocus2d = df1 + 0.5 * torch.cos(2 * (theta - 0))  # astigmatism angle = 0

        wl = self.wave_length()
        phaseshift = torch.arcsin(self.amplitude_contrast / 100.0)
        gamma = 2 * torch.pi * (-0.5 * defocus2d * 1e4 * wl * s2 + 0.25 * self.cs * 1e7 * wl**3 * s2**2) - phaseshift

        env = torch.ones_like(gamma)
        ctf = torch.sin(gamma) * env  # shape: (H, W)
        return ctf

    def forward(self, img):
        """
        Applies CTF to a single image in Fourier space.
        img: Tensor of shape (3, H, W), already FFT-processed.
        """
        if torch.cuda.is_available():
            img = img.to('cuda')

        ctf = self.ctf_2d(img, self.defocus)  # (H, W)
        ctf = ctf.unsqueeze(0).expand(3, -1, -1)  # Match (3, H, W)

        return img * ctf

# Crop out a 45-degree quadrant from the top half of the FFT image
class FFTWedgeTransform(torch.nn.Module):  
    def __init__(self):
        super().__init__()
          
    def forward(self, img, quadrant):
        top_half_img = img[:img.shape[0] // 2, :]
        h, w = top_half_img.shape

        # Initialize mask to ones
        half_mask = np.ones((h, w))
        
        # Create meshgrid for all pixel coordinates (y, x)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        y_flipped = h - y  # Flipped y-coordinates
        
        # Conditions for each quadrant
        if quadrant == 1:
            half_mask[(x < w // 2) & (y_flipped < (-2 * h * x) / w + h)] = 0
        elif quadrant == 2:
            half_mask[(x < w // 2) & (y_flipped >= (-2 * h * x) / w + h) & (y_flipped < h)] = 0
        elif quadrant == 3:
            half_mask[(x >= w // 2) & (y_flipped >= (2 * h * x) / w - h) & (y_flipped < h)] = 0
        elif quadrant == 4:
            half_mask[(x >= w // 2) & (y_flipped >= 0) & (y_flipped < (2 * h * x) / w - h)] = 0

        # Create full mask by flipping half-mask left to right and top-down and stacking vertically
        mask = np.vstack([half_mask, np.flipud(np.fliplr(half_mask))])
        
        masked_img = img * mask
        return masked_img

# Mask out a ring with inner radius r1 and outer radius r2 from the FFT image    
class FFTBandstopTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img, r1, r2):
        # Dimension checks
        assert r1 < r2, 'r1 must be less than r2'
        assert r2 < min(img.shape) // 2, 'r2 must be less than half the minimum dimension of the image'
        
        bandstop_mask = np.ones(img.shape)
        center = (img.shape[0] // 2, img.shape[1] // 2)
        Y, X = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        bandstop_mask[(dist_from_center >= r1) & (dist_from_center <= r2)] = 0
        masked_img = img * bandstop_mask
        return masked_img

class LPFTransform(torch.nn.Module):
    def __init__(self, resolution, pixel_size, img_size):
        super().__init__()
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.img_size = img_size # Tuple (height, width)
    
    def resolution_to_pixel_rad(self):
        return self.resulution * self.img_size / self.pixel_size
    
    def butterworth_filter(self, dist, cutoff, order):
        return 1 / (1 + (dist / cutoff) ** (2 * order)) 
    
    def get_mask(self):
        center = self.img_size // 2
        Y, X = np.meshgrid(np.arange(self.img_size[0]), np.arange(self.img_size[1]), indexing='ij')
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        resolution_mask = np.ones(self.img_size)
        # resolution_mask[dist_from_center > resolution_to_pixel_rad(resolution, img, pixel_size)] = 1
        cutoff = self.resolution_to_pixel_rad()
        order = 2  # You can adjust the order of the filter
        resolution_mask = np.ones(self.img_size) - self.butterworth_filter(dist_from_center, cutoff, order)
        return resolution_mask
    
    def forward(self, img):
        resolution_mask = self.get_mask()
        img_fft = FFT2Center()(img)
        masked_img = img_fft * resolution_mask
        img_ifft = IFFT2Center()(masked_img)
        return img_ifft
    
class ZScoreNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        if torch.cuda.is_available():
            img = img.to('cuda')
            
        mean = img.mean()
        std = img.std()
        return (img - mean) / std
    
class Quantize(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    # @timeit    
    def forward(self, img):
        if torch.cuda.is_available():
            img = img.to('cuda')
            
        mean = img.mean()
        std = img.std()
        min_cutoff, max_cutoff = mean - 2 * std, mean + 2 * std
        
        # img_scaled = torch.clamp((img - min_cutoff) / (max_cutoff - min_cutoff), 0, 1)
        img_scaled = (img - min_cutoff) / (max_cutoff - min_cutoff)
        img_scaled = (img_scaled * 1).to(torch.float32)
        clamped = torch.clamp(img_scaled, 0, 1)
        # print("img_scaled min, max: ", img_scaled.min(), img_scaled.max())
        # print("clamped min, max: ", clamped.min(), clamped.max())
        return clamped
    
class GrayscaleToRGB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        if torch.cuda.is_available():
            img = img.to('cuda')
            
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.ndim == 4 and img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        return img