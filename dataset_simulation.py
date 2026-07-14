import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


# depth: from [0,255] to [36,200]
def process_depth(d):
    d = np.abs(d - 255)
    d = np.round(d / 255. * 164 + 36)
    return d


# depth: from continuous to discrete
def depth_redistribution(depth):
    d_range = np.array([35, 37, 41, 45, 49, 55, 62, 71, 83, 100, 124, 164, 200])
    d = np.zeros_like(depth)
    for i in range(len(d_range) - 1):
        d_m = np.where((depth <= d_range[i+1]) & (depth > d_range[i]), 1., 0.)
        d = d_m * i + d
    return d


# simulate blur images
def make_blur_image(rgb, depth, psf, padding, device):
    D, C, H, W = psf.shape
    blur = np.zeros_like(psf[0])
    depth = depth_redistribution(depth)

    for d in range(D):
        depth_mask = np.where(depth == d, 1., 0.)
        if np.sum(depth_mask) > 0.:
            rgb_s = rgb * depth_mask
            img = np.pad(rgb_s, padding, mode='constant')  # 3,480,480
            blur_s = np.abs(ifftshift(ifft2(fft2(img) * fft2(psf[d])), axes=[-2, -1]))
            blur += blur_s
    blur = blur / np.max(blur)

    shot_noise = np.random.poisson(blur * 30144)
    read_noise = np.random.normal(loc=0., scale=63.6, size=blur.shape)
    blur_e = np.clip(shot_noise + read_noise, 0., 30144)
    blur_n = np.round(blur_e / 30144 * 255.) / 255
    return blur_n