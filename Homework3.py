import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.ndimage import median_filter

im1 = plt.imread('image68.tif')
im2 = median_filter(im1, size=3)
im4 = fftpack.fftshift(fftpack.fft2(im1))
#[(87, 118), (123, 120), (132, 136), (169, 138)]]
noise_points = [(x, y) for x, y in [(87, 118), (123, 120), (132, 136), (169, 138)]]
im5 = np.ones_like(im4)
for point in noise_points:
    x, y = point
    im5[x-5:x+6, y-5:y+6] = 0

im6 = im4 * im5
notch_filtered_image = np.abs(fftpack.ifft2(fftpack.ifftshift(im6)))


fig, axes = plt.subplots(2, 3, figsize=(12, 8))


axes[0, 0].imshow(im1, cmap='gray')
axes[0, 0].set_title('Noisy Image')

axes[0, 1].imshow(im2, cmap='gray')
axes[0, 1].set_title('Median Filtered Image')

axes[0, 2].imshow(notch_filtered_image, cmap='gray')
axes[0, 2].set_title('Notch Filtered Image')

axes[1, 0].imshow(np.log(1 + np.abs(im4)), cmap='gray')
axes[1, 0].set_title('Fourier Spectrum of Noisy Image')

axes[1, 1].imshow(np.log(1 + np.abs(im5)), cmap='gray')
axes[1, 1].set_title('Notch Filters')

axes[1, 2].imshow(np.log(1 + np.abs(im6)), cmap='gray')
axes[1, 2].set_title('Fourier Spectrum of Filtered Image')


plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()
