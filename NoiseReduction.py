import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, denoise_tv_bregman, estimate_sigma)
from skimage import img_as_float
from skimage.util import random_noise

img = plt.imread("NoiseReduction.png")
img = img_as_float(img[100:250, 50:300])

fig, field = plt.subplots(nrows=4, ncols=3, figsize=(10, 7), sharex=True, sharey=True)
plt.gray()

sigma = 0.155
noisy = random_noise(img, var=sigma**2)

denoised_wavelet = denoise_wavelet(noisy, rescale_sigma=True, multichannel=True)
denoised_bilateral = denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15, multichannel=True)
denoised_tv_chambolle = denoise_tv_chambolle(noisy, weight=0.5, multichannel=True)
denoised_nl_means = denoise_nl_means(noisy, multichannel=True)
denoised_tv_bregman = denoise_tv_bregman(noisy, weight=0.5)

field[0,0].axis('off')
field[0,1].axis('off')
field[0,1].set_title('Original')
field[0,1].imshow(img)
field[0,2].axis('off')
field[1,0].axis('off')
field[1,1].axis('off')
field[1,1].set_title('Noisy')
field[1,1].imshow(noisy)
field[1,2].axis('off')
field[2,0].axis('off')
field[2,0].set_title('Wavelet')
field[2,0].imshow(denoised_wavelet)
field[2,1].axis('off')
field[2,1].set_title('Bilateral')
field[2,1].imshow(denoised_bilateral)
field[2,2].axis('off')
field[2,2].set_title('TV Chambolle')
field[2,2].imshow(denoised_tv_chambolle)
field[3,0].axis('off')
field[3,0].set_title('NL Means')
field[3,0].imshow(denoised_nl_means)
field[3,1].axis('off')
field[3,2].axis('off')
field[3,2].set_title('TV Bregman')
field[3,2].imshow(denoised_tv_bregman)

fig.tight_layout()

plt.show()