import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, denoise_tv_bregman, estimate_sigma)
from skimage import img_as_float
from skimage.util import random_noise

fig, field = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), sharex=True, sharey=True)
plt.gray()

noisy = plt.imread("NoiseReduction2.jpg")

denoised_wavelet = denoise_wavelet(noisy, rescale_sigma=True, multichannel=True)
denoised_tv_chambolle = denoise_tv_chambolle(noisy, weight=0.5, multichannel=True)
denoised_tv_bregman = denoise_tv_bregman(noisy, weight=0.5)

field[0,0].axis('off')
field[0,1].axis('off')
field[0,1].set_title('Noisy')
field[0,1].imshow(noisy)
field[0,2].axis('off')
field[1,0].axis('off')
field[1,0].set_title('Wavelet')
field[1,0].imshow(denoised_wavelet)
field[1,1].axis('off')
field[1,1].set_title('TV Chambolle')
field[1,1].imshow(denoised_tv_chambolle)
field[1,2].axis('off')
field[1,2].set_title('TV Bregman')
field[1,2].imshow(denoised_tv_bregman)

fig.tight_layout()

plt.show()