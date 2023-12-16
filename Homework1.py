import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav


def read_bin(filename, shape):
    return np.fromfile(filename, dtype=np.uint8).reshape(shape)


def stretch(image):
    return ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255


# P1(a): Apply a 7x7 average filter to the salesman image by doing linear convolution in the image domain.
X = read_bin('salesmanbin.sec', (256, 256))
plt.figure(1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Original Image', fontsize=18)

X2 = np.zeros((262, 262))
X2[3:259, 3:259] = X
Y2 = np.zeros((262, 262))

for row in range(3, 259):
    for col in range(3, 259):
        Y2[row, col] = np.sum(X2[row - 3:row + 4, col - 3:col + 4]) / 49

Y = stretch(Y2[3:259, 3:259])
plt.figure(2)
plt.imshow(Y, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Filtered Image', fontsize=18)

# Save the result image for comparison with later results
Y1a = Y.copy()

# P1(b): Use the method of Example 3 on page 5.61 of the Notes to do the same linear convolution by pointwise multiplication of DFT's.
H = np.zeros((128, 128))
H[61:69, 61:69] = 1 / 49

Padsize = 256 + 128 - 1
ZPX = np.zeros((Padsize, Padsize))
ZPX[:256, :256] = X

plt.figure(3)
plt.imshow(ZPX, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Zero Padded Image', fontsize=18)

ZPH = np.zeros((Padsize, Padsize))
ZPH[:128, :128] = H

plt.figure(4)
plt.imshow(stretch(ZPH), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Zero Padded Impulse Resp', fontsize=18)

ZPXtilde = np.fft.fft2(ZPX)
ZPHtilde = np.fft.fft2(ZPH)

ZPXtildeDisplay = stretch(np.log(1 + np.abs(np.fft.fftshift(ZPXtilde))))
plt.figure(5)
plt.imshow(ZPXtildeDisplay, cmap='gray')
plt.axis('image')
plt.axis('off')
plt.title('Log-magnitude spectrum of zero padded image', fontsize=12)

ZPHtildeDisplay = stretch(np.log(1 + np.abs(np.fft.fftshift(ZPHtilde))))
plt.figure(6)
plt.imshow(ZPHtildeDisplay, cmap='gray')
plt.axis('image')
plt.axis('off')
plt.title('Log-magnitude spectrum of zero padded H image', fontsize=12)

ZPYtilde = ZPXtilde * ZPHtilde
ZPY = np.fft.ifft2(ZPYtilde)

ZPYtildeDisplay = stretch(np.log(1 + np.abs(np.fft.fftshift(ZPYtilde))))
plt.figure(7)
plt.imshow(ZPYtildeDisplay, cmap='gray')
plt.axis('image')
plt.axis('off')
plt.title('Log-magnitude spectrum of zero padded result', fontsize=12)

plt.figure(8)
plt.imshow(stretch(ZPY.real), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Zero Padded Result', fontsize=18)

Y = stretch(ZPY[64:319, 64:319].real)
plt.figure(9)
plt.imshow(Y, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Final Filtered Image', fontsize=18)

# Compare this result image with the one from part (a)


# P1(c): Use the method of Example 5 on page 5.76 of the Notes to do the same linear convolution again by pointwise multiplication of DFT's, this time using a 256x256 true zero-phase impulse response.
H = np.zeros((256, 256))
H[125:133, 125:133] = 1 / 49

H2 = np.fft.fftshift(H)
plt.figure(10)
plt.imshow(stretch(H2), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Zero Phase Impulse Resp', fontsize=18)

ZPX = np.zeros((512, 512))
ZPX[:256, :256] = X

ZPH2 = np.zeros((256, 256))
ZPH2[:128, :128] = H2[:128, :128]
ZPH2[:128, 128:256] = H2[:128, :128]  # Corrected line
ZPH2[128:256, :128] = H2[:128, :128]  # Corrected line
ZPH2[128:256, 128:256] = H2[:128, :128]  # Corrected line

plt.figure(11)
plt.imshow(stretch(ZPH2), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Zero Padded zero-phase H', fontsize=18)

# Y = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPH2))
# Y = stretch(Y[:256, :256].real)
# plt.figure(12)
# plt.imshow(Y, cmap='gray', vmin=0, vmax=255)
# plt.axis('image')
# plt.axis('off')
# plt.title('Final Filtered Image', fontsize=18)
# plt.savefig('MY1c.eps')

# Compare this result image with the one from part (a)
print('(c): max difference from part (a):', np.max(np.abs(Y - Y1a[:255, :255])))
plt.show()
