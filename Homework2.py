import numpy as np
import matplotlib.pyplot as plt


def read_bin(filename, shape):
    return np.fromfile(filename, dtype=np.uint8).resize(shape)


def stretch(image):
    return ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255


# P2(a)
X = read_bin('girl2bin.sec', (256 * 256))
XN = read_bin('girl2Noise32bin.sec', (256 * 256))
XNhi = read_bin('girl2Noise32Hibin.sec', (256 * 256))

xx = (XNhi - X) ** 2
MSE_Nhi = np.mean(xx)
print(f'MSE girl2Noise32Hi.bin: {MSE_Nhi}')

xx = (XN - X) ** 2
MSE_N = np.mean(xx)
print(f'MSE girl2Noise32.bin: {MSE_N}\n')
print(X.shape, X.dtype)
plt.figure(1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Original Tiffany Image', fontsize=18)

plt.figure(2)
plt.imshow(XNhi, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('girl2Noise32Hi', fontsize=18)

plt.figure(3)
plt.imshow(XN, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('girl2Noise32', fontsize=18)

# P2(b)
U_cutoff = 64
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HLtildeCenter = np.double(np.sqrt(U ** 2 + V ** 2) <= U_cutoff)
HLtilde = np.fft.fftshift(HLtildeCenter)

Y1 = np.fft.ifft2(np.fft.fft2(X) * HLtilde)
yy = (Y1 - X) ** 2
MSE_Y1 = np.mean(yy)
print(f'MSE: ideal LPF on girl2: {MSE_Y1}')

Y1Nhi = np.fft.ifft2(np.fft.fft2(XNhi) * HLtilde)
yy = (Y1Nhi - X) ** 2
MSE_Y1Nhi = np.mean(yy)
print(f'MSE: ideal LPF on Noise32Hi: {MSE_Y1Nhi}')
ISNR_Y1Nhi = 10 * np.log10(MSE_Nhi / MSE_Y1Nhi)
print(f'ISNR: ideal LPF on Noise32Hi: {ISNR_Y1Nhi} dB')

Y1N = np.fft.ifft2(np.fft.fft2(XN) * HLtilde)
yy = (Y1N - X) ** 2
MSE_Y1N = np.mean(yy)
print(f'MSE: ideal LPF on Noise32: {MSE_Y1N}')
ISNR_Y1N = 10 * np.log10(MSE_N / MSE_Y1N)
print(f'ISNR: ideal LPF on Noise32: {ISNR_Y1N} dB\n')

plt.figure(4)
plt.imshow(stretch(Y1), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('LPF on girl2', fontsize=18)

plt.figure(5)
plt.imshow(stretch(Y1Nhi), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('LPF on Noise32Hi', fontsize=18)

plt.figure(6)
plt.imshow(stretch(Y1N), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('LPF on Noise32', fontsize=18)

# P2(c)
U_cutoff_G = 64
SigmaG = 0.19 * 256 / U_cutoff_G
GtildeCenter = np.exp((-2 * np.pi ** 2 * SigmaG ** 2) / (256 ** 2) * (U ** 2 + V ** 2))
Gtilde = np.fft.fftshift(GtildeCenter)
G = np.fft.ifft2(Gtilde)
G2 = np.fft.fftshift(G)
ZPG2 = np.zeros((512, 512))
ZPG2[:256, :256] = G2

ZPX = np.zeros((512, 512))
ZPX[:256, :256] = X
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y2 = yy[128:384, 128:384]
yy = (Y2 - X) ** 2
MSE_Y2 = np.mean(yy)
print(f'MSE: Gaussian LPF on girl2: {MSE_Y2}')

ZPX[:256, :256] = XNhi
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y2Nhi = yy[128:384, 128:384]
yy = (Y2Nhi - X) ** 2
MSE_Y2Nhi = np.mean(yy)
print(f'MSE: Gaussian LPF on Noise32Hi: {MSE_Y2Nhi}')
ISNR_Y2Nhi = 10 * np.log10(MSE_Nhi / MSE_Y2Nhi)
print(f'ISNR: Gaussian LPF on Noise32Hi: {ISNR_Y2Nhi} dB')

ZPX[:256, :256] = XN
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y2N = yy[128:384, 128:384]
yy = (Y2N - X) ** 2
MSE_Y2N = np.mean(yy)
print(f'MSE: Gaussian LPF on Noise32: {MSE_Y2N}')
ISNR_Y2N = 10 * np.log10(MSE_N / MSE_Y2N)
print(f'ISNR: Gaussian LPF on Noise32: {ISNR_Y2N} dB\n')

plt.figure(7)
plt.imshow(stretch(Y2), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Gauss1 on girl2', fontsize=18)

plt.figure(8)
plt.imshow(stretch(Y2Nhi), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Gauss1 on Noise32Hi', fontsize=18)

plt.figure(9)
plt.imshow(stretch(Y2N), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Gauss1 on Noise32', fontsize=18)

# P2(d)
U_cutoff_G = 77.5
SigmaG = 0.19 * 256 / U_cutoff_G
GtildeCenter = np.exp((-2 * np.pi ** 2 * SigmaG ** 2) / (256 ** 2) * (U ** 2 + V ** 2))
Gtilde = np.fft.fftshift(GtildeCenter)
G = np.fft.ifft2(Gtilde)
G2 = np.fft.fftshift(G)
ZPG2 = np.zeros((512, 512))
ZPG2[:256, :256] = G2

ZPX[:256, :256] = X
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y3 = yy[128:384, 128:384]
yy = (Y3 - X) ** 2
MSE_Y3 = np.mean(yy)
print(f'MSE: Gaussian2 LPF on girl2: {MSE_Y3}')

ZPX[:256, :256] = XNhi
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y3Nhi = yy[128:384, 128:384]
yy = (Y3Nhi - X) ** 2
MSE_Y3Nhi = np.mean(yy)
print(f'MSE: Gaussian2 LPF on Noise32Hi: {MSE_Y3Nhi}')
ISNR_Y3Nhi = 10 * np.log10(MSE_Nhi / MSE_Y3Nhi)
print(f'ISNR: Gaussian2 LPF on Noise32Hi: {ISNR_Y3Nhi} dB')

ZPX[:256, :256] = XN
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y3N = yy[128:384, 128:384]
yy = (Y3N - X) ** 2
MSE_Y3N = np.mean(yy)
print(f'MSE: Gaussian2 LPF on Noise32: {MSE_Y3N}')
ISNR_Y3N = 10 * np.log10(MSE_N / MSE_Y3N)
print(f'ISNR: Gaussian2 LPF on Noise32: {ISNR_Y3N} dB\n')

plt.figure(10)
plt.imshow(stretch(Y3), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Gauss2 on girl2', fontsize=18)

plt.figure(11)
plt.imshow(stretch(Y3Nhi), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Gauss2 on Noise32Hi', fontsize=18)

plt.figure(12)
plt.imshow(stretch(Y3N), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Gauss2 on Noise32', fontsize=18)

plt.show()
