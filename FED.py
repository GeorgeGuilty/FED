import cv2
from matplotlib import pyplot as plt
import numpy as np

def gradient(im):
    # Sobel operator
    op1 = np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
    op2 = np.array([[-1, -2, -1],
                 [ 0,  0,  0],
                 [ 1,  2,  1]])
    kernel1 = np.zeros(im.shape)
    kernel1[:op1.shape[0], :op1.shape[1]] = op1
    kernel1 = np.fft.fft2(kernel1)

    kernel2 = np.zeros(im.shape)
    kernel2[:op2.shape[0], :op2.shape[1]] = op2
    kernel2 = np.fft.fft2(kernel2)

    fim = np.fft.fft2(im)
    Gx = np.real(np.fft.ifft2(kernel1 * fim)).astype(float)
    Gy = np.real(np.fft.ifft2(kernel2 * fim)).astype(float)

    G = np.sqrt(Gx**2 + Gy**2)
    Theta = np.arctan2(Gy, Gx) * 180 / np.pi
    return G, Theta


def maximum(det, phase):
  gmax = np.zeros(det.shape)
  for i in range(gmax.shape[0]):
    for j in range(gmax.shape[1]):
      if phase[i][j] < 0:
        phase[i][j] += 360

      if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
        # 0 degrees
        if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
          if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
            gmax[i][j] = det[i][j]
        # 45 degrees
        if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
          if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
            gmax[i][j] = det[i][j]
        # 90 degrees
        if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
          if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
            gmax[i][j] = det[i][j]
        # 135 degrees
        if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
          if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
            gmax[i][j] = det[i][j]
  return gmax

def exactEdgePixel(img,img_max,s,mu):
    if s % 2 != 1 and s < 3:
        raise Exception('s should be an odd matrix size more than 3x3')

    img_e = img_max
    new_img = img_e

    while np.sum(new_img) > 0:
        new_img = np.zeros(img_e.shape,dtype= 'int32')
        for i in range(img.shape[0]-s+1):
            for j in range(img.shape[1]-s+1):
                tmE = img_e[i:(i+s),j:(j+s)]
                tmf = img[i:(i+s),j:(j+s)]
                t1 = 0 == img_e[i+(s-1)//2,j+(s-1)//2]
                t2 = np.sum(np.where(tmE >0,1,0)) >= 1
                t3 = np.sum(np.where(tmE >0,1,0)) <= s
                t4 = np.argsort(tmf,axis=None)[(s**2-1)//2] > s*s -s
                t5 = img[i+(s-1)//2,j+(s-1)//2]>=mu*tmf.max()
                if t1 and t2 and t3 and t4 and t5:
                    new_img[i+(s-1)//2,j+(s-1)//2] = 1
        img_e += new_img*img_e.max()
    return img_e

img = cv2.imread('Ex4.png', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

u,v = np.meshgrid(range(dft.shape[1]),range(dft.shape[0]))
d = np.sqrt((u-dft.shape[1]/2)**2+(v-dft.shape[0]/2)**2)+1
alpha = 1
d0 = 70
n = 2
h = 1/(1+alpha*((d0/d)**(2*n)))
h = np.stack((h,h),axis=2)
fh = dft_shift*h

f_ishift = np.fft.ifftshift(fh)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
gamma = 0.2
img_back = np.where(img_back >= gamma*img_back.max(), img_back,0)
g,p = gradient(img_back)
img_max = maximum(g, p)

img_back = np.where(img_back > 0, 1,0)
img_max = np.where(img_back > 0, 1,0)

final = exactEdgePixel(img_back,img_max,3,0)

final = (final-np.min(final))/(np.max(final)-np.min(final))

cv2.imshow('final',final)
cv2.waitKey(0)
cv2.destroyAllWindows()
