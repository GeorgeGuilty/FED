import cv2
import numpy as np

img = cv2.imread('Ex2.jpeg', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

u,v = np.meshgrid(range(dft.shape[1]),range(dft.shape[0]))
d = np.sqrt((u-dft.shape[1]/2)**2+(v-dft.shape[0]/2)**2)+1
alpha = 1
d0 = 200
n = 1
h = 1/(1+alpha*((d0/d)**(2*n)))
h = np.stack((h,h),axis=2)
fh = dft_shift*h

f_ishift = np.fft.ifftshift(fh)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

imgshow = (img_back-np.min(img_back))/(np.max(img_back)-np.min(img_back))
cv2.imshow('Highpass filter',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgshow = (img_back-np.min(img_back))/(np.max(img_back)-np.min(img_back))
cv2.imshow('Artifact Suppresion',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_blur = cv2.GaussianBlur(img_back,(3,3),0)*-1
img_supr = cv2.Laplacian(img_blur,cv2.CV_64FC1)

imgshow = ((img_supr-np.min(img_supr))/(np.max(img_supr)-np.min(img_supr))*255).astype(np.uint8)

v, c = np.unique(imgshow,return_counts=True)

thr = (v[np.argsort(c)[-1]]/255)*(np.max(img_supr)-np.min(img_supr))+np.min(img_supr)

gamma = 0.1

img_thr = np.where(img_supr >= thr, img_supr,0)

img_thr = np.where(img_supr >= gamma*img_supr.max(), 1,0)

imgshow = (img_thr-np.min(img_thr))/(np.max(img_thr)-np.min(img_thr))

cv2.imshow('Ringing Suppresion',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()