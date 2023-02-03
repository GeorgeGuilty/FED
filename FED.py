import cv2
import numpy as np

img = cv2.imread('Ex1.png', 0)
img = cv2.GaussianBlur(img,(7,7),0)
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

imgshow = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])+1)
imgshow = (imgshow-np.min(imgshow))/(np.max(imgshow)-np.min(imgshow))
cv2.imshow('DFT',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

u,v = np.meshgrid(range(dft.shape[1]),range(dft.shape[0]))
d = np.sqrt((u-dft.shape[1]/2)**2+(v-dft.shape[0]/2)**2)+1
alpha = 1
d0 = 1000
n = 1
h = 1/(1+alpha*((d0/d)**(2*n)))
h = np.stack((h,h),axis=2)
fh = dft_shift*h

imgshow = np.log(cv2.magnitude(fh[:, :, 0], fh[:, :, 1])+1)
imgshow = (imgshow-np.min(imgshow))/(np.max(imgshow)-np.min(imgshow))
cv2.imshow('DFT with highpass filter',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

f_ishift = np.fft.ifftshift(fh)
img_back = cv2.idft(f_ishift)
margin = np.ones_like(img)
margin[0:2,:] = 0
margin[-1:-3:-1,:] = 0
margin[:,0:2] = 0
margin[:,-1:-3:-1] = 0
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])*margin

imgshow = (img_back-np.min(img_back))/(np.max(img_back)-np.min(img_back))
cv2.imshow('Highpass filter',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_blur = img_back
img_blur = cv2.GaussianBlur(img_blur,(15,15),0)

imgshow = (img_blur-np.min(img_blur))/(np.max(img_blur)-np.min(img_blur))
cv2.imshow('Artifact Suppresion by Gaussian Blur',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_blur *= -1

img_supr = cv2.Laplacian(img_blur,cv2.CV_64FC1)

imgshow = ((img_supr-np.min(img_supr))/(np.max(img_supr)-np.min(img_supr))*255).astype(np.uint8)

v, c = np.unique(imgshow,return_counts=True)

thr = (v[np.argsort(c)[-1]]/255)*(np.max(img_supr)-np.min(img_supr))+np.min(img_supr)

img_thr = np.where(img_supr >= thr, img_supr,0)

imgshow = (img_thr-np.min(img_thr))/(np.max(img_thr)-np.min(img_thr))
cv2.imshow('Laplacian filter',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

gamma = 0.05

img_thr = np.where(img_supr >= gamma*img_supr.max(), 255,0).astype(np.uint8)

imgshow = (img_thr-np.min(img_thr))/(np.max(img_thr)-np.min(img_thr))
cv2.imshow('Thresholding 2',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img_thr)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
min_size = 30
im_result = np.zeros_like(img_thr)
for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        im_result[im_with_separated_blobs == blob + 1] = 255

cv2.imshow('Removing small blobs',im_result)
cv2.waitKey(0)
cv2.destroyAllWindows()