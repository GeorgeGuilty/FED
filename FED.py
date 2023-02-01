import cv2
from matplotlib import pyplot as plt
import numpy as np

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
                t4 = np.argsort(tmf,axis=None)[(s**2-1)//2] >= s*s -s
                t5 = img[i+(s-1)//2,j+(s-1)//2]>=mu*tmf.max()
                if t1 and t2 and t3 and t4 and t5:
                    new_img[i+(s-1)//2,j+(s-1)//2] = 1
        img_e += new_img*img_e.max()
        imgshow = (img_e-np.min(img_e))/(np.max(img_e)-np.min(img_e))
        cv2.imshow('exact edge',imgshow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_e

img = cv2.imread('Ex1.png', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

u,v = np.meshgrid(range(dft.shape[1]),range(dft.shape[0]))
d = np.sqrt((u-dft.shape[1]/2)**2+(v-dft.shape[0]/2)**2)+1
alpha = 1
d0 = 20
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

gamma = 0.2
img_back = np.where(img_back >= gamma*img_back.max(), img_back,0)

imgshow = (img_back-np.min(img_back))/(np.max(img_back)-np.min(img_back))
cv2.imshow('Artifact Suppresion',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

# g,p = gradient(img_back)
# img_max = maximum(g, p)

kernel1 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
 
img_sup = (cv2.filter2D(src=img_back,ddepth=-1, kernel=kernel1)/9)-img_back

imgshow = (img_sup-np.min(img_sup))/(np.max(img_sup)-np.min(img_sup))
cv2.imshow('Maxima Suppresion',imgshow)
cv2.waitKey(0)
cv2.destroyAllWindows()

final = exactEdgePixel(img_back,img_sup,3,0.4)

final = (final-np.min(final))/(np.max(final)-np.min(final))

cv2.imshow('final',final)
cv2.waitKey(0)
cv2.destroyAllWindows()
