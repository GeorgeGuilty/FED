import cv2
import numpy as np

def FED(img, d0 = 1000,m = 7,sk = 5,g = 0.3,bks1 = 3,bks2=3, bks3=3,cc=-15,mk=2):
    if len(img.shape) > 2 or len(img.shape) == 0:
        raise ValueError('Only grayscale images are allowed')
    # Blur and sharp image to clean and prepare for a better edge detection
    img = cv2.GaussianBlur(img,(bks1,bks1),0)
    k = -np.ones((sk,sk))
    k[sk//2,sk//2] = sk*sk
    img = cv2.filter2D(img,-1,k)
    # DFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # Apply highpass filter to DFT
    u,v = np.meshgrid(range(dft.shape[1]),range(dft.shape[0]))
    d = np.sqrt((u-dft.shape[1]/2)**2+(v-dft.shape[0]/2)**2)+1
    h = 1/(1+((d0/d)**(2)))
    h = np.stack((h,h),axis=2)
    fh = dft_shift*h
    # IDFT plus margin to remove noise on the picture frame
    f_ishift = np.fft.ifftshift(fh)
    img_back = cv2.idft(f_ishift)
    margin = np.ones_like(img)
    margin[0:m,:] = 0
    margin[-1:(-1-m):-1,:] = 0
    margin[:,0:m] = 0
    margin[:,-1:(-1-m):-1] = 0
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])*margin
    # Apply first threshold and gaussian blur to join edges and blur noise and join both
    img_thr = np.where(img_back >= g*img_back.max(),img_back,0)
    img_blur = cv2.GaussianBlur(img_thr,(bks2,bks2),0)
    img_mul = (img_thr**0.5)*img_blur
    # Apply second threshold with adaptive gaussian and final image is delivered
    img_uint = ((img_mul-np.min(img_mul))/(np.max(img_mul)-np.min(img_mul))*255).astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img_uint,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bks3,cc)
    img_mcl = cv2.morphologyEx(img_thr,cv2.MORPH_CLOSE,np.ones((mk,mk)))
    return img_mcl

if __name__ == '__main__':
    # img = cv2.imread('Ex1.png',0)
    # cv2.imshow('FED', FED(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    vid = cv2.VideoCapture(0)
  
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the resulting frame
        cv2.imshow('frame',  frame)
        cv2.imshow('FED',  FED(gray))
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()