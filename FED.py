import cv2
import numpy as np

def FED(img, d0 = 1000,m = 5,g=0.05,bks1=13,bks2=3,cc=-1):
    if len(img.shape) > 2 or len(img.shape) == 0:
        raise ValueError('Only grayscale images are allowed')
    # Import Image and lil blur
    img = cv2.GaussianBlur(img,(3,3),0)
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
    # Remove noise and apply blur for wider edges
    img_thr = np.where(img_back >= g*img_back.max(), img_back,0)
    img_blur = cv2.GaussianBlur(img_thr,(bks1,bks1),0)
    # Apply threshold and final image is delivered
    img_thr = ((img_blur-np.min(img_blur))/(np.max(img_blur)-np.min(img_blur))*255).astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img_thr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bks2,cc)
    return img_thr

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
        cv2.imshow('frame',  FED(gray))
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()