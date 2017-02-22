import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys
from gaze import Gaze

class Raster:

    # This class needs an object to be imaged, and a Gaze object to move
    # it.

    def __init__(self,mosaic,subtense):

        self.mosaic = mosaic
        self.subtense = subtense
        self.sy,self.sx = mosaic.shape
        self.ymid,self.xmid = self.sy//2,self.sx//2
        
        
    def get(self,nx=512,ny=512,sx=0.5,sy=0.5,frame_rate=30.0,n_frames=1):

        mx0 = (self.sx-nx)//2-100
        my0 = (self.sy-ny)//2-100


        motion_free = self.mosaic[my0:my0+ny,mx0:mx0+nx]

        line_rate = float(frame_rate)*float(ny)
        dt = 1.0/line_rate
        
        g = Gaze(dt,drift_relaxation_rate=1e-3,drift_potential_slope=1.0,saccade_potential_slope=2.0,fractional_saccade_activation_threshold=np.inf,image=self.mosaic,image_subtense=self.subtense)

        for f in range(n_frames):

            frame = []
            
            if os.path.exists('./xtrace.npy') and False:
                gx = np.load('./xtrace.npy')
                gy = np.load('./ytrace.npy')
            else:
            
                for y in range(int(ny)-1):
                    #print y
                    g.step(True)

                gx,gy = g.history.xvec,g.history.yvec
                g.history.clear()
                np.save('./xtrace.npy',gx)
                np.save('./ytrace.npy',gy)
                
            for idx,(x,y) in enumerate(zip(gx,gy)):
                # now we have the fixation position,
                # we have to interpolate the image
                # from object coordinates into
                # these offset coordinates

                # first, convert x and y into pixels:
                xpx = x/self.subtense*self.sx+mx0
                ypx = y/self.subtense*self.sy+idx+my0

                x1 = np.floor(xpx)
                x2 = x1 + 1
                leftfrac = np.abs(xpx-x2)
                rightfrac = np.abs(xpx-x1)

                y1 = np.floor(ypx)
                y2 = y1 + 1
                topfrac = np.abs(ypx-y2)
                bottomfrac = np.abs(ypx-y1)

                topleft = self.mosaic[y1,x1:x1+nx]
                topright = self.mosaic[y1,x2:x2+nx]
                bottomleft = self.mosaic[y2,x1:x1+nx]
                bottomright = self.mosaic[y2,x2:x2+nx]

                line = leftfrac*topfrac*topleft + leftfrac*bottomfrac*bottomleft + rightfrac*topfrac*topright + rightfrac*bottomfrac*bottomright
                frame.append(line)

            frame = np.array(frame)
            plt.subplot(1,2,1)
            plt.cla()
            plt.imshow(motion_free,cmap='gray',interpolation='none')
            plt.subplot(1,2,2)
            plt.cla()
            plt.imshow(frame,cmap='gray',interpolation='none')
            plt.show()


if __name__=='__main__':

    im = np.load('./images/mosaic.npy')
    subtense = float(np.max(im.shape))/512.0*0.5
    
    r = Raster(im,subtense)
    r.get()
