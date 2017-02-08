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

        line_rate = float(frame_rate)*float(ny)
        dt = 1.0/line_rate
        
        g = Gaze(dt,drift_relaxation_rate=1e-3,drift_potential_slope=1.0,saccade_potential_slope=2.0,fractional_saccade_activation_threshold=np.inf,image=self.mosaic,image_subtense=self.subtense)

        for f in range(n_frames):
            for y in range(int(ny)):
                #print y
                g.step(True)

            gx,gy = g.history.xvec,g.history.yvec
            g.history.clear()

            for x,y in zip(gx,gy):
                
        


if __name__=='__main__':

    im = np.load('./images/mosaic.npy')

    r = Raster(im,1.0)
    r.get()
