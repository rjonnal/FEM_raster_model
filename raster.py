import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys
from gaze import Gaze
import time
from data_store import H5

class Raster:

    # This class needs an object to be imaged, and a Gaze object to move
    # it.

    def __init__(self,mosaic,subtense,nx=128,ny=128,frame_rate=30.0,h5fn='./test.hdf5',n_frames=1000,drift_speed=1.0):
        
        self.mosaic = mosaic
        self.subtense = subtense
        self.sy,self.sx = mosaic.shape
        self.ymid,self.xmid = self.sy//2,self.sx//2
        self.frame_rate = frame_rate
        self.nx = nx
        self.ny = ny
        self.n_frames = n_frames
        self.mx0 = (self.sx-nx)//2-150
        self.my0 = (self.sy-ny)//2-150
        self.motion_free = self.mosaic[self.my0:self.my0+self.ny,self.mx0:self.mx0+self.nx]

        self.line_rate = float(frame_rate)*float(ny)
        self.dt = 1.0/self.line_rate
        self.gaze = Gaze(self.dt,drift_relaxation_rate=1e-3,drift_potential_slope=2e5,saccade_potential_slope=2.0,fractional_saccade_activation_threshold=np.inf,drift_speed=drift_speed,image=self.mosaic,image_subtense=self.subtense)

        self.h5 = H5(h5fn)
        self.h5.put('/config/n_depth',1)
        self.h5.put('/config/n_fast',self.nx)
        self.h5.put('/config/n_slow',self.ny)
        self.h5.put('/config/n_vol',self.n_frames)
        self.h5.put('/projections/SLO',np.zeros((self.n_frames,self.ny,self.nx)))
        self.h5.put('/trace/x',np.zeros((self.n_frames,self.ny)))
        self.h5.put('/trace/y',np.zeros((self.n_frames,self.ny)))
        self.h5.put('/object/full',self.mosaic)
        self.h5.put('/object/motion_free',self.motion_free)
        self.h5.put('/object/full_subtense',self.subtense)
        self.h5.put('/object/full_mosaic',self.mosaic)

        fsy,fsx = self.mosaic.shape
        frame_subtense = float(fsx)/float(self.nx)*self.subtense
        self.h5.put('/config/subtense',frame_subtense)
        
    def get(self,n_frames=1,do_plot=False):

        t0 = time.time()
        frame = []

        if os.path.exists('./xtrace.npy') and False:
            gx = np.load('./xtrace.npy')
            gy = np.load('./ytrace.npy')
        else:

            for y in range(int(self.ny)):
                #print y
                self.gaze.step(True)

            gx,gy = self.gaze.history.xvec,self.gaze.history.yvec
            gx = gx[1:]
            gy = gy[1:]
            self.gaze.history.clear()
            np.save('./xtrace.npy',gx)
            np.save('./ytrace.npy',gy)

        for idx,(x,y) in enumerate(zip(gx,gy)):
            # now we have the fixation position,
            # we have to interpolate the image
            # from object coordinates into
            # these offset coordinates

            # first, convert x and y into pixels:
            xpx = x/self.subtense*self.sx+self.mx0
            ypx = y/self.subtense*self.sy+idx+self.my0

            x1 = np.floor(xpx)
            x2 = x1 + 1
            leftfrac = np.abs(xpx-x2)
            rightfrac = np.abs(xpx-x1)

            y1 = np.floor(ypx)
            y2 = y1 + 1
            topfrac = np.abs(ypx-y2)
            bottomfrac = np.abs(ypx-y1)

            topleft = self.mosaic[y1,x1:x1+self.nx]
            topright = self.mosaic[y1,x2:x2+self.nx]
            bottomleft = self.mosaic[y2,x1:x1+self.nx]
            bottomright = self.mosaic[y2,x2:x2+self.nx]

            line = leftfrac*topfrac*topleft + leftfrac*bottomfrac*bottomleft + rightfrac*topfrac*topright + rightfrac*bottomfrac*bottomright
            frame.append(line)

        frame = np.array(frame)
        if do_plot:
            plt.clf()
            plt.subplot(1,2,1)
            plt.cla()
            plt.imshow(self.motion_free,cmap='gray',interpolation='none')
            plt.subplot(1,2,2)
            plt.cla()
            plt.imshow(frame,cmap='gray',interpolation='none')
            plt.pause(.1)

        dt = time.time()-t0

        frame = (frame*1000).astype(np.uint16)
        return frame,gx,gy

    def run(self,do_plot=False):
        for k in range(self.n_frames):
            f,gx,gy = self.get(1,do_plot)
            self.h5['projections/SLO'][k,:,:] = f
            self.h5['trace/x'][k,:] = gx
            self.h5['trace/y'][k,:] = gy

if __name__=='__main__':

    obj_filename = './images/mosaic_2.npy'
    tag = os.path.split(obj_filename)[1].replace('.npy','')

    drift_speed = 1.0
    dsstr = '%0.1f'%drift_speed
    dsstr = dsstr.replace('.','p')
    
    outfn = 'phantom_%s_drift_%s.hdf5'%(tag,dsstr)
    outfn = os.path.join('/home/rjonnal/data/Dropbox/Share/ao_slo_data/Data/simulated',outfn)

    im = np.load('./images/mosaic_2.npy')
    subtense = float(np.max(im.shape))/512.0*1.0
    
    r = Raster(im,subtense,h5fn=outfn,drift_speed=drift_speed)
    r.run(False)
