import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys
from time import time,sleep
from fig2gif import GIF
from cone_density import ConeDensityInterpolator
from random import shuffle

class Cone:

    index = 0
    
    def __init__(self,x,y,intensity_mean=1000,intensity_std=50,rad=2):
        self.x = int(x)
        self.y = int(y)
        self.activation = np.inf
        self.index = Cone.index
        Cone.index = Cone.index + 1

        self.x_steps,self.y_steps = np.meshgrid(np.arange(-rad,rad+1),np.arange(-rad,rad+1))
        self.x_steps = self.x_steps.ravel()
        self.y_steps = self.y_steps.ravel()
        self.intensity = intensity_mean+intensity_std*np.random.randn()
        
    def step(self,retina,noisy=False):

        xx = self.x + self.x_steps
        yy = self.y + self.y_steps

        xx = xx%retina.N
        yy = yy%retina.N

        field = retina.field[yy,xx]
        if noisy:
            noise = np.sqrt(field)*np.random.randn(len(field))
            field = field + noise
        winner = np.argmin(field)

        self.x = xx[winner]
        self.y = yy[winner]
        self.activation = field[winner]
        
        

class Retina:

    def __init__(self,x1=-0.5,x2=0.5,y1=-0.5,y2=0.5,N=255,central_field_strength=0,integrity=0.01,potential_slope=-10000,intensity_slope=-25000,N_cones=0):
        
        self.N = N
        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1

        self.xstep = float(self.dx)/float(N)
        self.ystep = float(self.dy)/float(N)
        
        self.XX,self.YY = np.meshgrid(np.linspace(self.x1,self.x2,self.N),np.linspace(self.y1,self.y2,self.N))

        self.potential_slope = potential_slope
        self.intensity_slope = intensity_slope
       
        self.d = np.sqrt(self.XX**2+self.YY**2)
        self.cones = []
        self.age = 0
        self.field = np.zeros(self.XX.shape)
        self.centers = np.zeros(self.XX.shape)
        self.center_intensities = np.zeros(self.XX.shape)
        self.intensity = np.zeros(self.XX.shape)
        self.cone_field = self.get_cone_profile(self.potential_slope)
        self.cone_profile = self.get_cone_profile(self.intensity_slope)
        self.central_field_strength = central_field_strength
        self.integrity = integrity
        self.N_cones = N_cones

        for n in range(N_cones):
            self.add()
        
        
    def get_random_coordinates(self):
        return np.random.randint(self.N),np.random.randint(self.N)


    def xidx(self,x):
        # original: get stuck at edges:
        # np.clip(np.round((y-self.y1)/self.dy*self.N).astype(np.int),0,self.N-1)
        # new idea: just return the right indices and have the caller sort it out
        idx = (x-self.x1)/self.dx*self.N
        return idx
    
    def yidx(self,y):
        idx = (y-self.y1)/self.dy*self.N
        return idx
    
    def xdeg(self,x):
        return self.XX[0,x]

    def ydeg(self,y):
        return self.YY[y,0]
        
    def add(self,x=None,y=None):
        if x is None or y is None or xidx is None or yidx is None:
            x,y = self.get_random_coordinates()
            
        print 'Adding cone at %0.1f,%0.1f'%(x,y)
        self.cones.append(Cone(x,y))

    def get_central_field(self):
        return np.sqrt(self.XX**2+self.YY**2)*self.central_field_strength

    def get_cone_profile(self,slope):
        xx = self.XX - np.mean(self.XX)
        yy = self.YY - np.mean(self.YY)
        f = np.exp(slope*((xx**2)+(yy**2)))
        return f
                   
    def compute_total_field(self):
        self.centers = np.zeros(self.XX.shape)
        self.center_intensities = np.zeros(self.XX.shape)
        
        xidx = [c.x for c in self.cones]
        yidx = [c.y for c in self.cones]
        iidx = [c.intensity for c in self.cones]
        
        for x,y,I in zip(xidx,yidx,iidx):
            self.centers[y,x] = self.centers[y,x] + 1
            self.center_intensities[y,x] = self.center_intensities[y,x]+I
        
        # convolve cone centers with the single cone field to get the aggregate field
        def conv(a,b):
            
            sy,sx = a.shape
            # fft both after doubling size w/ zero-padding
            # this prevents circular convolution
            af = np.fft.fft2(a,s=(sy*2,sx*2))
            bf = np.fft.fft2(b,s=(sy*2,sx*2))

            # multiply
            abf = af*bf
            
            # inverse fft
            abfi = np.fft.ifft2(abf)

            # crop first (sy+1)//2-1 pixels because of zero-padding
            y1 = (sy+1)//2-1
            y2 = y1+sy
            x1 = (sx+1)//2-1
            x2 = x1+sx
            abfi = abfi[y1:y2,x1:x2]
            
            return np.abs(abfi)

        field = conv(self.centers,self.cone_field)
        self.intensity = conv(self.center_intensities,self.cone_profile)
        #field_sanity_check = convolve2d(centers,cone_field,mode='same')
        #assert(np.allclose(field,field_sanity_check))
        
        # add the central potential
        field = field + self.get_central_field()
        
        self.field = field

    def step(self,do_plot=True):
        shuffle(self.cones)
        if do_plot:
            self.show()
        self.age = self.age + 1
        for idx,c in enumerate(self.cones):
            c.step(self)
            if idx%(round(1.0/self.integrity))==0:
                self.compute_total_field()
        #print 'Age: %d'%self.age

    def show(self):
        plt.clf()
        plt.subplot(1,2,1)
        clim = np.percentile(self.intensity,(2,99))
        plt.imshow(self.intensity,extent=[self.XX.min(),self.XX.max(),self.YY.min(),self.YY.max()],interpolation='none',cmap='gray',clim=clim)
        plt.colorbar()
        plt.subplot(1,2,2)
        #plt.imshow(self.field,extent=[self.XX.min(),self.XX.max(),self.YY.min(),self.YY.max()],interpolation='none')
        plt.imshow(self.field,interpolation='none')
        plt.colorbar()
        plt.autoscale(False)
        cx = [c.x for c in self.cones]
        cy = [c.y for c in self.cones]
        plt.plot(cx,cy,'ko')
        #for x,y in zip(cx,cy):
            #print 'Cone at %0.3f,%0.3f'%(x,y)
        
        #plt.ylim((self.YY.max(),self.YY.min()))
        
    def save(self,tag=None):
        if tag is None:
            tag = self.tag()
        xfn = '%s_x.npy'%tag
        yfn = '%s_y.npy'%tag
        ifn = '%s_i.npy'%tag
        np.save(xfn,[c.x for c in self.cones])
        np.save(yfn,[c.y for c in self.cones])
        np.save(ifn,[c.intensity for c in self.cones])

        rec = [self.x1,self.x2,self.y1,self.y2]
        rfn = '%s_bounds.npy'%tag
        np.save(rfn,rec)

    def tag(self):
        out = '%0.2f_%0.2f_%0.2f_%0.2f_%04d_%0.3f_%0.2f_%06d'%(self.x1,self.x2,self.y1,self.y2,self.N,self.integrity,self.central_field_strength,self.N_cones)
        out = out.replace('-','m')
        return out
        
if __name__=='__main__':


    mini = False

    quick = {'x1':-.25,'x2':.25,'y1':-.25,'y2':.25,'N':51,'integrity':.1,'central_field_strength':5.0,'N_cones':200}
    full = {'x1':-.25,'x2':.25,'y1':-.25,'y2':.25,'N':251,'integrity':.01,'central_field_strength':2.0,'N_cones':2000}
    
    r = Retina(**full)
    tag = r.tag()

    mov = GIF('%s.gif'%tag,fps=3)
    f = plt.figure(figsize=(16,8))
    for k in range(50):
        t0 = time()
        r.step()
        plt.pause(.001)
        if k>0:
            mov.add(f)
        dt = time()-t0
        print dt
    mov.make()
    r.save(tag)
