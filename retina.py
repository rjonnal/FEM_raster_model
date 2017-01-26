import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys
from time import time,sleep
from fig2gif import GIF
from octopod import H5

ACTIVATION_THRESHOLD = 20.0

class Cone:

    index = 0
    slope = -20000
    
    def __init__(self,x,y):
        self.x = int(x)
        self.y = int(y)
        self.activation = np.inf
        self.index = Cone.index
        Cone.index = Cone.index + 1

        #self.x_steps = np.array([-1,0,1,-1,0,1,-1,0,1]).astype(np.int)
        #self.y_steps = np.array([-1,-1,-1,0,0,0,1,1,1]).astype(np.int)

        rad = 1
        self.x_steps,self.y_steps = np.meshgrid(np.arange(-rad,rad+1),np.arange(-rad,rad+1))
        self.x_steps = self.x_steps.ravel()
        self.y_steps = self.y_steps.ravel()
        self.intensity = 20+np.random.randn()
        
    def step(self,retina):

        xx = self.x + self.x_steps
        yy = self.y + self.y_steps

        xx = xx%retina.N
        yy = yy%retina.N

        field = retina.field[yy,xx]
        #noise = np.mean(field)*np.random.randn(len(field))*.1
        winner = np.argmin(field)

        #plt.figure()
        #plt.imshow(retina.field,interpolation='none')
        
        #print xx
        #print yy
        #print field
        #print winner
        self.x = xx[winner]
        self.y = yy[winner]
        
        self.activation = field[winner]
        
        

class Retina:

    def __init__(self,x1=-0.5,x2=0.5,y1=-0.5,y2=0.5,N=1024,k=2):
        
        self.N = N
        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1

        self.xstep = float(self.dx)/float(N)
        self.ystep = float(self.dy)/float(N)
        
        #self.XX,self.YY = np.meshgrid(np.arange(self.x1,self.x2+self.xstep,self.xstep),np.arange(self.y1,self.y2+self.ystep,self.ystep))
        self.XX,self.YY = np.meshgrid(np.linspace(self.x1,self.x2,self.N),np.linspace(self.y1,self.y2,self.N))

        #for idx,x in enumerate(self.XX[0,:]):
        #    print idx,x
        #sys.exit()
        
        self.cones = []
        self.age = 0
        self.clims = None
        self.field = np.zeros(self.XX.shape)
        self.centers = np.zeros(self.XX.shape)
        self.center_intensities = np.zeros(self.XX.shape)
        self.intensity = np.zeros(self.XX.shape)
        self.k = k
        self.cone_field = self.get_cone_profile(Cone.slope)
        self.cone_profile = self.get_cone_profile(Cone.slope*3)
        
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
        return np.sqrt(self.XX**2+self.YY**2)*5

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
        if do_plot:
            self.show()
        self.age = self.age + 1
        for idx,c in enumerate(self.cones):
            c.step(self)
            if idx%5==0:
                self.compute_total_field()
        #print 'Age: %d'%self.age

    def show(self):
        if self.clims is None:
            self.clims = (self.field.min()*1.1,self.field.max()*.5)
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
        
    def save(self,tag):
        xfn = '%s_x.npy'%tag
        yfn = '%s_y.npy'%tag
        np.save(xfn,[c.x for c in self.cones])
        np.save(yfn,[c.y for c in self.cones])

        rec = [self.x1,self.x2,self.y1,self.y2]
        rfn = '%s_bounds.npy'%tag
        np.save(rfn,rec)

        
if __name__=='__main__':


    mini = False
    
    if mini:
        r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25,N=251)
        for k in range(100):
            r.add()

        plt.figure()
        for k in range(10):
            r.step()
            #plt.show()
            plt.pause(.0001)

    else:
        r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25,N=255)
        for k in range(4000):
            r.add()
            #r.show(xx,yy)
            #plt.pause(.00001)

        mov = GIF('retina.gif',fps=3)
        f = plt.figure(figsize=(24,12))
        for k in range(100):
            t0 = time()
            r.step()
            plt.pause(.001)
            mov.add(f)
            dt = time()-t0
            print dt
        mov.make()
        r.save('foo')
