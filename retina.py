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
    slope = -10
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.activation = np.inf
        self.index = Cone.index
        Cone.index = Cone.index + 1
        
    def step(self,retina,N=360):

        theta = np.linspace(0,np.pi*2,N)
        #xx = retina.xidx(self.x+np.cos(theta)*step_size)
        #yy = retina.yidx(self.y+np.sin(theta)*step_size)

        xx = np.clip(retina.xidx(self.x) + np.array([-1,0,1,-1,1,-1,0,1]),0,retina.N-1).astype(np.int)
        yy = np.clip(retina.yidx(self.y) + np.array([-1,-1,-1,0,0,1,1,1]),0,retina.N-1).astype(np.int)
        
        field = retina.field[xx,yy]
        #noise = np.mean(field)*np.random.randn(len(field))*.1
        winner = np.argmin(field)
        
        self.x = retina.xdeg(xx[winner])
        self.y = retina.ydeg(yy[winner])
        self.activation = field[winner]
        
        

class Retina:

    def __init__(self,x1=-0.5,x2=0.5,y1=-0.5,y2=0.5,N=1024):
        
        self.N = N
        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1

        self.xstep = float(self.dx)/float(N)
        self.ystep = float(self.dy)/float(N)
        
        self.XX,self.YY = np.meshgrid(np.arange(self.x1,self.x2,self.xstep),np.arange(self.y1,self.y2,self.ystep))

        self.cones = []
        self.age = 0
        self.clims = None
        self.field = np.zeros(self.XX.shape)
        
    def get_random_coordinates(self):
        return np.random.randint(self.N),np.random.randint(self.N)


    def xidx(self,x):
        return np.clip(np.round((x-self.x1)/self.dx*self.N).astype(np.int),0,self.N-1)
        
    def yidx(self,y):
        return np.clip(np.round((y-self.y1)/self.dy*self.N).astype(np.int),0,self.N-1)

    def xdeg(self,x):
        return self.XX[0,x]

    def ydeg(self,y):
        return self.YY[y,0]
        
    def add(self,x=None,y=None):
        if x is None or y is None or xidx is None or yidx is None:
            xidx,yidx = self.get_random_coordinates()
            x = self.XX[0,xidx]
            y = self.YY[yidx,0]
            
        print 'Adding cone at %0.1f,%0.1f'%(x,y)
        self.cones.append(Cone(x,y))

    def get_central_field(self):
        return np.sqrt(self.XX**2+self.YY**2)*0

    def get_cone_field(self):
        xx = self.XX - np.mean(self.XX)
        yy = self.YY - np.mean(self.YY)
        f = np.exp(Cone.slope*((xx**2)+(yy**2)))
        return f
                   
    def compute_total_field(self):
        centers = np.zeros(self.XX.shape)

        xidx = [self.xidx(c.x) for c in self.cones]
        yidx = [self.yidx(c.y) for c in self.cones]

        for x,y in zip(xidx,yidx):
            centers[y,x] = centers[y,x] + 1

        self.centers = centers
        
        cone_field = self.get_cone_field()

        # convolve cone centers with the single cone field to get the aggregate field
        def conv(a,b,k=2):
            sy,sx = a.shape
            # fft both
            af = np.fft.fft2(a,s=(sy*k,sx*k))
            bf = np.fft.fft2(b,s=(sy*k,sx*k))

            # block swap them and multiply
            #afs = np.fft.fftshift(af)
            #bfs = np.fft.fftshift(bf)
            #abf = afs*bfs
            
            # multiply
            abf = af*bf
            #abf = np.fft.ifftshift(abf)
            
            # inverse fft
            abfi = np.fft.ifft2(abf)
            #abfi = np.fft.ifftshift(abfi)
            return np.abs(abfi)

        field1 = conv(centers,cone_field)
        #field1 = np.abs(np.fft.ifftshift(np.fft.ifft2((np.fft.fft2(centers)*np.fft.fft2(cone_field)))))
        field2 = convolve2d(centers,cone_field,mode='same')

        
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(centers,interpolation='none')
        plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(cone_field,interpolation='none')
        plt.colorbar()
        plt.subplot(2,2,3)
        plt.imshow(field1,interpolation='none')
        plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow(field2,interpolation='none')
        plt.colorbar()
        plt.show()
        
        # add the central potential
        field = field + self.get_central_field()
        
        self.field = field

    def step(self,do_plot=True):
        self.compute_total_field()
        if do_plot:
            self.show()
        self.age = self.age + 1
        for c in self.cones:
            print c.index
            c.step(self)
        print 'Age: %d'%self.age

    def show(self):
        if self.clims is None:
            self.clims = (self.field.min()*1.1,self.field.max()*.5)
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(self.centers,extent=[self.XX.min(),self.XX.max(),self.YY.min(),self.YY.max()],interpolation='none')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(self.field,extent=[self.XX.min(),self.XX.max(),self.YY.min(),self.YY.max()],interpolation='none')
        plt.colorbar()
        plt.autoscale(False)
        cx = [c.x for c in self.cones]
        cy = [c.y for c in self.cones]
        plt.plot(cx,cy,'ko')
        for x,y in zip(cx,cy):
            print 'Cone at %0.3f,%0.3f'%(x,y)
        
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


    mini = True
    
    if mini:
        r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25,N=32)
        for k in range(3):
            r.add()

        plt.figure()
        for k in range(20):
            r.step()
            plt.show()
            plt.pause(5)

    else:
        r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25,N=1000)
        for k in range(4500):
            r.add()
            #r.show(xx,yy)
            #plt.pause(.00001)

        mov = GIF('retina.gif')
        f = plt.figure()
        for k in range(100):
            t0 = time()
            r.step()
            plt.pause(.001)
            mov.add(f)
            dt = time()-t0
            print dt
        mov.make()
        plt.show()
        r.save('foo')
