import numpy as np
from matplotlib import pyplot as plt
import os,sys
from time import time,sleep
from fig2gif import GIF
from octopod import H5

ACTIVATION_THRESHOLD = 20.0

class Cone:

    index = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.activation = np.inf
        self.index = Cone.index
        Cone.index = Cone.index + 1
        
    def field(self,xx,yy):

        # exponential
        f = np.exp(-100*((xx-self.x)**2+(yy-self.y)**2))
        
        # quadratic
        #f = -(xx-self.x)**2-(yy-self.y)**2
        # conical
        #f = -np.sqrt((xx-self.x)**2+(yy-self.y)**2)
        
        return f

    def show_field(self,xx,yy):
        plt.imshow(self.field(xx,yy))
        plt.colorbar()
        plt.show()

    def plot_field(self,x):
        plt.plot(self.field(x,np.ones(x.shape)*self.y))
        plt.show()

    def step(self,retina,step_size=.001,N=36):

        if self.activation>ACTIVATION_THRESHOLD:
            
            theta = np.linspace(0,np.pi*2,N)
            xx = self.x+np.cos(theta)*step_size
            yy = self.y+np.sin(theta)*step_size

            field = retina.compute_total_field(xx,yy)
            midx = np.argmin(field)
            self.x = xx[midx]
            self.y = yy[midx]
            self.activation = field[midx]

class Retina:

    def __init__(self,x1=-0.5,x2=0.5,y1=-0.5,y2=0.5,pixel_size=1e-2):
        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        self.xmid = (self.x1+self.x2)/2.0
        self.ymid = (self.y1+self.y2)/2.0
        self.dy = self.y2-self.y1
        self.dx = self.x2-self.x1
        
        self.pixel_size = pixel_size

        # central field
        self.cones = []
        self.age = 0
        self.clims = None

    def add(self):
        r = min(self.dy,self.dx)/2.0*.95
        x = np.random.rand()*self.dx+self.x1
        y = np.random.rand()*self.dy+self.y1
        print 'Adding cone at %0.1f,%0.1f'%(x,y)
        self.cones.append(Cone(x,y))

    def compute_central_field(self,xx,yy):
        return np.sqrt(xx**2+yy**2)*1


    def compute_total_field0(self,xx,yy):
        f = self.compute_central_field(xx,yy)
        for idx,c in enumerate(self.cones):
            f = f + c.field(xx,yy)
        return f

    def compute_total_field(self,xx,yy):
        
        f = self.compute_central_field(xx,yy)

        cx = np.array([c.x for c in self.cones])
        cy = np.array([c.y for c in self.cones])

        xstack = np.array([xx]*len(cx))
        ystack = np.array([yy]*len(cy))
        if len(xx.shape)==2:
            xstack = np.transpose(xstack,(1,2,0))
            ystack = np.transpose(ystack,(1,2,0))
        else:
            xstack = xstack.T
            ystack = ystack.T
            
        xstack = xstack - cx
        ystack = ystack - cy
        fstack = np.exp(-100*(xstack**2+ystack**2))
        f = np.sum(fstack,axis=len(fstack.shape)-1)
        return f

    def step(self):
        self.age = self.age + 1
        for c in self.cones:
            print c.index
            c.step(self)
        print 'Age: %d'%self.age

    def show(self,xx,yy):
        field = self.compute_total_field(xx,yy)
        if self.clims is None:
            self.clims = (field.min()*1.1,field.max()*.5)
        plt.clf()
        plt.imshow(field,extent=[xx.min(),xx.max(),yy.min(),yy.max()],interpolation='bilinear')
        plt.colorbar()
        plt.autoscale(False)
        cx = [c.x for c in self.cones]
        cy = [c.y for c in self.cones]
        plt.plot(cx,cy,'k.')

    def save(self,tag):
        xfn = '%s_x.npy'%tag
        yfn = '%s_y.npy'%tag
        np.save(xfn,[c.x for c in self.cones])
        np.save(yfn,[c.y for c in self.cones])

        rec = [self.x1,self.x2,self.y1,self.y2]
        rfn = '%s_bounds.npy'%tag
        np.save(rfn,rec)
        
if __name__=='__main__':

    r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25)
    pixel_size = 1e-2
    xx,yy = np.meshgrid(np.arange(r.x1,r.x2+pixel_size,pixel_size),np.arange(r.y1,r.y2+pixel_size,pixel_size))
    x = np.arange(r.x1,r.x2,pixel_size)
    # c = Cone(0,0)
    # c.plot_field(x)
    # c.show_field(xx,yy)
    # sys.exit()

    for k in range(4500):
        r.add()
        #r.show(xx,yy)
        #plt.pause(.00001)

    mov = GIF('retina.gif')
    f = plt.figure()
    for k in range(100):
        r.show(xx,yy)
        mov.add(f)
        plt.pause(.001)

        t0 = time()
        r.step()
        dt = time()-t0
        print dt
    mov.make()
    plt.show()
    r.save('foo')
