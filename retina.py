import numpy as np
from matplotlib import pyplot as plt
import os,sys

class Cone:

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def field(self,xx,yy):

        # exponential
        f = np.exp(-10000*((xx-self.x)**2+(yy-self.y)**2))
        
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

    def step(self,retina,step_size=1e-2,N=100):
        theta = np.linspace(0,np.pi*2,N)
        xx = self.x+np.cos(theta)*step_size
        yy = self.y+np.sin(theta)*step_size

        
        field = retina.compute_total_field(xx,yy)
        midx = np.argmin(field)
        #print self.x,self.y,'->',
        self.x = xx[midx]
        self.y = yy[midx]
        #print self.x,self.y


class Retina:

    def __init__(self,x1=-1.0,x2=1.0,y1=-1.0,y2=1.0,pixel_size=1e-2):
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

    def add(self):
        theta = np.random.rand()*2*np.pi
        r = min(self.dy,self.dx)/2.0*.95
        x = np.cos(theta)*r
        y = np.sin(theta)*r
        print 'Adding cone at %0.1f,%0.1f'%(x,y)
        self.cones.append(Cone(x,y))

    def compute_central_field(self,xx,yy):
        return np.sqrt(xx**2+yy**2)*10
            
    def compute_total_field(self,xx,yy):
        f = self.compute_central_field(xx,yy)
        for idx,c in enumerate(self.cones):
            f = f + c.field(xx,yy)
        return f

    def step(self):
        self.age = self.age + 1
        for c in self.cones:
            c.step(self)
        print 'Age: %d'%self.age

    def show(self,xx,yy):
        field = self.compute_total_field(xx,yy)
        plt.imshow(field,extent=[xx.min(),xx.max(),yy.min(),yy.max()])
        plt.colorbar()
        plt.autoscale(False)
        for c in self.cones:
            print c.x,c.y
            plt.plot(c.x,c.y,'ko')
        plt.show()
        
if __name__=='__main__':

    r = Retina()
    pixel_size = 1e-3
    xx,yy = np.meshgrid(np.arange(r.x1,r.x2,pixel_size),np.arange(r.y1,r.y2,pixel_size))
    x = np.arange(r.x1,r.x2,pixel_size)
    # c = Cone(0,0)
    # c.plot_field(x)
    # c.show_field(xx,yy)
    # sys.exit()

    for k in range(100):
        print k
        r.add()
        r.step()
    for k in range(100):
        r.step()    
    r.show(xx,yy)
