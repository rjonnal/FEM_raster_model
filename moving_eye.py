import numpy as np
from matplotlib import pyplot as plt
import os,sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Eye:

    def __init__(self,pixel_size=1e-6):
        self.pixel_size = pixel_size
        
    def step(self):
        pass

class Depression:
    A0 = 1.0 # initial amplitude
    relaxation_rate = 1.0e-2
    
    def __init__(self,x,y):
        self.age = 0.0
        self.x = x
        self.y = y
        self.A = self.A0
        
    def step(self):
        self.age = self.age + 1
        self.A = self.A * np.exp(-self.age*self.relaxation_rate)
        print self.A
        
    def evaluate(self,xx0,yy0):
        xx = xx0.copy()-self.x
        yy = yy0.copy()-self.y
        return -1.0/np.exp(np.sqrt(xx**2+yy**2)*10)*self.A

    def plot(self,ax,xx,yy):
        surf = self.evaluate(xx,yy)
        ax.cla()
        ax.plot_surface(xx, yy, surf, rstride=10, cstride=10, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim((-1,0))

class DepressionSet:

    ageout = 10
    
    def __init__(self):
        self.depressions = []
        self.age = 0
        
    def add(self,depression):
        self.depressions.append(depression)
        
    def step(self):
        self.age = self.age + 1
        for d in self.depressions:
            d.step()
            if d.age>self.ageout:
                self.depressions.remove(d)
            
    def evaluate(self,xx,yy):
        depression_sum = np.zeros(xx.shape)
        for d in self.depressions:
            depression_sum = depression_sum + d.evaluate(xx,yy)
        return depression_sum

    def plot(self,ax,xx,yy):
        surf = self.evaluate(xx,yy)
        ax.cla()
        ax.plot_surface(xx, yy, surf, rstride=10, cstride=10, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim((-1,0))


class Gaze:

    d_theta = np.pi/100.0
    step_size = 0.1
    
    def __init__(self,x0=0.0,y0=0.0,potential_strength=0.0):
        self.x = x0
        self.y = y0
        self.x0 = x0
        self.y0 = y0
        self.potential_strength = potential_strength
        self.x_path = []
        self.y_path = []
        self.landscape = DepressionSet()
        self.landscape.add(Depression(x0,y0))

    def plot_surface(self,ax,xx,yy):
        self.landscape.plot(ax,xx,yy)

    def plot(self):
        plt.cla()
        plt.plot(self.x_path,self.y_path,'ks')
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        
    def get_ring(self,r):
        thetas = np.arange(0,np.pi*2,self.d_theta)
        xx = np.cos(thetas)*r+self.x
        yy = np.sin(thetas)*r+self.y
        return xx,yy

    def step(self,power=1):

        # increment the landscape
        self.landscape.step()
        
        # implement a self-avoiding random walk
        # self-avoidance is implemented using a continuous
        # version of discrete swamp walk described by
        # Engbert et al., PNAS, 2011
        # [http://www.pnas.org/content/108/39/E765.full]

        # compute points along ring r distance from current location
        xx,yy = self.get_ring(self.step_size)

        # get the depths along this ring
        depths = self.landscape.evaluate(xx,yy)

        # raise the depths to positive values
        depths = depths - np.min(depths)

        # compute the potential term and add it to the depths
        potential = ((xx-self.x0)**2+(yy-self.y0)**2)
        potential = potential**10
        potential = potential/np.max(potential)
        potential = potential*self.potential_strength

        depths = depths + potential
        
        # compute a CDF using a power of the resulting positive depths
        weights = depths**power/np.sum(depths**power)
        csum = np.cumsum(weights)
        cdf = csum/np.max(csum)

        
        # search the weights for a random number
        test = np.random.rand()
        lower_bound = 0
        for idx,p in enumerate(cdf):
            if test>=lower_bound and test<=p:
                x = xx[idx]
                y = yy[idx]
                winner = idx
                break
            lower_bound = p
            
        #plt.figure()
        #plt.plot(weights)
        #plt.plot(winner,weights[winner],'go')
        #plt.show()

        # move to new location and add a depression there
        self.x = x
        self.y = y
        self.x_path.append(x)
        self.y_path.append(y)
        self.landscape.add(Depression(x,y))
        
if __name__=='__main__':


    XX,YY = np.meshgrid(np.arange(-1,1,.01),np.arange(-1,1,.01))
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    
    g = Gaze(potential_strength=1.0)
    while True:
        g.step()
        g.plot()
        #g.plot_surface(ax,XX,YY)
        plt.pause(.1)
        
    plt.show()
    sys.exit()
    
    x,y = g.get_ring(1.0)
    plt.plot(x,y,'ks')
    plt.show()
    sys.exit()
    ds = DepressionSet()
    ds.add(Depression(0.0,0.0))

    while True:
        if np.random.rand()>.9:
            x = (np.random.rand()-.5)*2.0
            y = (np.random.rand()-.5)*2.0
            ds.add(Depression(x,y))
            
        ds.step()
        ds.plot(ax,XX,YY)
        #ds.evaluate(XX,YY)
        
        
        plt.pause(.001)
    
    sys.exit()
    

    mo = MovingObject('./images/grass.npy')
    x = np.arange(0,300e-6,1e-6)
    y = 1.0/(1+np.exp(-(x-mo.drift_amplitude)*2e4))

    plt.plot(x,y)
    plt.show()
        
