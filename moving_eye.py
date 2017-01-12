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
    steepness = 10
    
    def __init__(self,x,y):
        self.age = 0.0
        self.x = x
        self.y = y
        self.A = self.A0
        
    def step(self):
        self.age = self.age + 1
        self.A = self.A * np.exp(-self.age*self.relaxation_rate)
        
    def evaluate(self,xx0,yy0):
        xx = xx0.copy()-self.x
        yy = yy0.copy()-self.y
        return -1.0/np.exp(np.sqrt(xx**2+yy**2)*self.steepness)*self.A

    def plot3d(self,ax,xx,yy):
        surf = self.evaluate(xx,yy)
        ax.cla()
        ax.plot_surface(xx, yy, surf, rstride=10, cstride=10, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim((-1,0))

    def plot2d(self,xx,yy):
        plt.cla()
        surf = self.evaluate(xx,yy)
        plt.imshow(surf)

class ConstantPeak(Depression):

    def __init__(self,x,y,h_factor=1.0,v_factor=1.0):
        Depression.__init__(self,x,y)
        self.h_factor = h_factor
        self.v_factor = v_factor

    def evaluate(self,xx0,yy0):
        xx = xx0.copy()-self.x
        yy = yy0.copy()-self.y
        surf = self.A0-((xx*self.h_factor)**2+(yy*self.v_factor)**2)
        return surf
        
    def step(self):
        self.age = self.age + 1
        self.A = self.A0


class DepressionSet:

    Aepsilon = .01
    def __init__(self):
        self.depressions = []
        self.age = 0
        
    def add(self,depression):
        self.depressions.append(depression)
        
    def step(self):
        self.age = self.age + 1
        for d in self.depressions:
            d.step()
            print '.',
            # remove depressions with negligible amplitudes
            if d.A<self.Aepsilon:
                self.depressions.remove(d)
        print
        
    def evaluate(self,xx,yy):
        depression_sum = np.zeros(xx.shape)
        for d in self.depressions:
            depression_sum = depression_sum + d.evaluate(xx,yy)
        return depression_sum

    def plot3d(self,ax,xx,yy):
        surf = self.evaluate(xx,yy)
        ax.cla()
        ax.plot_surface(xx, yy, surf, rstride=10, cstride=10, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim((-1,0))

    def plot2d(self,xx,yy):
        plt.cla()
        surf = self.evaluate(xx,yy)
        plt.imshow(surf)
    
        
class Gaze:

    d_theta = np.pi/100.0
    step_size = 0.1
    
    def __init__(self,x0=0.0,y0=0.0,potential_strength=0.0,fixation_quality=1.0):
        self.x = x0
        self.y = y0
        self.x0 = x0
        self.y0 = y0
        self.potential_strength = potential_strength
        self.x_path = []
        self.y_path = []
        self.landscape = DepressionSet()
        self.landscape.add(ConstantPeak(x0,y0,v_factor=1.0*fixation_quality,h_factor=1.0*fixation_quality))
        self.landscape.add(Depression(x0,y0))

    def plot_surface(self,ax,xx,yy):
        self.landscape.plot(ax,xx,yy)

    def plot(self,xlims=(-2,2),ylims=(-2,2),N=128):
        plt.subplot(1,2,1)
        plt.cla()
        xx,yy = np.meshgrid(np.linspace(xlims[0],xlims[1],N),np.linspace(ylims[0],ylims[1],N))
        self.landscape.plot2d(xx,yy)
        plt.subplot(1,2,2)
        plt.cla()
        plt.plot(self.x_path,self.y_path,'k.')
        plt.xlim(xlims)
        plt.ylim(ylims)
        
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

        mode='stochastic'

        if mode=='stochastic':
            # compute a CDF using a power of the resulting positive depths
            # check for corner case where depths are all zero, and assign
            # a same-size vector of ones, i.e. all directions equally probable
            if any(depths):
                weights = depths**power/np.sum(depths**power)
            else:
                weights = np.ones(depths.shape)
            csum = np.cumsum(weights)
            cdf = csum/np.max(csum)

            # search the weights for a random number
            test = np.random.rand()
            lower_bound = 0
            for idx,p in enumerate(cdf):
                if test>=lower_bound and test<=p:
                    winner = idx
                    break
                lower_bound = p

        if mode=='deterministic':
            weights = depths
            winners = np.where(depths==np.max(depths))[0]
            if len(winners)>1:
                winner = winners[np.random.randint(len(winners))]
            else:
                winner = winners[0]
                               
            
        x = xx[winner]
        y = yy[winner]
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


    XX,YY = np.meshgrid(np.arange(-3,3,.01),np.arange(-3,3,.01))
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    
    g = Gaze(potential_strength=1.0,fixation_quality=0.5)
    while True:
        g.step()
        #g.plot_surface(ax,XX,YY)
        if g.landscape.age%1==0:
            g.plot()
            plt.pause(.00000001)
        
