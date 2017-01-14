import numpy as np
from matplotlib import pyplot as plt
import os,sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Feature:
    A0 = 1.0 # initial amplitude
    relaxation_rate = 1.0e-1
    steepness = 10
    
    def __init__(self,x,y):
        self.age = 0.0
        self.x = x
        self.y = y
        self.A = self.A0
        
    def step(self):
        self.age = self.age + 1
        self.A = (1.0-self.relaxation_rate)*self.A
        
    def evaluate(self,xx0,yy0):
        xx = xx0.copy()-self.x
        yy = yy0.copy()-self.y
        return 1.0/np.exp(np.sqrt(xx**2+yy**2)*self.steepness)*self.A

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

class ConicalPotential(Feature):

    # a conical potential with a peak at the center
    # unlike footprints, the potential doesn't relax
    # in its step() function
    
    def __init__(self,x,y,L=1.0):
        Feature.__init__(self,x,y)
        self.L = L
        self.A = 1.0
        
    def evaluate(self,xx0,yy0):
        xx = xx0.copy()-self.x
        yy = yy0.copy()-self.y
        surf = self.L*(xx**2+yy**2)
        return surf
        
    def step(self):
        self.age = self.age + 1
        self.A = 1.0

class FeatureSet:

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
            # remove depressions with negligible amplitudes
            if d.A<self.Aepsilon:#+np.random.randn()*self.Aepsilon*.1:
                self.depressions.remove(d)
        
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
    saccade_threshold = 3.0
    
    def __init__(self,x0=0.0,y0=0.0,drift_potential_slope=1.0,image=None):
        self.x = x0
        self.y = y0
        self.x0 = x0
        self.y0 = y0
        self.drift_potential_slope = drift_potential_slope
        self.x_path = []
        self.y_path = []
        self.landscape = FeatureSet()
        self.landscape.add(ConicalPotential(x0,y0,L=drift_potential_slope))
        self.landscape.add(Feature(x0,y0))
        if image is None:
            self.image = np.zeros((100,100))
        else:
            self.image = image
        
    def plot_surface(self,ax,xx,yy):
        self.landscape.plot(ax,xx,yy)

    def plot(self,xlims=(-2,2),ylims=(-2,2),N=128):
        plt.clf()
        plt.subplot(1,3,1)
        plt.cla()
        xx,yy = np.meshgrid(np.linspace(xlims[0],xlims[1],N),np.linspace(ylims[0],ylims[1],N))
        self.landscape.plot2d(xx,yy)
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.cla()
        plt.plot(self.x_path,self.y_path,'k.')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.subplot(1,3,3)
        plt.cla()
        plt.imshow(self.image,cmap='gray',interpolation='none')
        sy,sx = self.image.shape
        plt.xlim((sx//2+50*self.x-400,sx//2+50*self.x+400))
        plt.ylim((sy//2+50*self.y-400,sy//2+50*self.y+400))
        
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
        #depths = depths - np.min(depths)

        #plt.plot(depths)
        #plt.show()
        
        mode='stochastic'

        if mode=='stochastic':
            # compute a CDF using a power of the resulting positive depths
            # check for corner case where depths are all zero, and assign
            # a same-size vector of ones, i.e. all directions equally probable
            depths = np.max(depths)-depths
            
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
            winners = np.where(depths==np.min(depths))[0]
            if len(winners)>1:
                winner = winners[np.random.randint(len(winners))]
            else:
                winner = winners[0]
                               
            
        x = xx[winner]
        y = yy[winner]


        # check for a saccade:
        activation = self.landscape.evaluate(x,y)
        if activation>self.saccade_threshold:
            # evaluate over a fine grid +/- 2 deg, centered about origin
            xx,yy = np.meshgrid(np.linspace(-2,2,128),np.linspace(-2,2,128))
            surf = self.landscape.evaluate(xx,yy)
            #plt.figure()
            #plt.imshow(surf)
            #plt.colorbar()
            #plt.show()
            #print np.where(surf==np.min(surf))
            ywin = np.where(surf==np.min(surf))[0][0]
            xwin = np.where(surf==np.min(surf))[1][0]
            y = yy[ywin,xwin]
            x = xx[ywin,xwin]
        #print 'Moving to %0.2f,%0.2f.'%(x,y)
        #plt.figure()
        #plt.plot(weights)
        #plt.plot(winner,weights[winner],'go')
        #plt.show()

        # move to new location and add a depression there
        self.x = x
        self.y = y
        self.x_path.append(x)
        self.y_path.append(y)
        self.landscape.add(Feature(x,y))
        
if __name__=='__main__':


    im = np.load('./images/rocks.npy')
    
    XX,YY = np.meshgrid(np.arange(-3,3,.01),np.arange(-3,3,.01))
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    
    g = Gaze(drift_potential_slope=1.0,image=im)
    while True:
        g.step()
        #g.plot_surface(ax,XX,YY)
        if g.landscape.age%10==0:
            g.plot()
            plt.pause(.00000001)
        