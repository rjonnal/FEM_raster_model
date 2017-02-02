import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys
from time import time,sleep
from fig2gif import GIF
from cone_density import ConeDensityInterpolator
from random import shuffle

class Retina:

    def __init__(self,x1=-0.25,x2=0.25,y1=-0.25,y2=0.25,central_field_strength=0.0,potential_fwhm_deg=1e-6,N_cones=0,locality=0.05,granularity=0.001):

        self.age = 0
        self.factor = np.exp(-self.age/10.0)
        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1


        max_rad = np.min(self.dx/2.0,self.dy/2.0)
        
        self.cones_rad = np.random.rand(N_cones)**.5*max_rad
        self.cones_theta = np.random.rand(N_cones)*np.pi*2
        self.cones_x = np.cos(self.cones_theta)*self.cones_rad
        self.cones_y = np.sin(self.cones_theta)*self.cones_rad
        
        #self.cones_x = np.random.rand(N_cones)*self.dx+self.x1
        #self.cones_y = np.random.rand(N_cones)*self.dy+self.y1

        self.N_cones = N_cones

        self.locality = locality
        self.granularity = granularity
        self.theta_step = np.pi/100.0
        self.theta = np.arange(0,np.pi*2,self.theta_step)
        self.neighborhood = self.locality

        self.potential_sigma = potential_fwhm_deg/(2.0*np.sqrt(2.0*np.log(2)))
        self.central_field_strength = central_field_strength
        
    def find_neighbors(self,x,y,rad):
        d = np.sqrt((x-self.cones_x)**2+(y-self.cones_y)**2)
        #neighbors = np.where(np.logical_and(d<rad,d>0))[0]
        neighbors = np.where(d<rad)[0]
        return neighbors

    def compute_field(self,x,y):
        neighbors = self.find_neighbors(x,y,self.neighborhood)

        theta = self.theta + np.random.rand()*self.theta_step

        #self.factor = np.exp(-self.age/50.0)
        mag = self.factor*self.granularity
        
        XX = np.cos(theta)*mag
        YY = np.sin(theta)*mag
        XX = np.array(list(XX)+[0.0])
        YY = np.array(list(YY)+[0.0])
        
        xx = XX+x
        yy = YY+y
        
        return self.compute_field_helper(xx,yy,neighbors)

    def compute_full_field(self,N=64):
        xvec = np.linspace(self.x1,self.x2,N)
        yvec = np.linspace(self.y1,self.y2,N)
        xx,yy = np.meshgrid(xvec,yvec)
        xx = xx.ravel()
        yy = yy.ravel()
        neighbors = np.arange(len(self.cones_x))
        return self.compute_field_helper(xx,yy,neighbors)
        
    def compute_field_helper(self,xx,yy,neighbors):
        N_neighbors = len(neighbors)
        
        # build a matrix of coordinates
        dx = np.tile(xx,(N_neighbors,1)).T
        dy = np.tile(yy,(N_neighbors,1)).T

        
        # get the coordinates of the neighbors
        neighbor_x_coords = self.cones_x[neighbors]
        neighbor_y_coords = self.cones_y[neighbors]

        dx = (dx - neighbor_x_coords).T
        dy = (dy - neighbor_y_coords).T

        field = np.exp(-(dx**2+dy**2)/(2.0*self.potential_sigma**2))

        #field = field + np.sqrt(field)*np.random.randn(*field.shape)
        
        field = np.sum(field,axis=0)
        if False:
            plt.figure()
            self.plot()
            plt.plot(xx,yy,'gs')
            plt.figure()
            self.plot()
            plt.plot(dx,dy,'gs')
            plt.figure()
            plt.plot(field)
            plt.show()

        # box coords for plotting
        x1 = np.min(xx)
        x2 = np.max(xx)
        y1 = np.min(yy)
        y2 = np.max(yy)

        if False:
            plt.subplot(1,2,1)
            plt.plot(x,y,'ro')
            self.plot()
            plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'b-')
            plt.subplot(1,2,2)
            plt.imshow(np.reshape(field,(self.NY,self.NX)),interpolation='none')
            plt.show()
            
        #cfield = np.exp(xx**2+yy**2)*self.central_field_strength
        cfield = np.sqrt(xx**2+yy**2)*self.central_field_strength
        
        field = field + cfield
        return field,neighbors,xx,yy

    def xidx(self,x):
        return (x-self.subx1)/self.subdx*self.NX
    
    def yidx(self,y):
        return (y-self.suby1)/self.subdy*self.NY
    
    def display_field(self,x,y,f=None,n=None):
        if f is None:
            f,n = self.compute_field(x,y)
        f = np.reshape(f,(self.NY,self.NX))[::-1,::-1]
        locality = self.locality
        
        plt.imshow(f,interpolation='none')
        #plt.colorbar()
        plt.autoscale(False)
        x = self.xidx(self.cones_x[n])
        y = self.yidx(self.cones_y[n])
        plt.plot(x,y,'ks')


    def display_full_field(self,N=64):
        f,n = r.compute_full_field(N)
        f = np.reshape(f,(N,N))
        plt.imshow(f,interpolation='none')
        plt.colorbar()
    
        
    def plot(self):
        plt.plot(self.cones_x,self.cones_y,'k.')
        plt.xlim((self.x1,self.x2))
        plt.ylim((self.y1,self.y2))
        
    def step(self):
        self.age = self.age + 1
        idx_vec = range(self.N_cones)
        #shuffle(idx_vec)

        oldxs = []
        oldys = []
        newxs = []
        newys = []
        vars = []
        for idx in idx_vec:
            #print '\t%d'%idx
            x,y = self.cones_x[idx],self.cones_y[idx]
            f,n,XX,YY = self.compute_field(x,y)
            #print f
            fmin = np.min(f)
            vars.append(np.var(f))
            #print fmin,
            winners = list(np.where(f==fmin)[0])
            #shuffle(winners)
            #print winners
            oldxs.append(self.cones_x[idx])
            oldys.append(self.cones_y[idx])
            self.cones_x[idx] = XX[winners[0]]
            self.cones_y[idx] = YY[winners[0]]
            newxs.append(self.cones_x[idx])
            newys.append(self.cones_y[idx])
        #print

        #self.factor = np.sqrt(np.max(vars))*3.0
        self.factor = np.sqrt(np.percentile(vars,99))*3.3
        G = 10
        if self.age%1==0:
            plt.clf()
            plt.subplot(1,2,1)
            plt.cla()
            plt.semilogy(f)
            plt.ylim((0,len(self.cones_x)))
            plt.subplot(1,2,2)
            plt.cla()
            self.plot()
            for oldx,oldy,newx,newy in zip(oldxs[-G:],oldys[-G:],newxs[-G:],newys[-G:]):
                plt.plot([oldx,newx],[oldy,newy],'b-')
                plt.plot(newx,newy,'ro')
            #plt.subplot(1,3,3)
            #if self.age%100==0:
            #    plt.cla()
            #    self.display_full_field(16)
            plt.pause(.001)
        
if __name__=='__main__':

    locality = 1e-1
    granularity = locality/20.0
    cone_potential_fwhm = locality/10.0
    
    r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25,N_cones=1000,locality=locality,granularity=granularity,central_field_strength=1.0e1,potential_fwhm_deg=cone_potential_fwhm)
    #r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25,N_cones=4000,locality=.0001,granularity=.00001,central_field_strength=1.0e-1,potential_fwhm_m=1e-7)

    
    while r.factor>.01:
        print r.factor
        r.step()
    plt.show()
