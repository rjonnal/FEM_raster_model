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

    def __init__(self,x1=-0.25,x2=0.25,y1=-0.25,y2=0.25,central_field_strength=0.0,potential_fwhm_m=3e-5,N_cones=0,locality=0.05,granularity=0.001):

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

        self.XX,self.YY = np.meshgrid(np.arange(-locality,locality+granularity,granularity),
                                      np.arange(-locality,locality+granularity,granularity))

        

        self.subx1 = np.min(self.XX)
        self.subx2 = np.max(self.XX)
        self.subdx = self.subx2-self.subx1
        self.suby1 = np.min(self.YY)
        self.suby2 = np.max(self.YY)
        self.subdy = self.suby2-self.suby1
        self.NX,self.NY = self.XX.shape
        
        self.XX,self.YY = self.XX.ravel(),self.YY.ravel()

        self.neighborhood = self.locality

        potential_fwhm_deg = potential_fwhm_m/3e-4
        
        self.potential_sigma = potential_fwhm_deg/(2.0*np.sqrt(2.0*np.log(2)))
        self.central_field_strength = central_field_strength
        
    def find_neighbors(self,x,y,rad):
        d = np.sqrt((x-self.cones_x)**2+(y-self.cones_y)**2)
        neighbors = np.where(d<rad)[0]
        return neighbors

    def compute_field(self,x,y):
        neighbors = self.find_neighbors(x,y,self.neighborhood)
        N_neighbors = len(neighbors)

        xx = self.XX.copy()+x
        yy = self.YY.copy()+y

        def rs(vec):
            return np.reshape(vec,(self.NY,self.NX))

        if False:
            print rs(self.XX)
            print rs(self.YY)
            print x,y
            print rs(xx)
            print rs(yy)
        
        # build a matrix of coordinates
        dx = np.tile(xx,(N_neighbors,1)).T
        dy = np.tile(yy,(N_neighbors,1)).T

        # get the coordinates of the neighbors
        neighbor_x_coords = self.cones_x[neighbors]
        neighbor_y_coords = self.cones_y[neighbors]

        dx = (dx - neighbor_x_coords).T
        dy = (dy - neighbor_y_coords).T

        if False:
            print 'Neighborhoods:'
            for nidx in range(N_neighbors):
                print rs(dx[nidx,:])
                print rs(dy[nidx,:])
                print
        
        
        field = np.exp(-(dx**2+dy**2)/(2.0*self.potential_sigma**2))

        if False:
            plt.figure()
            plt.imshow(dx,interpolation='none')
            plt.colorbar()
            plt.figure()
            plt.imshow(dy,interpolation='none')
            plt.colorbar()
            plt.figure()
            plt.imshow(field,interpolation='none')
            plt.colorbar()
            plt.show()
        
        field = np.sum(field,axis=0)

        if False:
            plt.figure()
            self.plot()
            plt.plot(x,y,'gs')
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
            
        cfield = np.exp(xx**2+yy**2)*self.central_field_strength
        
        field = field + cfield
        return field,neighbors

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


    def plot(self):
        plt.plot(self.cones_x,self.cones_y,'k.')
        
    def step(self):
        idx_vec = range(self.N_cones)
        shuffle(idx_vec)

        for idx in idx_vec:
            x,y = self.cones_x[idx],self.cones_y[idx]
            f,n = self.compute_field(x,y)
            winners = list(np.where(f==np.min(f))[0])
            shuffle(winners)
            self.cones_x[idx] = x + self.XX[winners[0]]
            self.cones_y[idx] = y + self.YY[winners[0]]

            if idx%10000==0:
                plt.cla()
                self.plot()
                plt.pause(.001)
        
if __name__=='__main__':

    #r = Retina(x1=-.01,x2=.01,y1=-.01,y2=.01,N_cones=10,locality=.01,granularity=.01,central_field_strength=1000.0)
    r = Retina(x1=-.25,x2=.25,y1=-.25,y2=.25,N_cones=4000,locality=.005,granularity=.00005,central_field_strength=1.0)

    f,n = r.compute_field(0,0)
    f = np.reshape(f,(r.NY,r.NX))
    
    for k in range(10):
        print k
        r.step()
    plt.show()
