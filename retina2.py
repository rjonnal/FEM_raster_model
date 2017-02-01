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

    def __init__(self,x1=-0.25,x2=0.25,y1=-0.25,y2=0.25,central_field_strength=0.0,potential_fwhm_m=1e-5,N_cones=0,locality=0.05,granularity=0.001):

        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1

        self.cones_x = np.random.rand(N_cones)*self.dx+self.x1
        self.cones_y = np.random.rand(N_cones)*self.dy+self.y1

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
        
        # build a matrix of coordinates
        dx = np.tile(xx,(N_neighbors,1)).T
        dy = np.tile(yy,(N_neighbors,1)).T

        # get the coordinates of the neighbors
        neighbor_x_coords = self.cones_x[neighbors]
        neighbor_y_coords = self.cones_y[neighbors]

        dx = dx + neighbor_x_coords
        dy = dy + neighbor_y_coords

        field = np.exp(-(dx**2+dy**2)/(2.0*self.potential_sigma**2)).T
        field = np.sum(field,axis=0)


        # box coords for plotting
        x1 = np.min(xx)
        x2 = np.max(xx)
        y1 = np.min(yy)
        y2 = np.max(yy)
        
        plt.subplot(1,2,1)
        plt.plot(x,y,'ro')
        self.plot()
        plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'b-')
        plt.subplot(1,2,2)
        plt.imshow(np.reshape(field,(self.NY,self.NX)),interpolation='none')
        plt.show()

        cfield = 0*np.exp(xx**2+yy**2)*self.central_field_strength
        
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
        plt.plot(self.cones_x,self.cones_y,'ks')
        
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

            if idx%10==0:
                plt.cla()
                self.plot()
                plt.pause(.001)
        
        
if __name__=='__main__':

    r = Retina(N_cones=1000,granularity=0.01,central_field_strength=1000.0)
    for k in range(10):
        r.step()
    #r.display_field(0,0)
    #print r.find_neighbors(0.0,0.0,0.25)
