import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys
from time import time,sleep
from fig2gif import GIF
from cone_density import ConeDensityInterpolator
from random import shuffle
from data_store import H5
import datetime

class Mosaic:

    def __init__(self,x1=-0.25,x2=0.25,y1=-0.25,y2=0.25,central_field_strength=0.0,potential_fwhm_deg=1e-6,N_cones=0,locality=0.05,granularity=0.001,noise=0.0):

        self.age = 0
        self.noise = noise
        self.use_cdf = False
        
        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
        self.xmid = (self.x1+self.x2)/2.0
        self.ymid = (self.y1+self.y2)/2.0

        max_rad = np.min(self.dx/2.0,self.dy/2.0)*np.sqrt(2)
        
        self.cones_rad = np.random.rand(N_cones)**.5*max_rad
        self.cones_theta = np.random.rand(N_cones)*np.pi*2
        self.cones_x = (np.cos(self.cones_theta)*self.cones_rad).astype(np.float32)
        self.cones_y = (np.sin(self.cones_theta)*self.cones_rad).astype(np.float32)

        self.cones_I = (10.0 + np.random.randn(N_cones)*2).clip(1.0,np.inf)
        
        self.N_cones = N_cones
        self.N_stationary = 0
        
        self.locality = locality
        self.granularity = granularity
        self.theta_step = np.pi/100.0
        self.theta = np.arange(0,np.pi*2,self.theta_step)
        self.neighborhood = self.locality*2

        self.cone_potential_fwhm_deg = potential_fwhm_deg
        self.potential_sigma = potential_fwhm_deg/(2.0*np.sqrt(2.0*np.log(2)))
        self.intensity_sigma = self.potential_sigma
        self.central_field_strength = central_field_strength
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        self.tag = self.get_tag()
        self.h5 = H5('./histories/%s.hdf5'%self.tag)

        self.h5.put('/params/N_cones',self.N_cones)
        self.h5.put('/params/locality',self.locality)
        self.h5.put('/params/granularity',self.granularity)
        self.h5.put('/params/theta_step',self.theta_step)
        self.h5.put('/params/neighborhood',self.neighborhood)
        self.h5.put('/params/cone_potential_fwhm_deg',self.cone_potential_fwhm_deg)
        self.h5.put('/params/potential_sigma',self.potential_sigma)
        self.h5.put('/params/central_field_strength',self.central_field_strength)
        self.h5.put('/params/rect',[self.x1,self.x2,self.y1,self.y2])
        self.h5.put('/inentsities',self.cones_I)

    def save_mosaic(self,N=512):
        x1 = self.x1
        x2 = self.x2
        y1 = self.y1
        y2 = self.y2
        xr = np.linspace(x1,x2,N)
        yr = np.linspace(y1,y2,N)
        XX,YY = np.meshgrid(xr,yr)
        out = np.zeros(XX.shape,dtype=np.float32)
        for idx,(x,y,I) in enumerate(zip(self.cones_x,self.cones_y,self.cones_I)):
            xx = XX - x
            yy = YY - y
            out = out + np.exp(-(xx**2+yy**2)/(2.0*self.intensity_sigma**2))*I
        self.h5.put('/%06d/mosaic'%self.age,out)
        self.mosaic = out
            
    def get_tag(self):
        # main part of tag: ncones_centralfield_conefield_rect
        fmts = ['%06d','%0.1f','%0.4f','%0.2f','%0.2f','%0.2f','%0.2f']
        params = (self.N_cones,self.central_field_strength,self.cone_potential_fwhm_deg,
                  self.x1,self.x2,self.y1,self.y2)

        tag = ''
        for f in fmts:
            tag = tag + f + '_'

        tag = tag%params
        tag = tag.replace('-','m')
        tag = tag + self.timestamp
        return tag

    def record(self):
        def put(name,data):
            self.h5.put('/%06d/%s'%(self.age,name),data)

        put('cones_x',self.cones_x)
        put('cones_y',self.cones_y)
        put('N_stationary',self.N_stationary)
    
    def find_neighbors(self,x,y,rad):
        d = np.sqrt((x-self.cones_x)**2+(y-self.cones_y)**2)
        neighbors = np.where(np.logical_and(d<rad,d>0))[0]
        #neighbors = np.where(d<rad)[0]
        return neighbors

    def compute_field(self,x,y):
        neighbors = self.find_neighbors(x,y,self.neighborhood)

        theta = self.theta + np.random.rand()*self.theta_step
        mag = self.granularity
        
        XX = np.cos(theta)*mag
        YY = np.sin(theta)*mag
        XX = np.array(list(XX)+[0.0])
        YY = np.array(list(YY)+[0.0])
        
        xx = XX+x
        yy = YY+y

        f,n,xx,yy = self.compute_field_helper(xx,yy,neighbors)
        f = f+np.random.randn(len(f))*np.sqrt(f)*self.noise
        return f,n,xx,yy

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
    
        
    def plot(self,zoom=1.0):
        plt.plot(self.cones_x,self.cones_y,'k.')

        x1 = self.xmid-self.dx/2.0*np.sqrt(2)/zoom
        x2 = self.xmid+self.dx/2.0*np.sqrt(2)/zoom
        y1 = self.ymid-self.dy/2.0*np.sqrt(2)/zoom
        y2 = self.ymid+self.dy/2.0*np.sqrt(2)/zoom
        
        plt.xlim((x1,x2))
        plt.ylim((y1,y2))


    def f2cdf(self,f):
        # convert a field into a CDF such that the
        # lowest points in the field have the highest
        # chance of being selected

        if not np.min(f)==np.max(f):
            weights = np.max(f)-f
        else:
            weights = np.ones(f.shape)

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

        return winner

    def stationary_fraction(self):
        return float(self.N_stationary)/float(self.N_cones)
    
    def step(self):
        self.record()
        self.age = self.age + 1
        idx_vec = range(self.N_cones)
        #shuffle(idx_vec)

        oldxs = []
        oldys = []
        newxs = []
        newys = []
        contrasts = []
        neighbor_counts = []
        
        self.N_stationary = 0.0

        n_expected = 0.0
        
        for idx in idx_vec:
            #print '\t%d'%idx
            x,y = self.cones_x[idx],self.cones_y[idx]
            f,n,XX,YY = self.compute_field(x,y)
            neighbor_counts.append(len(n))
            #print f

            #f = f + np.sqrt(f)*np.random.randn(len(f))*.1
            
            fmin = np.min(f)
            fmax = np.max(f)
            contrasts.append((fmax-fmin)/fmax)

            if self.use_cdf:
                winner = self.f2cdf(f)
            else:
                winner = np.argmin(f)
    


            if f[winner]==fmin:
                n_expected = n_expected + 1
            
            if f[-1]==fmin:
                self.N_stationary = self.N_stationary+1
                winner = len(f)-1
                
            oldxs.append(self.cones_x[idx])
            oldys.append(self.cones_y[idx])
            self.cones_x[idx] = XX[winner]
            self.cones_y[idx] = YY[winner]
            newxs.append(self.cones_x[idx])
            newys.append(self.cones_y[idx])

        nn = np.mean(neighbor_counts)
        
        print 'N_stationary: %d (%d%%)'%(self.N_stationary,self.stationary_fraction()*100.)
        G = 10
        if self.age%1==0:
            plt.clf()
            plt.subplot(1,3,1)
            plt.cla()
            plt.semilogy(f)
            plt.ylim((0,len(self.cones_x)))
            plt.subplot(1,3,2)
            plt.cla()
            self.plot()
            for oldx,oldy,newx,newy in zip(oldxs[-G:],oldys[-G:],newxs[-G:],newys[-G:]):
                plt.plot([oldx,newx],[oldy,newy],'b-')
                plt.plot(newx,newy,'ro')
            plt.subplot(1,3,3)
            plt.cla()
            self.plot(zoom=5.0)
            plt.pause(.001)
        
if __name__=='__main__':

    locality = .02
    granularity = .0025

    cone_potential_fwhm_vec = [.01,.02]
    central_field_strength_vec = [5.0,10.0]

    for cone_potential_fwhm in cone_potential_fwhm_vec:
        for central_field_strength in central_field_strength_vec:
    
            m = Mosaic(x1=-.25,x2=.25,y1=-.25,y2=.25,N_cones=4500,locality=locality,granularity=granularity,central_field_strength=central_field_strength,potential_fwhm_deg=cone_potential_fwhm)

            while m.stationary_fraction()<.95 and m.age<200:
                plt.figure(1)
                m.step()
                m.save_mosaic()
                plt.figure(2)
                plt.imshow(m.mosaic,cmap='gray',interpolation='none')
                plt.pause(1)