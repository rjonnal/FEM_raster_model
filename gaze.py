import numpy as np
from matplotlib import pyplot as plt
import os,sys,time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#DRIFT_SPEED = 1.0 # deg/s
#DRIFT_SPEED = 5.0 # deg/s
DRIFT_SPEED = 0.5 # deg/s

class Feature:
    A0 = 1.0 # initial amplitude
    def __init__(self,x,y,dt,relaxation_rate):
        self.age = 0.0
        self.x = x
        self.y = y
        self.A = self.A0
        self.relaxation_rate = relaxation_rate
        self.dt = dt
        
    def step(self):
        self.age = self.age + self.dt
        self.A = np.exp(-self.age*self.relaxation_rate)

    def evaluate_pit(self,xx0,yy0):
        d = np.sqrt((xx0-self.x)**2+(yy0-self.y)**2)
        height = np.ones(xx0.shape)*self.A
        test = np.where(d>self.dt*DRIFT_SPEED)
        height[test] = 0.0
        return height
        
    def evaluate_exponential(self,xx0,yy0):
        xx0 = np.array(xx0)
        yy0 = np.array(yy0)
        xx = xx0.copy()-self.x
        yy = yy0.copy()-self.y
        return 1.0/np.exp(np.sqrt(xx**2+yy**2)*self.steepness)*self.A

    def evaluate(self,xx0,yy0):
        return self.evaluate_pit(xx0,yy0)
    
    def plot3d(self,ax,xx,yy):
        surf = self.evaluate(xx,yy)
        ax.cla()
        ax.plot_surface(xx, yy, surf, rstride=10, cstride=10, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim((-.1,1.1))

    def plot2d(self,xx=None,yy=None):
        if xx is None or yy is None:
            xx,yy = np.meshgrid(np.linspace(-.1,.1,1024),np.linspace(-.1,.1,1024))
        plt.cla()
        surf = self.evaluate(xx,yy)
        plt.imshow(surf)

class ConicalPotential(Feature):

    # a conical potential with a peak at the center
    # unlike footprints, the potential doesn't relax
    # in its step() function
    
    def __init__(self,x,y,dt,L=1.0):
        Feature.__init__(self,x,y,dt,relaxation_rate=1.0)
        self.L = L
        self.A = 1.0
        
    def evaluate(self,xx0,yy0):
        xx0 = np.array(xx0)
        yy0 = np.array(yy0)
        xx = xx0.copy()-self.x
        yy = yy0.copy()-self.y
        surf = self.L*(xx**2+yy**2)
        return surf
        
    def step(self):
        self.age = self.age + self.dt
        self.A = 1.0

class FeatureSet:

    Aepsilon = .01
    Aepsilon = .99999
    def __init__(self):
        self.depressions = []
        self.age = 0
        
    def add(self,depression):
        self.depressions.append(depression)
        
    def step(self):
        #print len(self.depressions)
        for d in self.depressions:
            d.step()
            # remove depressions with negligible amplitudes
            if d.A<self.Aepsilon:#+np.random.randn()*self.Aepsilon*.1:
                self.depressions.remove(d)
        #self.plot2d()
        #plt.colorbar()
        #plt.show()
        
    def evaluate(self,xx,yy,do_plot=False):
        xx = np.array(xx)
        yy = np.array(yy)
        depression_sum = np.zeros(xx.shape)

        if do_plot:
            plt.figure()
            
        for d in self.depressions:
            depression_sum = depression_sum + d.evaluate(xx,yy)
            if do_plot:
                plt.cla()
                plt.plot(depression_sum)
                plt.title(d)
                plt.pause(.1)

        if do_plot:
            plt.show()
        #depression_sum = np.clip(depression_sum,0.0,1.0)
        return depression_sum

    def plot3d(self,ax,xx,yy):
        surf = self.evaluate(xx,yy)
        ax.cla()
        ax.plot_surface(xx, yy, surf, rstride=10, cstride=10, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim((-.1,1.1))

    def plot2d(self,xx=None,yy=None):
        if xx is None or yy is None:
            xx,yy = np.meshgrid(np.linspace(-.01,.01,1024),np.linspace(-.01,.01,1024))
        plt.cla()
        surf = self.evaluate(xx,yy)
        plt.imshow(surf)

class GazeHistory:

    drift_marker = 'k.'
    saccade_marker = 'r-'
    
    def __init__(self):
        self.xvec = [0]
        self.yvec = [0]
        self.saccades = []
        
    def plot(self):
        plt.plot(self.xvec,self.yvec,self.drift_marker)
        for s in self.saccades:
            plt.plot(s[0],s[1],self.saccade_marker,linewidth=2)
                
    def add(self,x,y,is_saccade):
        if is_saccade:
            self.saccades.append([[self.xvec[-1],x],[self.yvec[-1],y]])
        self.xvec.append(x)
        self.yvec.append(y)

    def clear(self):
        self.xvec = [0]
        self.yvec = [0]
        self.saccades = []
        
class Gaze:

    
    def __init__(self,dt,x0=0.0,y0=0.0,drift_relaxation_rate=1.0e-1,drift_potential_slope=1.0,saccade_potential_slope=2.0,fractional_saccade_activation_threshold=1.1,image=None,image_subtense=None):

        self.d_theta = np.pi/100.0
        self.dt = dt
        self.step_size = DRIFT_SPEED*self.dt
        self.age = 0.0
        self.x = x0
        self.y = y0
        self.x0 = x0
        self.y0 = y0
        self.drift_relaxation_rate = drift_relaxation_rate
        self.drift_potential_slope = drift_potential_slope
        self.saccade_potential_slope = saccade_potential_slope

        self.saccade_threshold = fractional_saccade_activation_threshold*np.abs(drift_potential_slope)
        
        self.x_path = []
        self.y_path = []
        self.landscape = FeatureSet()
        self.landscape.add(Feature(self.dt,x0,y0,relaxation_rate=self.drift_relaxation_rate))
        self.conical_potential = ConicalPotential(x0,y0,self.dt,L=drift_potential_slope)
        if image is None:
            self.image = np.zeros((100,100))
        else:
            self.image = image
        self.history = GazeHistory()
        if image_subtense is None:
            self.image_subtense = 1.0
        else:
            self.image_subtense = image_subtense
        
    def plot_surface(self,ax,xx,yy):
        self.landscape.plot3d(ax,xx,yy)

    def plot(self,xlims=(-.001,.001),ylims=(-.001,.001),N=128,zoom=1.0):
        plt.clf()
        plt.subplot(2,3,1)
        plt.cla()
        xx,yy = np.meshgrid(np.linspace(xlims[0],xlims[1],N),np.linspace(ylims[0],ylims[1],N))
        current_surface = self.landscape.evaluate(xx,yy)+self.conical_potential.evaluate(xx,yy)
        plt.imshow(current_surface)
        plt.colorbar()
        plt.title('Age %0.2f'%self.landscape.age)
        plt.subplot(2,3,2)
        plt.cla()
        self.history.plot()
        #plt.plot(self.x_path,self.y_path,'k.')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.subplot(2,3,3)
        plt.cla()
        plt.imshow(self.image,cmap='gray',interpolation='none')

        px_per_deg = float(self.image.shape[0])/self.image_subtense
        winx = 500.
        winy = 500.

        xoff_px = self.x*px_per_deg
        yoff_px = self.y*px_per_deg

        sy,sx = self.image.shape
        xmid = sx/2.0
        ymid = sy/2.0
        x1 = xmid-winx/2.0+xoff_px
        x2 = xmid+winx/2.0+xoff_px
        y1 = ymid-winy/2.0+yoff_px
        y2 = ymid+winy/2.0+yoff_px
        
        plt.xlim((x1,x2))
        plt.ylim((y1,y2))


        plt.subplot(2,3,4)
        plt.cla()
        xx,yy = np.meshgrid(np.linspace(xlims[0]/zoom,xlims[1]/zoom,N),np.linspace(ylims[0]/zoom,ylims[1]/zoom,N))
        current_surface = self.landscape.evaluate(xx,yy)+self.conical_potential.evaluate(xx,yy)
        plt.imshow(current_surface)
        plt.colorbar()
        plt.title('Age %0.2f'%self.landscape.age)
        plt.subplot(2,3,5)
        plt.cla()
        self.history.plot()
        #plt.plot(self.x_path,self.y_path,'k.')
        plt.xlim([xl/zoom for xl in xlims])
        plt.ylim([yl/zoom for yl in ylims])
        plt.subplot(2,3,6)
        plt.cla()
        plt.imshow(self.image,cmap='gray',interpolation='none')

        px_per_deg = float(self.image.shape[0])/self.image_subtense
        winx = 500./zoom
        winy = 500./zoom

        xoff_px = self.x*px_per_deg
        yoff_px = self.y*px_per_deg

        sy,sx = self.image.shape
        xmid = sx/2.0
        ymid = sy/2.0
        x1 = xmid-winx/2.0+xoff_px
        x2 = xmid+winx/2.0+xoff_px
        y1 = ymid-winy/2.0+yoff_px
        y2 = ymid+winy/2.0+yoff_px
        
        plt.xlim((x1,x2))
        plt.ylim((y1,y2))


        

        
        
    def get_ring(self,r):
        thetas = np.arange(0,np.pi*2,self.d_theta)
        xx = np.cos(thetas)*r+self.x
        yy = np.sin(thetas)*r+self.y
        return xx,yy


    def step(self,show_time=False):

        if show_time:
            t0 = time.time()
        self.landscape.step()
        xarr = np.array([self.x])
        yarr = np.array([self.y])
        
        current_activation = self.landscape.evaluate(xarr,yarr) + self.conical_potential.evaluate(xarr,yarr)
        do_saccade = current_activation>self.saccade_threshold

        if do_saccade:
            new_x,new_y = self.compute_saccade()
            way='saccad'
        else:
            new_x,new_y = self.compute_drift()
            way='drift'
        #print '%sing from'%way,self.x,self.y
        #print 'to',new_x,new_y
        #print
        
        self.x = new_x
        self.y = new_y
        self.history.add(new_x,new_y,do_saccade)
        
        self.landscape.add(Feature(new_x,new_y,self.dt,self.drift_relaxation_rate))
        if show_time:
            print 'age: %d ms, step time (real): %d ms, step history: %d, x: %0.5f, y: %0.5f'%(self.age*1000,(time.time()-t0)*1000,len(self.landscape.depressions),self.x,self.y)

        self.age = self.age + self.dt

    def compute_saccade(self):
        # for saccades, unlike drifts, we have to compute the whole landscape
        # because we're goint to saccade to the global activation minimum
        rad = 2.0
        N = 256
        # a function to get us back from degrees to grid location, for plotting
        def rep(x,y):

            dpp = 2*rad/float(N) # deg/pixel
            xpx = x/dpp+float(N)/2.0
            ypx = y/dpp+float(N)/2.0
            return xpx,ypx
        
        xx,yy = np.meshgrid(np.linspace(-rad,rad,N),np.linspace(-rad,rad,N))

        # compute the drift+potential surface:
        self_avoidance_potential_surface = self.landscape.evaluate(xx,yy)
        centering_potential_surface = self.conical_potential.evaluate(xx,yy)
        drift_potential_surface = self_avoidance_potential_surface + centering_potential_surface
    
        
        # compute and add the saccade potential
        sxx = xx.copy() - self.x
        syy = yy.copy() - self.y
        saccade_potential_surface = sxx**2*syy**2*self.saccade_potential_slope
        potential_surface = drift_potential_surface + saccade_potential_surface


        
        minimum = np.where(potential_surface==np.min(potential_surface))
        #print np.min(potential_surface),np.log(np.min(potential_surface))
        #print minimum
        ymin = minimum[0][0]
        xmin = minimum[1][0]

        new_x = xx[ymin,xmin]
        new_y = yy[ymin,xmin]

        if False:
            plt.figure()
            plt.imshow(np.log(potential_surface))
            plt.colorbar()
            plt.autoscale(False)
            plt.plot(*rep(self.x,self.y),marker='o',markerfacecolor='g',markeredgecolor='w',markersize=10)
            plt.plot(*rep(new_x,new_y),marker='o',markerfacecolor='r',markeredgecolor='w',markersize=10)
            plt.show()

        
        return new_x,new_y
        

    def compute_drift(self):
        # implement a self-avoiding random walk
        # self-avoidance is implemented using a continuous
        # version of discrete swamp walk described by
        # Engbert et al., PNAS, 2011
        # [http://www.pnas.org/content/108/39/E765.full]

        # compute points along ring r distance from current location
        xx,yy = self.get_ring(self.step_size)

        # get the depths along this ring
        depths = self.landscape.evaluate(xx,yy)
        depths = depths + self.conical_potential.evaluate(xx,yy)
        # convert the depths into a cumulative density
        # function such that the lowest depth makes the
        # highest CDF contribution
        if not np.min(depths)==np.max(depths):
            weights = np.max(depths)-depths
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

        new_y = yy[winner]
        new_x = xx[winner]

        return new_x,new_y
        
if __name__=='__main__':
    step_duration = 1.0/(30.0*512.0)
    
    im = np.load('./images/mosaic.npy')
    
    XX,YY = np.meshgrid(np.arange(-1,1,.005),np.arange(-1,1,.005))
    fig = plt.figure()
    
    g = Gaze(step_duration,drift_relaxation_rate=2.5e-1,drift_potential_slope=1.0,saccade_potential_slope=2.0,fractional_saccade_activation_threshold=2.0,image=im,image_subtense=1.0)
    while True:
        g.step()
        g.plot()
        plt.pause(.00001)
        
