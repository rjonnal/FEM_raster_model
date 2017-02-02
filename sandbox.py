import numpy as np
from matplotlib import pyplot as plt
import os,sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

if True:
    xx,yy = np.meshgrid(np.linspace(-1,1,256),np.linspace(-1,1,256))
    im = xx**2+yy**2
    #im = np.sqrt(xx**2+yy**2)
    #im = np.exp(xx**2+yy**2)
    plt.imshow(im)
    plt.colorbar()
    plt.show()
    

if False:

    # deformation of the field due to a movement to this position:
    # -1.0/np.exp(np.sqrt(xx**2+yy**2)*self.steepness)*self.A

    x = np.linspace(-1,1,1024)
    y = -1.0/np.exp(np.sqrt(x**2)*10.0)

    plt.plot(x,y)
    plt.show()

# drift potential
if False:
    L = 51.0
    L0 = (L-1.0)/2.0
    II,JJ = np.meshgrid(np.arange(L),np.arange(L))

    l = 1.0

    u = l*L*(((II-L0)/L0)**2 + ((JJ-L0)/L0)**2)
    plt.imshow(u)
    plt.show()

# saccade potential
if False:
    L = 51.0
    L0 = (L-1.0)/2.0
    II,JJ = np.meshgrid(np.arange(L),np.arange(L))

    X = 2.0

    i1 = 25.0
    j1 = 25.0

    t1 = X*L
    t2 = ((II-i1)/L0)**2
    t3 = ((JJ-j1)/L0)**2
    #u1 = X*L*(((II-i1)/L0)**2 * ((JJ-j1)/L0)**2)
    plt.imshow(t2*t3)
    plt.autoscale(False)
    plt.plot(i1,j1,'ks')
    plt.colorbar()
    plt.show()


