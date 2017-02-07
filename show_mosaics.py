import numpy as np
from matplotlib import pyplot as plt
import os,sys
import glob
from data_store import H5
from mosaic import Mosaic

for f in glob.glob('./histories/*.hdf5'):
    m = Mosaic(hdf5fn=f)
    
    cp = m.cone_potential_fwhm_deg
    cfs = m.central_field_strength
    
    mosaic = m.get_mosaic(1024)
    plt.figure()
    plt.imshow(mosaic,interpolation='none',cmap='gray')
    plt.title('%0.5f,%0.1f'%(cp,cfs))
plt.show()
