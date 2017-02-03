import numpy as np
from matplotlib import pyplot as plt
import os,sys
import glob
from data_store import H5

for f in glob.glob('./histories/*.hdf5'):
    h5 = H5(f)
    ages = []
    for k in h5.keys():
        try:
            ages.append(int(k))
        except:
            pass
    age_max = np.max(ages)
    plt.figure()
    plt.imshow(h5.get('%06d/mosaic'%age_max),interpolation='none',cmap='gray')
    plt.colorbar()

plt.show()
