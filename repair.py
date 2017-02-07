from data_store import H5
import glob,sys
import numpy as np

flist = glob.glob('./histories/*.hdf5')

print flist

for f in flist:
    h5 = H5(f)
    
    if False:
        N_cones = h5.get('/params/N_cones').value
        I =  (10.0 + np.random.randn(N_cones)*2).clip(1.0,np.inf)
        h5.delete('/inentsities')
        h5.put('/intensities',I)

    if False:
        intsig = .08/(2.0*np.sqrt(2.0*np.log(2)))
        h5.delete('/params/intensity_fwhm_deg')
        h5.put('/params/intensity_sigma',intsig)

