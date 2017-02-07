import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys
from gaze import Gaze

class Raster:

    # This class needs an object to be imaged, and a Gaze object to move
    # it.

    def __init__(self,mosaic,subtense):
        self.subtense = subtense
        g = Gaze(drift_relaxation_rate=2.5,drift_potential_slope=1.0,saccade_potential_slope=2.0,fractional_saccade_activation_threshold=2.0,image=mosaic,image_subtense=subtense)
        for k in range(10000):
            if k%100==0:
                print k
                g.step(True)
                plt.cla()
                g.history.plot()
                plt.pause(.001)
            else:
                g.step()

        plt.show()


if __name__=='__main__':

    im = np.load('./images/mosaic.npy')

    r = Raster(im,1.0)
