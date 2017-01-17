    def step0(self,power=1):

        # increment the landscape
        self.landscape.step()

        # implement a self-avoiding random walk
        # self-avoidance is implemented using a continuous
        # version of discrete swamp walk described by
        # Engbert et al., PNAS, 2011
        # [http://www.pnas.org/content/108/39/E765.full]

        # compute points along ring r distance from current location
        xx,yy = self.get_ring(self.step_size)

        # get the depths along this ring
        depths = self.landscape.evaluate(xx,yy)

        # raise the depths to positive values
        #depths = depths - np.min(depths)

        #plt.plot(depths)
        #plt.show()
        
        mode='stochastic'

        if mode=='stochastic':
            # compute a CDF using a power of the resulting positive depths
            # check for corner case where depths are all zero, and assign
            # a same-size vector of ones, i.e. all directions equally probable
            depths = np.max(depths)-depths
            
            if any(depths):
                weights = depths**power/np.sum(depths**power)
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

        if mode=='deterministic':
            weights = depths
            winners = np.where(depths==np.min(depths))[0]
            if len(winners)>1:
                winner = winners[np.random.randint(len(winners))]
            else:
                winner = winners[0]
                               
            
        x = xx[winner]
        y = yy[winner]

        movement_type = 'd'
        # check for a saccade:
        activation = self.landscape.evaluate(x,y)
        if activation>self.saccade_threshold:
            # evaluate over a fine grid +/- 2 deg, centered about origin
            rad = 2
            N = 256
            def rep(x,y):
                repx = x/float(rad)+N/2.0
                repy = y/float(rad)+N/2.0
                return repx,repy
            
            xx,yy = np.meshgrid(np.linspace(-rad,rad,N),np.linspace(-rad,rad,N))
            surf0 = self.landscape.evaluate(xx,yy)

            # generate saccade potential: bias toward horizontal and vertical
            # movements
            xx = xx - self.x
            yy = yy - self.y
            sacc_surf = xx**2*yy**2*self.saccade_potential_slope
            surf = surf0 + sacc_surf

            plot_saccade_potentials = True
            if plot_saccade_potentials:
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(surf0)
                ymax,xmax=np.where(surf==np.min(surf))
                plt.autoscale(False)
                plt.plot(*rep(self.x,self.y),marker='o',markeredgecolor='w',markerfacecolor='g',markersize=20)
                plt.plot(xmax,ymax,marker='o',markeredgecolor='w',markerfacecolor='r',markersize=20)
                plt.title('age = %d'%(self.landscape.age))
                plt.colorbar()
                plt.subplot(1,2,2)
                plt.imshow(sacc_surf)
                ymax,xmax=np.where(surf==np.min(surf))
                plt.autoscale(False)
                plt.plot(*rep(self.x,self.y),marker='o',markeredgecolor='w',markerfacecolor='g',markersize=20)
                plt.plot(xmax,ymax,marker='o',markeredgecolor='w',markerfacecolor='r',markersize=20)
                plt.title('age = %d'%(self.landscape.age))
                plt.colorbar()
                plt.show()
            
            ywin = np.where(surf==np.min(surf))[0][0]
            xwin = np.where(surf==np.min(surf))[1][0]
            y = yy[ywin,xwin]
            x = xx[ywin,xwin]
            movement_type = 's'
            
        #print 'Moving to %0.2f,%0.2f.'%(x,y)
        #plt.figure()
        #plt.plot(weights)
        #plt.plot(winner,weights[winner],'go')
        #plt.show()

        # move to new location and add a depression there
        self.x = x
        self.y = y
        self.x_path.append(x)
        self.y_path.append(y)
        self.history.add(x,y,movement_type)
        self.landscape.add(Feature(x,y,self.drift_relaxation_rate))
        
