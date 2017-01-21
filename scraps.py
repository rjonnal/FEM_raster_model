class Retina0:

    def __init__(self,x1=-0.5,x2=0.5,y1=-0.5,y2=0.5,pixel_size=1e-2):
        self.x1 = min(x1,x2)
        self.x2 = max(x1,x2)
        self.y1 = min(y1,y2)
        self.y2 = max(y1,y2)
        self.xmid = (self.x1+self.x2)/2.0
        self.ymid = (self.y1+self.y2)/2.0
        self.dy = self.y2-self.y1
        self.dx = self.x2-self.x1
        
        self.pixel_size = pixel_size

        # central field
        self.cones = []
        self.age = 0
        self.clims = None

    def add(self):
        r = min(self.dy,self.dx)/2.0*.95
        x = np.random.rand()*self.dx+self.x1
        y = np.random.rand()*self.dy+self.y1
        print 'Adding cone at %0.1f,%0.1f'%(x,y)
        self.cones.append(Cone(x,y))

    def compute_central_field(self,xx,yy):
        return np.sqrt(xx**2+yy**2)*1


    def compute_total_field0(self,xx,yy):
        f = self.compute_central_field(xx,yy)
        for idx,c in enumerate(self.cones):
            f = f + c.field(xx,yy)
        return f

    def compute_total_field(self,xx,yy):
        
        f = self.compute_central_field(xx,yy)

        cx = np.array([c.x for c in self.cones])
        cy = np.array([c.y for c in self.cones])

        xstack = np.array([xx]*len(cx))
        ystack = np.array([yy]*len(cy))
        if len(xx.shape)==2:
            xstack = np.transpose(xstack,(1,2,0))
            ystack = np.transpose(ystack,(1,2,0))
        else:
            xstack = xstack.T
            ystack = ystack.T
            
        xstack = xstack - cx
        ystack = ystack - cy
        fstack = np.exp(-100*(xstack**2+ystack**2))
        f = np.sum(fstack,axis=len(fstack.shape)-1)
        return f

    def step(self):
        self.age = self.age + 1
        for c in self.cones:
            print c.index
            c.step(self)
        print 'Age: %d'%self.age

    def show(self):
        field = self.compute_total_field()
        if self.clims is None:
            self.clims = (field.min()*1.1,field.max()*.5)
        plt.clf()
        plt.imshow(field,extent=[xx.min(),xx.max(),yy.min(),yy.max()],interpolation='bilinear')
        plt.colorbar()
        plt.autoscale(False)
        cx = [c.x for c in self.cones]
        cy = [c.y for c in self.cones]
        plt.plot(cx,cy,'k.')

    def save(self,tag):
        xfn = '%s_x.npy'%tag
        yfn = '%s_y.npy'%tag
        np.save(xfn,[c.x for c in self.cones])
        np.save(yfn,[c.y for c in self.cones])

        rec = [self.x1,self.x2,self.y1,self.y2]
        rfn = '%s_bounds.npy'%tag
        np.save(rfn,rec)

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
        
