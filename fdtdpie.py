#!/usr/bin/evn python3

import numpy as np
import cmasher as cmr
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm

class FD1Dfree():
    '''
    1D FDTD in free space

    Parameters
    ----------
    sim_params : dict
        Dictionary containing the following simulation parameters
        T : float
            Total simulation time in seconds
        L : float
            Length of the simulation domain in um
        c : float
            Speed of light in um/ps
        dx : float
            Spatial resolution in um
        Jsource : function
            Source function for the electric current density
        boundary : str
            Boundary condition to use. Options are 'metallic' or 'periodic'
    '''
    def __init__(self, sim_params):
        self.T  = sim_params['T']
        self.L  = sim_params['L']
        self.c  = sim_params['c']
        self.dx = sim_params['dx']
        self.Jsource = sim_params['Jsource']
        self.boundary = sim_params['boundary']

        self.dt = self.dx/self.c
        self.N  = int(self.T/self.dt)
        self.n  = int(self.L/self.dx)
        self.cNum = self.dt/self.dx
        self.yeeChunks = 2 * self.n

        if self.yeeChunks % 2 != 0: 
            self.yeeChunks += 1

        if self.boundary == 'periodic':
            self.xcoords     = np.linspace(-self.L/2, self.L/2, self.yeeChunks)
            self.xcoords_Hz  = self.xcoords[1::2]
            self.xcoords_Ey  = self.xcoords[0::2]
        elif self.boundary == 'metallic':
            self.xcoords     = np.linspace(-self.L/2, self.L/2, self.yeeChunks)
            self.xcoords_Hz  = self.xcoords[1::2]
            self.xcoords_Ey  = self.xcoords[0::2]
        
        self.yeePrev     = np.zeros(self.yeeChunks)
        
        self.yeeNext     = np.zeros(self.yeeChunks)
        self.yeeHistory  = np.zeros((self.N,self.yeeChunks))
        self.t           = 0.
        self.times       = np.zeros(self.N)
        self.Ey          = np.zeros(self.yeeNext[::2].shape)
        self.Hz          = np.zeros(self.yeeNext[1::2].shape)
    
    def run_metallic(self):
        '''
        Run the simulation assuming a metallic boundary condition
        in which the tangential electric field at the boundaries
        is zero.
        '''
        print("Running simulation for {} frogsteps ...".format(self.N))
        for iter in tqdm(range(self.N)):
            # update Ey
            # collect the values of Hz
            self.Hz = self.yeeNext[1::2]
            # calculate the gradient of Hz
            Hzgradient    = np.diff(self.Hz, prepend=self.Hz[0])
            # determine the contribution to the change in Ey
            delta         = -self.cNum*Hzgradient
            # add the source term
            delta += -self.dt * self.Jsource(self.xcoords_Ey, self.t)
            self.yeeNext[::2]  += delta
            self.yeeNext[0] = 0
            self.yeeNext[-2] = 0

            # update Hz
            # collect the values of Ey
            self.Ey         = self.yeeNext[::2]
            # calculate the gradient of Ey
            Eygradient = np.diff(self.Ey, append=0)
            # determine the contribution to the change in Hz
            delta      = -self.cNum*Eygradient
            # update Hz with the calculated delta
            self.yeeNext[1::2] += delta
            
            # update the past field to be the current field
            self.yeePrev = self.yeeNext
            if self.yeeNext[-2] != 0:
                1/0
            if self.yeeNext[0] != 0:
                1/0
            # add the current field to the history books
            self.yeeHistory[iter] = self.yeeNext
            # increment time by dt
            self.times[iter] = self.t
            self.t += self.dt
        print("Simulation complete")
    
    def run_periodic(self):
        '''
        Run the simulation assuming periodic boundary conditions.
        '''
        print("Running simulation for {} frogsteps ...".format(self.N))
        for iter in tqdm(range(self.N)):
            # update Ey
            # query the values of H
            self.Hz              = self.yeeNext[1::2]
            # calculate the gradient of H
            Hzgradient           = np.diff(self.Hz,prepend=self.Hz[-1])
            # determine the contribution to the change in Ey
            delta                = -self.cNum * Hzgradient
            # add the source term
            delta     += -self.dt * self.Jsource(self.xcoords_Ey, self.t)
            self.yeeNext[::2]    = self.yeePrev[::2] + delta

            # update Hz
            # calculate the gradient of Ey
            self.Ey    = self.yeeNext[::2]
            Eygradient = np.diff(self.Ey, append=self.Ey[0])
            delta      = -self.cNum * Eygradient
            # update Hz with the calculated delta
            self.yeeNext[1::2]  = self.yeePrev[1::2] + delta

            # update the past field to be the current field
            self.yeePrev = self.yeeNext
            # add the current field to the history books
            self.yeeHistory[iter] = self.yeeNext
            # increment time by dt
            self.times[iter] = self.t
            self.t += self.dt
        print("Simulation complete")

    def run(self):
        if self.boundary == 'metallic':
            self.run_metallic()
        elif self.boundary == 'periodic':
            self.run_periodic()

    def get_field_history(self, field):
        if self.boundary == 'periodic':
            if field == 'Hz':
                return self.yeeHistory[:,1::2]
            elif field == 'Ey':
                return self.yeeHistory[:,::2] 
            else:
                raise ValueError("Invalid field to return")
        elif self.boundary == 'metallic':
            if field == 'Hz':
                return self.yeeHistory[:,1::2]
            elif field == 'Ey':
                return self.yeeHistory[:,0::2] 
            else:
                raise ValueError("Invalid field to return")

    def plot(self, plotField):
        '''
        Plots the requested field or fields.
        '''
        extent   = [-self.L/2, self.L/2, 
                        0, self.N*self.dt]
        if type(plotField) == str:
            plotData = self.get_field_history(plotField)
            minF = np.min(plotData)
            maxF = np.max(plotData)
            range = max(abs(minF), abs(maxF))
            plt.figure(figsize=(10,10))
            plt.imshow(plotData,
                    extent=extent,
                    cmap=cmr.wildfire,
                    vmin=-range,
                    vmax=range,
                    aspect=1,
                    origin='lower')
            plt.xlabel('x (μm)')
            plt.ylabel('t (μm/c)')
            plt.title(plotField)
            plt.colorbar()
            plt.show()
        elif type(plotField) == list:
            fig, ax = plt.subplots(ncols=len(plotField), nrows=1, figsize=(5*len(plotField), 10))
            for i, field in enumerate(plotField):
                plotData = self.get_field_history(field)
                minF = np.min(plotData)
                maxF = np.max(plotData)
                range = max(abs(minF), abs(maxF))
                implot = ax[i].imshow(plotData,
                        extent=extent,
                        cmap=cmr.wildfire,
                        vmin=-range,
                        vmax=range,
                        aspect=1,
                        origin='lower')
                ax[i].set_xlabel('x (μm)')
                ax[i].set_ylabel('t (μm/c)')
                ax[i].set_title(field)
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes("right", size="10%", pad=0.1)
                plt.colorbar(implot, cax=cax)
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("Invalid field to plot. It should be either a single string or a list of strings.")

class FD1Dvarep():
    '''
    1D FDTD with variable epsilon

    Parameters
    ----------
    sim_params : dict
        Dictionary containing the following simulation parameters
        T : float
            Total simulation time in seconds
        L : float
            Length of the simulation domain in um
        c : float
            Speed of light in um/ps
        dx : float
            Spatial resolution in um
        Jsource : function
            Source function for the electric current density
        epsilonFun: function
            Function that returns the value of the permitivity at a given point
        boundary : str
            Boundary condition to use. Options are 'metallic' or 'periodic'
    '''
    def __init__(self, sim_params):
        self.T  = sim_params['T']
        self.L  = sim_params['L']
        self.c  = sim_params['c']
        self.dx = sim_params['dx']
        self.Jsource = sim_params['Jsource']
        self.boundary = sim_params['boundary']
        self.epsilonFun = sim_params['epsilonFun']

        self.dt = self.dx/self.c
        self.N  = int(self.T/self.dt)
        self.n  = int(self.L/self.dx)
        self.cNum = self.dt/self.dx
        self.yeeChunks = 2 * self.n

        if self.yeeChunks % 2 != 0: 
            self.yeeChunks += 1

        if self.boundary == 'periodic':
            self.xcoords     = np.linspace(-self.L/2, self.L/2, self.yeeChunks)
            self.xcoords_Hz  = self.xcoords[1::2]
            self.xcoords_Ey  = self.xcoords[0::2]
        elif self.boundary == 'metallic':
            self.xcoords     = np.linspace(-self.L/2, self.L/2, self.yeeChunks)
            self.xcoords_Hz  = self.xcoords[1::2]
            self.xcoords_Ey  = self.xcoords[0::2]
        
        self.invEpsilon  = 1./self.epsilonFun(self.xcoords_Ey)
        self.yeePrev     = np.zeros(self.yeeChunks)
        
        self.yeeNext     = np.zeros(self.yeeChunks)
        self.yeeHistory  = np.zeros((self.N,self.yeeChunks))
        self.t           = 0.
        self.times       = np.zeros(self.N)
        self.Ey          = np.zeros(self.yeeNext[::2].shape)
        self.Hz          = np.zeros(self.yeeNext[1::2].shape)
    
    def run_metallic(self):
        '''
        Run the simulation assuming a metallic boundary condition
        in which the tangential electric field at the boundaries
        is zero.
        '''
        print("Running simulation for {} frogsteps ...".format(self.N))
        for iter in tqdm(range(self.N)):
            # update Ey
            # collect the values of Hz
            self.Hz = self.yeeNext[1::2]
            # calculate the gradient of Hz
            Hzgradient    = np.diff(self.Hz, prepend=self.Hz[0])
            # determine the contribution to the change in Ey
            delta         = -self.cNum*Hzgradient
            # add the source term
            delta += -self.dt * self.Jsource(self.xcoords_Ey, self.t)
            delta *= self.invEpsilon
            self.yeeNext[::2]  += delta
            # self.yeeNext[2] = (self.yeeNext[2]-self.yeeNext[0])/2
            self.yeeNext[0] = 0
            # self.yeeNext[-4] = (self.yeeNext[-4]-self.yeeNext[-2])/2
            self.yeeNext[-2] = 0

            # update Hz
            # collect the values of Ey
            self.Ey         = self.yeeNext[::2]
            # calculate the gradient of Ey
            Eygradient = np.diff(self.Ey, append=0)
            # determine the contribution to the change in Hz
            delta      = -self.cNum*Eygradient
            # update Hz with the calculated delta
            self.yeeNext[1::2] += delta
            
            # update the past field to be the current field
            self.yeePrev = self.yeeNext
            if self.yeeNext[-2] != 0:
                1/0
            if self.yeeNext[0] != 0:
                1/0
            # add the current field to the history books
            self.yeeHistory[iter] = self.yeeNext
            # increment time by dt
            self.times[iter] = self.t
            self.t += self.dt
        print("Simulation complete")
    
    def run_periodic(self):
        '''
        Run the simulation assuming periodic boundary conditions.
        '''
        print("Running simulation for {} frogsteps ...".format(self.N))
        for iter in tqdm(range(self.N)):
            # update Ey
            # query the values of H
            self.Hz              = self.yeeNext[1::2]
            # calculate the gradient of H
            Hzgradient           = np.diff(self.Hz,prepend=self.Hz[-1])
            # determine the contribution to the change in Ey
            delta                = -self.cNum * Hzgradient
            # add the source term
            delta     += -self.dt * self.Jsource(self.xcoords_Ey, self.t)
            delta     *= self.invEpsilon
            self.yeeNext[::2]    = self.yeePrev[::2] + delta

            # update Hz
            # calculate the gradient of Ey
            self.Ey    = self.yeeNext[::2]
            Eygradient = np.diff(self.Ey, append=self.Ey[0])
            delta      = -self.cNum * Eygradient
            # update Hz with the calculated delta
            self.yeeNext[1::2]  = self.yeePrev[1::2] + delta

            # update the past field to be the current field
            self.yeePrev = self.yeeNext
            # add the current field to the history books
            self.yeeHistory[iter] = self.yeeNext
            # increment time by dt
            self.times[iter] = self.t
            self.t += self.dt
        print("Simulation complete")

    def run(self):
        if self.boundary == 'metallic':
            self.run_metallic()
        elif self.boundary == 'periodic':
            self.run_periodic()

    def get_field_history(self, field):
        if self.boundary == 'periodic':
            if field == 'Hz':
                return self.yeeHistory[:,1::2]
            elif field == 'Ey':
                return self.yeeHistory[:,::2] 
            else:
                raise ValueError("Invalid field to return")
        elif self.boundary == 'metallic':
            if field == 'Hz':
                return self.yeeHistory[:,1::2]
            elif field == 'Ey':
                return self.yeeHistory[:,0::2] 
            else:
                raise ValueError("Invalid field to return")

    def plot_epsilon(self):
        plt.figure(figsize=(10,2))
        plt.plot(self.xcoords_Ey, self.epsilonFun(self.xcoords_Ey))
        plt.xlabel('x (μm)')
        plt.ylabel('ε')
        plt.title('ε(x)')
        plt.xlim(-self.L/2, self.L/2)
        plt.ylim(0, None)
        plt.show()

    def plot(self, plotField):
        '''
        Plots the requested field or fields.
        '''
        extent   = [-self.L/2, self.L/2, 
                        0, self.N*self.dt]
        if type(plotField) == str:
            plotData = self.get_field_history(plotField)
            minF = np.min(plotData)
            maxF = np.max(plotData)
            range = max(abs(minF), abs(maxF))
            plt.figure(figsize=(10,10))
            plt.imshow(plotData,
                    extent=extent,
                    cmap=cmr.wildfire,
                    vmin=-range,
                    vmax=range,
                    aspect=1,
                    origin='lower')
            plt.xlabel('x (μm)')
            plt.ylabel('t (μm/c)')
            plt.title(plotField)
            plt.colorbar()
            plt.show()
        elif type(plotField) == list:
            fig, ax = plt.subplots(ncols=len(plotField), nrows=1, figsize=(5*len(plotField), 10))
            for i, field in enumerate(plotField):
                plotData = self.get_field_history(field)
                minF = np.min(plotData)
                maxF = np.max(plotData)
                range = max(abs(minF), abs(maxF))
                implot = ax[i].imshow(plotData,
                        extent=extent,
                        cmap=cmr.wildfire,
                        vmin=-range,
                        vmax=range,
                        aspect=1,
                        origin='lower')
                ax[i].set_xlabel('x (μm)')
                ax[i].set_ylabel('t (μm/c)')
                ax[i].set_title(field)
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes("right", size="10%", pad=0.1)
                plt.colorbar(implot, cax=cax)
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("Invalid field to plot. It should be either a single string or a list of strings.")
