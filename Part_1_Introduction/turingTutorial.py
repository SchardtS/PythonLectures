"""
Adapted from blog post on Turing Patterns (http://www.degeneratestate.org/posts/2017/May/05/turing-patterns/, accessed on 23/05/2019).
"""

import matplotlib.pyplot as plt
import numpy as np

from tutils import BaseStateSystem
from IPython.display import HTML

def laplacian1D(a, dx):
    return (
        - 2 * a
        + np.roll(a,1,axis=0) 
        + np.roll(a,-1,axis=0)
    ) / (dx ** 2)

def laplacian2D(a, dx):
    return (
        - 4 * a
        + np.roll(a,1,axis=0) 
        + np.roll(a,-1,axis=0)
        + np.roll(a,+1,axis=1)
        + np.roll(a,-1,axis=1)
    ) / (dx ** 2)
    
class OneDimensionalDiffusionEquation(BaseStateSystem):
    def __init__(self, D):
        self.D = D
        self.width = 1000
        self.dx = 10 / self.width
        self.dt = 0.9 * (self.dx ** 2) / (2 * D)
        self.steps = int(0.1 / self.dt)
        
    def initialise(self):
        self.t = 0 
        self.X = np.linspace(-5,5,self.width)
        self.a = np.exp(-self.X**2)
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):      
        La = laplacian1D(self.a, self.dx)
        delta_a = self.dt * (self.D * La)       
        self.a += delta_a
        
    def draw(self, ax, xLabel="x-Achse", yLabel="y-Achse",n_steps=100):
        ax.clear()
        ax.plot(self.X,self.a, color="r")
        ax.set_ylim(0,1)
        ax.set_xlim(-5,5)
        ax.set_title("t = {:.2f}".format(self.t))
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        
        
class ReactionEquation(BaseStateSystem):
    def __init__(self, Ra, Rb):
        self.Ra = Ra
        self.Rb = Rb
        self.dt = 0.01
        self.steps = int(0.1 / self.dt)
        
    def initialise(self):
        self.t = 0
        self.a = 0.1
        self.b = 0.7
        self.Ya = []
        self.Yb = []
        self.X = []
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):      
        delta_a = self.dt * self.Ra(self.a,self.b)      
        delta_b = self.dt * self.Rb(self.a,self.b)      

        self.a += delta_a
        self.b += delta_b
        
    def draw(self, ax, xlabel, ylabel, n_steps):
        ax.clear()
        
        self.X.append(self.t)
        self.Ya.append(self.a)
        self.Yb.append(self.b)

        ax.plot(self.X,self.Ya, color="r", label="A")
        ax.plot(self.X,self.Yb, color="b", label="B")
        ax.legend()
        
        ax.set_ylim(0,1)
        ax.set_xlim(0,n_steps)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
def random_initialiser(shape):
    return(
        np.random.normal(loc=0, scale=0.05, size=shape),
        np.random.normal(loc=0, scale=0.05, size=shape)
    )

class OneDimensionalRDEquations(BaseStateSystem):
    def __init__(self, Da, Db, Ra, Rb,
                 initialiser=random_initialiser,
                 width=100, dx=1, 
                 dt=0.001, steps=100):
        
        self.Da = Da
        self.Db = Db
        self.Ra = Ra
        self.Rb = Rb
        
        self.initialiser = initialiser
        self.width = width
        self.dx = dx
        self.dt = dt
        self.steps = steps
        
    def initialise(self):
        self.t = 0
        self.a, self.b = self.initialiser(self.width)
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):
        
        # unpack so we don't have to keep writing "self"
        a,b,Da,Db,Ra,Rb,dt,dx = (
            self.a, self.b,
            self.Da, self.Db,
            self.Ra, self.Rb,
            self.dt, self.dx
        )
        
        La = laplacian1D(a, dx)
        Lb = laplacian1D(b, dx)
        
        delta_a = dt * (Da * La + Ra(a,b))
        delta_b = dt * (Db * Lb + Rb(a,b))
        
        self.a += delta_a
        self.b += delta_b
        
    def draw(self, ax, xlabel="x-Achse", ylabel="y-Achse",n_steps=100):
        ax.clear()
        ax.plot(self.a, color="r", label="A")
        ax.plot(self.b, color="b", label="B")
        ax.legend()
        ax.set_ylim(-1,1)
        ax.set_title("t = {:.2f}".format(self.t))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
class TwoDimensionalRDEquations(BaseStateSystem):
    def __init__(self, Da, Db, Ra, Rb,
                 initialiser=random_initialiser,
                 width=100, height=100,
                 dx=1, dt=0.001, steps=100):
        
        self.Da = Da
        self.Db = Db
        self.Ra = Ra
        self.Rb = Rb

        self.initialiser = initialiser
        self.width = width
        self.height = height
        self.shape = (width, height)
        self.dx = dx
        self.dt = dt
        self.steps = steps
        
    def initialise(self):
        self.t = 0
        self.a, self.b = self.initialiser(self.shape)
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()

    def _update(self):
        
        # unpack so we don't have to keep writing "self"
        a,b,Da,Db,Ra,Rb,dt,dx = (
            self.a, self.b,
            self.Da, self.Db,
            self.Ra, self.Rb,
            self.dt, self.dx
        )
        
        La = laplacian2D(a, dx)
        Lb = laplacian2D(b, dx)
        
        delta_a = dt * (Da * La + Ra(a,b))
        delta_b = dt * (Db * Lb + Rb(a,b))
        
        self.a += delta_a
        self.b += delta_b
        
    def draw(self, ax, xlabel= "", ylabel="",n_steps=100):
        ax[0].clear()
        ax[1].clear()

        ax[0].imshow(self.a, cmap='jet')
        ax[1].imshow(self.b, cmap='brg')
        
        #ax[0].grid(b=False)
        #ax[1].grid(b=False)
        
        ax[0].grid()
        ax[1].grid()
        
        ax[0].set_title("A, t = {:.2f}".format(self.t))
        ax[1].set_title("B, t = {:.2f}".format(self.t))
        
    def initialise_figure(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
        return fig, ax        
        

