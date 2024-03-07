"""
Some utility functions adapted from blog post on Turing Patterns (http://www.degeneratestate.org/posts/2017/May/05/turing-patterns/, accessed on 23/05/2019).
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

class BaseStateSystem:
    """
    Base object for "State System".

    We are going to repeatedly visualise systems which are Markovian:
    the have a "state", the state evolves in discrete steps, and the next
    state only depends on the previous state.

    To make things simple, I'm going to use this class as an interface.
    """
    def __init__(self):
        raise NotImplementedError()

    def initialise(self):
        raise NotImplementedError()

    def initialise_figure(self):
        fig, ax = plt.subplots()
        return fig, ax

    def update(self):
        raise NotImplementedError()

    def draw(self, ax):
        raise NotImplementedError()

    def plot_time_evolution(self, n_steps=30,xLabel="x-Achse", yLabel="y-Achse"):
        """
        Creates an animation from the time evolution of a basic state syste.
        """
        self.initialise()
        fig, ax = self.initialise_figure()
        
        def step(t):
            self.t = t
            if t == 0:
                self.initialise()
                self.draw(ax,xLabel, yLabel, n_steps)
                self.update()
            else:
                self.draw(ax,xLabel, yLabel, n_steps)
                self.update()
            

        anim = animation.FuncAnimation(fig, step, frames=np.arange(n_steps), interval=20)
        plt.close()
        return anim
        #anim.save(filename=filename, dpi=60, fps=10, writer='pillow')
                
    def plot_evolution_outcome(self, filename, n_steps):
        """
        Evolves and saves the outcome of evolving the system for n_steps
        """
        self.initialise()
        fig, ax = self.initialise_figure()
        
        for _ in range(n_steps):
            self.update()

        self.draw(ax)
        fig.savefig(filename)
        #plt.close()
