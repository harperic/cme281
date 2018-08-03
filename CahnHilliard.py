# For ease of use and simplicity of display, I have
# create a separate file that will contain all the
# Cahn-Hilliard math. This also shows how easy it
# is to import a python function from another file

import numpy as np

class CH(object):
    def __init__(self, nx, ny, dx, t0, tf, dt):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.nSteps = (tf - t0) / dt
        # set default model params
        self.W = 1.0
        self.eps = 0.1
        self.M = 1.0
        self.C0avg = 0.5
        self.C0amp = 0.1
        # init a random background

        self.initRandom()

    def initRandom(self):
        self.C = (2 * np.random.rand(self.nx, self.ny) - 1) * self.C0amp + self.C0avg

    def delF(self, F):
        """
        Compute the Laplacian of the concentration as required
        by the Cahn-Hilliard equation. In this case
        We use a "stencil" or "kernel" to perform
        the operation. Note this automatically handles
        the periodic boundary condition.
        :param F: Discrete concentration (image)
        :param dx: pixel size of F (assumed to be square
        :type F: numpy array
        :type dx: float
        """
        # np.roll (circshift in MATLAB) will roll an array
        # along the specified axis, keeping boundaries
        # periodic
        return (-4*F +
                np.roll(F, 1, axis=0) +
                np.roll(F, -1, axis=0) +
                np.roll(F, 1, axis=1) +
                np.roll(F, -1, axis=1)) / (self.dx**2)

    def nextC(self, C):
        """
        Compute the next concentration
        """
        print(C)
        lapC = self.delF(C)
        print(lapC)
        dfdc = (self.W / 2.0) * np.multiply(np.multiply(C, 1.0 - C), 1.0 - 2.0 * C)
        mu = dfdc - (self.eps**2) * lapC
        lapMu = self.delF(mu)
        dcdt = self.M * lapMu
        self.C += dcdt * self.dt
        # self.C = self.initRandom()
