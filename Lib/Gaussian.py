import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import copy

class Gaussian:
    def __init__(self, mu, Sigma):
        self.m = mu
        self.dim = np.size(mu)
        self.S = Sigma

        self.d = self.S
        self.iS = 1/self.S
        self.ct = 1/(2*np.pi*self.d)**0.5

    def __add__(self, g2):
        if isinstance(g2, Gaussian):
            return Gaussian(self.m+g2.m, self.S+g2.S)
        elif isinstance(g2, int) or isinstance(g2, float):
            return self.Offset(g2)

    def Offset(self, o):
        return Gaussian(self.m+o, self.S)

    def Value(self, x):
        """
        :param x: a given set of points where the Gaussian evaluated
        :return: Evaluation of a Gaussian
        """
        l = np.size(x)
        Sn = -0.5 * self.iS

        # v = np.zeros((1, l))
        # if l == 1:
        m = x - self.m
        return self.ct * np.exp(m*Sn*m)
        # else:
        #     for i in range(l):
        #         m = x[i] - self.m
        #         v[i] = self.ct * np.exp(m.T@Sn@m)
        #     return v

    def rand(self, n=1):
        """
        Generates a random point following the given Gaussian distribution
        """
        if self.d == 0:
            pass
        else:
            try:
                L = np.linalg.cholesky(self.S).T.conj()
            except:
                L = np.linalg.cholesky([[self.S]])[0][0]
            A_temp = np.random.normal(size=(self.dim, n))
            try:
                A = L @ A_temp
                r = matlib.repmat(self.m, 1, n) + A
            except:
                A = L * A_temp[0][0]
                r = self.m + A

            return r

    def Crop(self, Sp):
        """
        Forces the Gaussian mean to be in a given space.

        Crops the mean of a given Gaussian so that is is inside the
        continuous sub-space defined by Sp.
        :param Sp:
        :return:
        """
        gOut = copy.deepcopy(self)
        gOut.m = Sp.Crop(self.m)
        return gOut

    def GaussianKL(self, g2):
        """
        Computes the KL distance between two Gaussians.
        """
        assert (self.dim == g2.dim), "Gaussians' dimensions don't match in GaussianKL"
        m12 = g2.m - self.m
        if self.dim == 1:
            return (np.log(g2.d/self.d) + g2.iS*self.S + m12*g2.iS*m12 - self.dim)/2
        else:
            return (np.log(g2.d / self.d) + np.trace(g2.iS @ self.iS) + m12.T @ g2.iS @ m12 - self.dim) / 2


    def Product(self, g2):
        iS = self.iS + g2.iS

        if self.dim == 1:
            R = np.linalg.cholesky([[iS]])[0][0]
            iR = 1/R
            S = iR*iR
            d = self.ProductNormFactor(g2)
            return Gaussian(S*(self.iS*self.m+g2.iS*g2.m), S), d
        else:
            R = np.linalg.cholesky(iS).T.conj()

    def ProductNormFactor(self, g2):
        """
        Normalization constant of the product of two Gaussians.
        """
        return Gaussian(g2.m, self.S+g2.S).Value(self.m)

    def plot(self):
        if self.dim ==1:
            r = 3*self.S**.5
            X = np.linspace(self.m-r, self.m+r)
            Y = self.Value(X)
            plt.plot(X, Y)
            plt.show()






