import numpy as np
import copy
import matplotlib.pyplot as plt
from Lib.utils import *
from Lib.Gaussian import Gaussian

class GMixture:
    def __init__(self, weights=None, g=None, gm=None):
        """

        :param weights: has to be numpy array
        :param g:
        :param gm:
        """
        if gm is not None:
            self.w = gm.w
            self.g = gm.g
            self.n = gm.n
        else:
            n1 = np.size(weights)
            n2 = len(g)
            assert (n1 == n2), "Size missmatch in GMixture constructor"

            self.w = np.squeeze(weights)
            if np.size(self.w) == 1:
                self.w = self.w.reshape(1)
            self.g = g
            self.n = n1

    def __add__(self, other):
        # WARNING: scalar addition instead of list concatenation
        return GMixture(np.hstack((self.w, other.w)), self.g+other.g)

    def __mul__(self, gm2):
        if isinstance(gm2, GMixture):
            return self.Product(gm2)
        else:
            gm = copy.deepcopy(self)
            gm.w = gm2 * gm.w
            return gm

    def __truediv__(self, ct):
        return self*(1/ct)

    def Mean(self):
        """ Returns the weighted average of the mean of the components in the input Gaussian mixture."""
        if self.n > 0:
            m = self.w[0] * self.g[0].m
            for i in range(1, self.n):
                m += self.w[i] * self.g[i].m
            return m
        # elif self.n == 1:
        #     return self.w*self.g[0].m
        else:
            return 0

    def Covariance(self):
        """ Average covariance of a Gaussian mixture. """
        if self.n > 0:
            C = self.w[0]*self.g[0].S
            for i in range(1, self.n):
                C += self.w[i] * self.g[i].S
            return C
        # elif self.n == 1:
        #     return self.w * self.g[0].S
        else:
            return 0

    def Value(self, x):
        v = 0
        for i in range(self.n):
            v += self.w[i]*self.g[i].Value(x)
        return v

    def Product(self, gm2):
        n = self.n * gm2.n
        w = np.zeros((1, n))
        g = []
        k = 0

        for i in range(self.n):
            for j in range(gm2.n):
                gk, d = self.g[i].Product(gm2.g[j])
                w[:, k] = self.w[i] * gm2.w[j] * d
                g.append(gk)
                k += 1

        return GMixture(w, g)

    def ProductInt(self, gm2):
        """ Product and marginalization of two GMixtures. """
        c = 0
        for i in range(self.n):
            m1 = self.g[i].m
            S1 = self.g[i].S
            val = []
            for x in gm2.g:
                ig = Gaussian(x.m, S1+x.S)
                val.append(ig.Value(m1))
            c += self.w[i] * np.sum(gm2.w * np.array(val))
        return c

    def Compress(self, m):
        """
        Gaussian mixture compression
        :param m:
        :return:
        """
        if self.n <= m or m==0:
            return self
        else:
            gmN = self.Normalize()

            # Remove non-important components (to speed up next step)
            #   - gmLC is the sub-set of gmN with relevant components
            #   - mapLC is the index of the elements of gmN used to form gmLC
            gmLC, mapLC = gmN.RemoveSmallComponents(0.1 / gmN.n)

            # Use the Goldberger and Roweis compression
            #   - gmC is the compressed mixture
            #   - mapC are pointers from eleements in gmLC to elements in gmC
            #     (this allow to identify the elements in gmLC aggregated to form
            #      an element in gmC)
            gmC, mapC = (gmLC.Normalize()).CompressGR(m, 1e-5, 50)

            gmC.w = np.zeros(gmC.n)
            for i in range(gmC.n):
                ndx = np.where(mapC == i)[0]
                # Original code self.w
                gmC.w[i] = np.sum(self.w[mapLC[ndx]])
                # gmC.w[i] = np.sum(gmN.w[mapLC[ndx]])

#            gmC, map = gmC.RemoveSmallComponents(1e-6)

            return gmC

    def CompressGR(self, m, epsilon, MaxIterations):
        """
        Gaussian mixture compression using the Goldberger and Roweis method.
        :param m: The maximum number of components in  the output compressed
        normalized mixture.
        :param epsilon: Convergence criterion. If the relative error between two
        iterations changes less than epsilon we stop the process.
        :param MaxIterations:
        :return:
        gmC: The output normalized mixture.
        map: Map between components in the input mixtures and those in the
        output one
        """
        if self.n <= m:
            gmC = copy.deepcopy(self)
            map = np.arange(self.n)
        else:
            gmC = self.GetLargestComponents(m)
            d = np.inf
            map = np.arange(self.n)
            iteration = 1
            stop = False
            while not stop:
                for i in range(self.n):
                    f = self.g[i]
                    kl = []
                    for j in range(gmC.n):
                        kl.append(f.GaussianKL(gmC.g[j]))
                    map[i] = np.argmin(kl)

                for j in range(m):
                    ndx = np.where(map == j)[0]
                    sw = np.sum(self.w[ndx])
                    if sw > 0:
                        gmC.w[j] = sw
                        g = []
                        for k in ndx:
                            g.append(copy.deepcopy(self.g[k]))
                        gmC.g[j] = GMixture(self.w[ndx]/sw, g).FuseComponents()
                    else:
                        # None of the components of gm is close to this component of gmC
                        # Replace with one random component from gm
                        m1 = int(np.ceil(np.random.rand()*self.n-1))
                        gmC.w[j] = self.w[m1]
                        gmC.g[j] = self.g[m1]
                gmC = gmC.Normalize()

                d1 = d
                ds = np.zeros(self.n)
                for i in range(self.n):
                    ds[i] = self.g[i].GaussianKL(gmC.g[map[i]])
                d = ds @ self.w

                iteration += 1
                stop = (d < epsilon) or (np.abs(d1-d)/d < epsilon)

        return gmC, map

    def FuseComponents(self):
        """Fuses a Gaussian mixture into a single Gaussian."""
        m = self.Mean()
        S = self.Covariance()
        for i in range(self.n):
            v = self.g[i].m - m
            if self.g[i].dim == 1:
                dS = v*v
            else:
                dS = v@v.T
            # try:
            S += self.w[i] * dS
            # except:
            #     S += self.w * dS
        return Gaussian(m, S)

    def GetLargestComponents(self, m):
        """a Gaussian mixture formed by the 'm' elements of the input mixture with larger weight. """
        if self.n <= m:
            return copy.deepcopy(self)
        else:
            # Sort weights in descending order
            idx = np.argsort(self.w)[::-1][:m]
            nw = self.w[idx]
            g = []
            for i in idx:
                g.append(copy.deepcopy(self.g[i]))
            return GMixture(nw/np.sum(nw), g)

    def RemoveSmallComponents(self, t):
        """
        Eliminates the small components of a GMixture.
        :param t: Weight threshold. Components with a weight with absolute value
        below this threshold are removed.
        :return:
        gmOut: The mixture resulting from removing the small components.
        map: Vector indicating which elements in the input mixture are used
        in the output one (map(i)=j if the  i-th component of the output
        mixture is the j-th component of the input mixture).
        """
        map = np.where(np.abs(self.w) > t)[0]
        g = []
        for i in map:
            g.append(copy.deepcopy(self.g[i]))
        gmOut = GMixture(self.w[map], g)
        return gmOut, map

    def Normalize(self):
        w = np.abs(self.w)
        return GMixture(w/np.sum(w), self.g)

    def rand(self, n=1):
        """
        Generates random points on a GMixture
        :param n: the number of samples to drawn.
        :return:
        """
        if self.n == 0:
            return 0

        dim = self.g[1].dim
        v = np.zeros((dim, n))
        for i in range(n):
            v[:, i] = self.g[RandVector(self.w)].rand(1)

        return v

    def Compose(self, g):
        """
        Composes a Gaussian mixture with a Gaussian.
        Convolutes all the components of a Gaussian mixture with a given
        gaussian g.
        :param g:
        :return:
        """
        gmOut = copy.deepcopy(self)
        for i in range(self.n):
            gmOut.g[i] = g + self.g[i]
        return gmOut

    def Crop(self, Sp):
        """
        Crops all the elements in the mixture so that their mean is inside the
        given continuous space Sp.
        :param Sp:
        :return:
        """
        c = []
        for i in range(self.n):
            c.append(self.g[i].Crop(Sp))
        return GMixture(self.w, c)

    def Distance(self, gm2):
        """
        Approximated KL distance between Gaussian mixtures.
        """
        if self.n == 0 and gm2.n == 0:
            d = np.inf
        else:
            d = 0
            for i in range(self.n):
                g1 = self.g[i]
                kl = []
                for j in range(len(gm2.g)):
                    kl.append(g1.GaussianKL(gm2.g[j]))
                d += self.w[i]*np.min(kl)

        return d

    def plot_(self, s=None):
        """
        :param s: the underlying state
        """
        if self.n > 0:
            if self.g[0].dim == 1:
                l = np.zeros(self.n)
                u = np.zeros(self.n)
                for i in range(self.n):
                    c = self.g[i].S**.5
                    m = self.g[i].m
                    l[i] = m - 3*c
                    u[i] = m + 3*c
                mi = np.min(l)
                ma = np.max(u)

                n = 1000
                X = np.linspace(mi, ma, n)
                Y = self.Value(X)
                plt.plot(X, Y)
                if s is not None:
                    plt.title("s=" + str(s))

    def plot(self, s=None):
        self.plot_(s)
        plt.show()







