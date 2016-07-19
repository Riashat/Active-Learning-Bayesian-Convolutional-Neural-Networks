import numpy as np

import math

import theano

import theano.tensor as T
import sys
# sys.path.append('../libraries/tools')
# from system_tools import *


class EQ_kernel:

    def compute_kernel_numpy(self, lls, lsf, x, z):
        ls2 = np.exp(2.0*lls)
        sf2 = np.exp(2.0*lsf)
        diff = (x[:, None, :] - z[None, :, :])
        diff2 = diff**2 / ls2
        r2 = diff2.sum(2)
        k = sf2 * np.exp(-0.5*r2)
        dklsf = 2*k
        dklls = k*np.transpose(diff2, [2, 0, 1])
        dkx = - k*np.transpose(diff/ls2, [2, 0, 1])
        return k, dklls, dklsf, dkx

    def compute_psi0_numpy(self, lls, lsf, xmean, xvar):
        sf2 = np.exp(2.0*lsf)
        return sf2

    def compute_psi1_numpy(self, lls, lsf, xmean, xvar, z):
        ls2 = np.exp(2.0*lls)
        sf2 = np.exp(2.0*lsf)
        ls2pxvar = ls2 + xvar
        constterm1 = ls2 / ls2pxvar
        constterm2 = np.prod(np.sqrt(constterm1))
        r2_psi1 = ((xmean - z[None, :, :])**2.0 / ls2pxvar)\
            .sum(2)
        psi1 = sf2*constterm2*np.exp(-0.5*r2_psi1)
        return psi1

    def compute_psi2_numpy(self, lls, lsf, xmean, xvar, z):
        ls2 = np.exp(2.0*lls)
        sf2 = np.exp(2.0*lsf)
        ls2p2xvar = ls2 + 2.0*xvar
        constterm1 = ls2 / ls2p2xvar
        constterm2 = np.prod(np.sqrt(constterm1))
        z1mz2 = z[:, None, :] - z[None, :, :]
        expo1 = -z1mz2**2 / (4.0*ls2)
        z1pz2 = z[:, None, :] + z[None, :, :]
        z1pz2mx = z1pz2 - 2.0*xmean
        expo2 = -z1pz2mx**2.0 / (4.0*ls2p2xvar)
        expoterm = np.exp((expo1 + expo2).sum(2))
        psi2 = sf2**2.0 * constterm2 * expoterm
        return psi2

    def compute_kernel_theano(self, lls, lsf, x, z):
        ls2 = T.exp(2.0*lls)
        sf2 = T.exp(2.0*lsf)
        r2 = ((x[:, None, :] - z[None, :, :])**2 / ls2).sum(2)
        k = sf2[0] * T.exp(-0.5*r2)
        return k

    def compute_psi0_theano(self, lls, lsf, xmean, xvar):
        sf2 = T.exp(2.0*lsf)
        return sf2[0]

    def compute_psi1_theano(self, lls, lsf, xmean, xvar, z):
        ls2 = T.exp(2.0*lls)
        sf2 = T.exp(2.0*lsf)
        ls2pxvar = ls2 + xvar
        constterm1 = ls2 / ls2pxvar
        constterm2 = T.prod(T.sqrt(constterm1))
        r2_psi1 = ((xmean - z[None, :, :])**2.0 / ls2pxvar).sum(2)
        psi1 = sf2[0]*constterm2*T.exp(-0.5*r2_psi1)
        return psi1

    def compute_psi2_theano(self, lls, lsf, xmean, xvar, z):
        ls2 = T.exp(2.0*lls)
        sf2 = T.exp(2.0*lsf)
        ls2p2xvar = ls2 + xvar + xvar
        constterm1 = ls2 / ls2p2xvar
        constterm2 = T.prod(T.sqrt(constterm1))
        z1mz2 = z[:, None, :] - z[None, :, :]
        a = -z1mz2**2
        b = (4.0*ls2)
        expo1 = a / b
        z1pz2 = z[:, None, :] + z[None, :, :]
        z1pz2mx = z1pz2 - xmean - xmean
        expo2 = -z1pz2mx**2.0 / (4.0*ls2p2xvar)
        expoterm = T.exp((expo1 + expo2).sum(2))
        psi2 = sf2[0]**2.0 * constterm2 * expoterm
        return psi2