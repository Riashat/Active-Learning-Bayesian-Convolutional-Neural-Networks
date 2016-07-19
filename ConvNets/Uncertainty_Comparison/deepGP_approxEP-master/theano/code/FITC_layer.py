import sys
import math
import theano
import theano.tensor as T
import theano.tensor.nlinalg as Talg
import EQ_kernel
import numpy as np
# import numpy.linalg as npalg
import scipy.linalg as npalg
# from system_tools import *

class FITC_layer:
    def __init__(self, Ntrain, layer_size, no_pseudos):

        # layer parameters
        self.Din = layer_size[0]
        self.Dout = layer_size[1]
        self.M = no_pseudos
        self.Ntrain = Ntrain
        self.kern = EQ_kernel.EQ_kernel()
        self.jitter = 1e-6
        self.SEP_damping = 0.1*1.0/Ntrain

        # theano variables for the mean and covariance of q(u)
        self.mu = [
            theano.shared(
                value=np.zeros([self.M, 1]).astype(theano.config.floatX),
                name='mu_%d' % d, borrow=True
            )
            for d in range(self.Dout)
            ]
        self.Su = [
            theano.shared(
                value=np.zeros([self.M, self.M]).astype(theano.config.floatX),
                name='Su_%d' % d, borrow=True
            )
            for d in range(self.Dout)
            ]

        self.Suhatinv = [np.zeros([self.M, self.M]) for _ in range(self.Dout)]

        # theano variables for the cavity distribution
        self.muhat = [
            theano.shared(
                value=np.zeros([self.M, 1]).astype(theano.config.floatX),
                name='muhat_%d' % d, borrow=True
            )
            for d in range(self.Dout)
            ]
        self.Suhat = [
            theano.shared(
                value=np.zeros([self.M, self.M]).astype(theano.config.floatX),
                name='Suhat_%d' % d, borrow=True
            )
            for d in range(self.Dout)
            ]

        # theano variable for Kuuinv
        self.Kuuinv = [
            theano.shared(
                value=np.zeros([self.M, self.M]).astype(theano.config.floatX),
                name='Kuuinv',
                borrow=True
            ) for d in range(self.Dout)
        ]

        # numpy variable for Kuu and its gradients
        self.Kuu = [np.zeros([self.M, self.M]) for d in range(self.Dout)]
        self.dKuudls = [np.zeros([self.Din, self.M, self.M]) for d in range(self.Dout)]
        self.dKuudsf = [np.zeros([self.M, self.M]) for d in range(self.Dout)]
        self.dKuudzu = [np.zeros([self.Din, self.M, self.M]) for d in range(self.Dout)]
        self.dKuuinvdls = [np.zeros([self.Din, self.M, self.M]) for d in range(self.Dout)]
        self.dKuuinvdsf = [np.zeros([self.M, self.M]) for d in range(self.Dout)]
        self.dKuuinvdzu = [np.zeros([self.M, self.Din, self.M, self.M]) for d in range(self.Dout)]
        self.Kuuinv_val = [np.zeros([self.M, self.M]) for d in range(self.Dout)]

        # theano variables for the hyperparameters and inducing points
        ls_init = np.zeros([self.Din, ])
        sf_init = np.zeros([1, ])
        sn_init = np.zeros([1, ])
        zu_init = np.zeros([self.M, self.Din])
        self.ls = theano.shared(value=ls_init.astype(theano.config.floatX),
                                name='ls', borrow=True)
        self.sf = theano.shared(value=sf_init.astype(theano.config.floatX),
                                name='sf', borrow=True)
        self.sn = theano.shared(value=sn_init.astype(theano.config.floatX),
                                name='sn', borrow=True)
        self.zu = [
            theano.shared(
                value=zu_init.astype(theano.config.floatX),
                name='zu_%d' % d, borrow=True
            ) for d in range(self.Dout)
            ]

        # # and natural parameters
        self.theta_1_R = [np.zeros([self.M, self.M]) for d in range(self.Dout)]
        self.theta_2 = [np.zeros([self.M, 1]) for d in range(self.Dout)]
        self.theta_1 = [np.zeros([self.M, self.M]) for d in range(self.Dout)]


    def compute_kuu(self):
        ls_val = self.ls.get_value()
        sf_val = self.sf.get_value()
        for d in range(self.Dout):
            zu_val = self.zu[d].get_value()
            # print zu_val.shape
            # print self.M
            self.Kuu[d], self.dKuudls[d], self.dKuudsf[d], self.dKuudzu[d] = \
                self.kern.compute_kernel_numpy(ls_val, sf_val, zu_val, zu_val)
            self.Kuu[d] += np.eye(self.M)*self.jitter
            Kinv_val = npalg.inv(self.Kuu[d])
            self.Kuuinv_val[d] = Kinv_val
            self.Kuuinv[d].set_value(Kinv_val)

    def compute_cavity(self):
        # compute the leave one out moments
        factor = (self.Ntrain - 1.0) * 1.0 / self.Ntrain
        for d in range(self.Dout):
            # use numpy inverse
            self.Suhatinv[d] = self.Kuuinv_val[d] + factor * self.theta_1[d]
            ShatinvMhat = factor * self.theta_2[d]
            Shat = npalg.inv(self.Suhatinv[d])
            self.Suhat[d].set_value(Shat)
            self.muhat[d].set_value(np.dot(Shat, ShatinvMhat))

    def update_posterior(self):
        # compute the posterior approximation
        for d in range(self.Dout):
            # this uses numpy inverse
            Sinv = self.Kuuinv_val[d] + self.theta_1[d]
            SinvM = self.theta_2[d]
            S = npalg.inv(Sinv)
            self.Su[d].set_value(S)
            self.mu[d].set_value(np.dot(S, SinvM))

    def output_probabilistic(self, mx_previous, vx_previous):
        # create place holders
        mout = []
        vout = []

        # compute the psi terms
        psi0 = self.kern.compute_psi0_theano(
            self.ls, self.sf,
            mx_previous, vx_previous
        )

        for d in range(self.Dout):
            psi1 = self.kern.compute_psi1_theano(
                self.ls, self.sf,
                mx_previous, vx_previous, self.zu[d]
            )
            psi1psi1T =  T.outer(psi1, psi1.T)
            psi2 = self.kern.compute_psi2_theano(
                self.ls, self.sf,
                mx_previous, vx_previous, self.zu[d]
            )

            # precompute some terms
            psi1Kinv = T.dot(psi1, self.Kuuinv[d])
            Kinvpsi2 = T.dot(self.Kuuinv[d], psi2)
            Kinvpsi2Kinv = T.dot(Kinvpsi2, self.Kuuinv[d])
            vconst = T.exp(2.0 * self.sn) + (psi0 - Talg.trace(Kinvpsi2))
            mud = self.mu[d]
            Sud = self.Su[d]
            moutd = T.sum(T.dot(psi1Kinv, mud))
            mout.append(moutd)

            Splusmm = Sud + T.outer(mud, mud)
            voutd = vconst + Talg.trace(T.dot(Splusmm, Kinvpsi2Kinv)) - moutd ** 2
            vout.append(T.sum(voutd))

        return mout, vout

    def output_probabilistic_sep(self, mx_previous, vx_previous):
        # create place holders
        mout = []
        vout = []

        # compute the psi0 term
        psi0 = self.kern.compute_psi0_theano(
            self.ls, self.sf,
            mx_previous, vx_previous
        )
        for d in range(self.Dout):
            # compute the psi1 and psi2 term
            psi1 = self.kern.compute_psi1_theano(
                self.ls, self.sf,
                mx_previous, vx_previous, self.zu[d]
            )
            psi1psi1T =  T.outer(psi1, psi1.T)
            psi2 = self.kern.compute_psi2_theano(
                self.ls, self.sf,
                mx_previous, vx_previous, self.zu[d]
            )

            # precompute some terms
            psi1Kinv = T.dot(psi1, self.Kuuinv[d])
            Kinvpsi2 = T.dot(self.Kuuinv[d], psi2)
            Kinvpsi2Kinv = T.dot(Kinvpsi2, self.Kuuinv[d])
            vconst = T.exp(2 * self.sn) + (psi0 - Talg.trace(Kinvpsi2))

            mud = self.muhat[d]
            Sud = self.Suhat[d]
            moutd = T.sum(T.dot(psi1Kinv, mud))
            mout.append(moutd)

            Splusmm = Sud + T.outer(mud, mud)
            voutd = vconst + Talg.trace(T.dot(Splusmm, Kinvpsi2Kinv)) - moutd ** 2
            vout.append(T.sum(voutd))

        return mout, vout


    def compute_phi_prior(self):
        phi = 0
        for d in range(self.Dout):
            # s, a = npalg.slogdet(self.Kuu[d])
            a = np.log(npalg.det(self.Kuu[d]))
            phi += 0.5 * a
        return phi

    def compute_phi_posterior(self):
        phi = 0
        for d in range(self.Dout):
            mud_val = self.mu[d].get_value()
            Sud_val = self.Su[d].get_value()
            # s, a = npalg.slogdet(Sud_val)
            a = np.log(npalg.det(Sud_val))
            phi += 0.5 * np.dot(mud_val.T, np.dot(npalg.inv(Sud_val), mud_val))[0, 0]
            phi += 0.5 * a
        return phi

    def compute_phi_cavity(self):
        phi = 0
        for d in range(self.Dout):
            muhatd_val = self.muhat[d].get_value()
            Suhatd_val = self.Suhat[d].get_value()
            # s, a = npalg.slogdet(Sud_val)
            a = np.log(npalg.det(Suhatd_val))
            phi += 0.5 * np.dot(muhatd_val.T, np.dot(npalg.inv(Suhatd_val), muhatd_val))[0, 0]
            phi += 0.5 * a
        return phi
