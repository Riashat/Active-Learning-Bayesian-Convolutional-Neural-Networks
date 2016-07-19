import sys, os
sys.path.append(os.path.expanduser('~/synced/code/epsdgp/lib/tools/'))
from system_tools import *
import numpy as np
from EQ_kernel import *
import scipy.linalg as spalg
import scipy as scipy
from linalg_tools import *
import pprint as pp
import time
from tools import *

from scipy.cluster.vq import vq, kmeans2

from scipy.spatial.distance import cdist


class FITC_network:
    def __init__(self, Ntrain, layer_sizes, no_pseudos, lik, n_classes=None, zu_tied=False):
        self.lik = lik
        self.n_classes = n_classes
        self.no_layers = len(no_pseudos)
        self.layer_sizes = layer_sizes
        self.no_pseudos = no_pseudos
        self.Ntrain = Ntrain
        self.jitter = 1e-6
        self.zu_tied = zu_tied
        self.no_output_noise = self.lik != 'Gaussian'

        self.ones_M = [np.ones(Mi) for Mi in no_pseudos]
        self.ones_D = [np.ones(Di) for Di in layer_sizes[:-1]]
        self.ones_M_ls = [0 for Di in layer_sizes[:-1]]

        self.mu = []
        self.Su = []
        self.Splusmm = []
        self.muhat = []
        self.Suhat = []
        self.Suhatinv = []
        self.Splusmmhat = []
        self.Kuu = []
        self.Kuuinv = []
        self.dKuudls = []
        self.dKuudsf = []
        self.dKuudzu = []
        self.ls = []
        self.sf = []
        self.sn = []
        self.zu = []
        self.theta_1_R = []
        self.theta_2 = []
        self.theta_1 = []
        self.Ahat = []
        self.Bhat = []
        self.A = []
        self.B = []

        for l in range(self.no_layers):
            Din_l = self.layer_sizes[l]
            Dout_l = self.layer_sizes[l+1]
            M_l = self.no_pseudos[l]

            # variables for the mean and covariance of q(u)
            self.mu.append([ np.zeros([M_l,]) for _ in range(Dout_l) ])
            self.Su.append([ np.zeros([M_l, M_l]) for _ in range(Dout_l) ])
            self.Splusmm.append([ np.zeros([M_l, M_l]) for _ in range(Dout_l) ])

            # variables for the cavity distribution
            self.muhat.append([ np.zeros([M_l,]) for _ in range(Dout_l) ])
            self.Suhat.append([ np.zeros([M_l, M_l]) for _ in range(Dout_l) ])
            self.Suhatinv.append([ np.zeros([M_l, M_l]) for _ in range(Dout_l) ])
            self.Splusmmhat.append([ np.zeros([M_l, M_l]) for _ in range(Dout_l) ])

            # numpy variable for inducing points, Kuuinv, Kuu and its gradients
            if not self.zu_tied:
                self.zu.append([np.zeros([M_l, Din_l]) for _ in range(Dout_l)])
                self.Kuu.append([np.zeros([M_l, M_l]) for _ in range(Dout_l)])
                self.Kuuinv.append([np.zeros([M_l, M_l]) for _ in range(Dout_l)])
                self.dKuudls.append([np.zeros([Din_l, M_l, M_l]) for _ in range(Dout_l)])
                self.dKuudsf.append([np.zeros([M_l, M_l]) for _ in range(Dout_l)])
                self.dKuudzu.append([np.zeros([Din_l, M_l, M_l]) for _ in range(Dout_l)])
            else:
                self.zu.append(np.zeros([M_l, Din_l]))
                self.Kuu.append(np.zeros([M_l, M_l]))
                self.Kuuinv.append(np.zeros([M_l, M_l]))
                self.dKuudls.append(np.zeros([Din_l, M_l, M_l]))
                self.dKuudsf.append(np.zeros([M_l, M_l]))
                self.dKuudzu.append(np.zeros([Din_l, M_l, M_l]))


            # variables for the hyperparameters
            self.ls.append(np.zeros([Din_l, ]))
            self.sf.append(0)
            if not ( self.no_output_noise and (l == self.no_layers - 1) ):
                self.sn.append(0)
            
            # and natural parameters
            self.theta_1_R.append([np.zeros([M_l, M_l]) for _ in range(Dout_l)])
            self.theta_2.append([np.zeros([M_l,]) for _ in range(Dout_l)])
            self.theta_1.append([np.zeros([M_l, M_l]) for _ in range(Dout_l)])

            # terms that are common to all datapoints in each minibatch
            self.Ahat.append([np.zeros([M_l,]) for _ in range(Dout_l)])
            self.Bhat.append([np.zeros([M_l, M_l]) for _ in range(Dout_l)])
            self.A.append([np.zeros([M_l,]) for _ in range(Dout_l)])
            self.B.append([np.zeros([M_l, M_l]) for _ in range(Dout_l)])

    def compute_phi_prior(self):
        logZ_prior = 0
        for i in range(self.no_layers):
            Dout_i = self.layer_sizes[i+1]
            if not self.zu_tied:
                for d in range(Dout_i):
                    (sign, logdet) = np.linalg.slogdet(self.Kuu[i][d])
                    logZ_prior += 0.5 * logdet
            else:
                (sign, logdet) = np.linalg.slogdet(self.Kuu[i])
                logZ_prior += Dout_i * 0.5 * logdet

        return logZ_prior

    def compute_phi_posterior(self):
        logZ_posterior = 0
        for i in range(self.no_layers):

            Dout_i = self.layer_sizes[i+1]
            for d in range(Dout_i):
                mud_val = self.mu[i][d]
                Sud_val = self.Su[i][d]
                (sign, logdet) = np.linalg.slogdet(Sud_val)
                # print 'phi_poste: ', 0.5 * logdet, 0.5 * np.sum(mud_val * spalg.solve(Sud_val, mud_val.T))
                logZ_posterior += 0.5 * logdet
                logZ_posterior += 0.5 * np.sum(mud_val * spalg.solve(Sud_val, mud_val.T))
        return logZ_posterior

    def compute_phi_cavity(self):
        phi_cavity = 0
        for i in range(self.no_layers):
            Dout_i = self.layer_sizes[i+1]
            for d in range(Dout_i):
                muhatd_val = self.muhat[i][d]
                Suhatd_val = self.Suhat[i][d]
                (sign, logdet) = np.linalg.slogdet(Suhatd_val)
                phi_cavity += 0.5 * logdet
                phi_cavity += 0.5 * np.sum(muhatd_val * spalg.solve(Suhatd_val, muhatd_val.T))
        return phi_cavity

    def output_probabilistic(self, x, inter_layer=None):

        if inter_layer is None:
            inter_layer = self.no_layers

        Dini = self.layer_sizes[0]
        Douti = self.layer_sizes[1]
        Mi = self.no_pseudos[0]
        mx = x.reshape((Dini,))
        # Recursively compute output
        # deal with input layer
        ls_i = self.ls[0]
        sf_i = self.sf[0]
        Dout_i = self.layer_sizes[1]
        # create place holders
        mout_i = np.zeros((Douti,))
        vout_i = np.zeros((Douti,))
        # compute the psi terms
        psi0 = np.exp(2.0*sf_i)
        if not (self.no_output_noise and (self.no_layers == 1)):
            sn2 = np.exp(2.0*self.sn[0])
        else:
            sn2 = 0.0
        if self.zu_tied:
            psi1 = compute_kernel(2*ls_i, 2*sf_i, mx, self.zu[0])

        for d in range(Dout_i):
            if not self.zu_tied:
                psi1 = compute_kernel(2*ls_i, 2*sf_i, mx, self.zu[0][d])

            psi1 = psi1.reshape((Mi, ))
            Aid = self.A[0][d]
            Bid = self.B[0][d]
            moutid = np.sum(np.dot(psi1, Aid))
            Bid_psi1 = np.dot(Bid, psi1)
            sumJ = np.sum(np.dot(psi1.T, Bid_psi1))
            voutid = sn2 + psi0 + sumJ
            mout_i[d] = moutid
            vout_i[d] = voutid

        mx_previous = mout_i
        vx_previous = vout_i

        # and other layers
        for i in range(1, self.no_layers, 1):
            if i == inter_layer:
                break
            
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            Mi = self.no_pseudos[i]
            ls_i = self.ls[i]
            sf_i = self.sf[i]
            Dout_i = self.layer_sizes[i+1]
            # create place holders
            mout_i = np.zeros((Douti,))
            vout_i = np.zeros((Douti,))
            # compute the psi terms
            psi0 = np.exp(2.0*sf_i)
            if not (self.lik != 'Gaussian' and (i == self.no_layers-1)):
                sn2 = np.exp(2.0 * self.sn[i])
            else:
                sn2 = 0.0

            if self.zu_tied:
                psi1 = compute_psi1(2*ls_i, 2*sf_i, mx_previous, vx_previous, self.zu[i])
                psi2 = compute_psi2(2*ls_i, 2*sf_i, mx_previous, vx_previous, self.zu[i])

            for d in range(Dout_i):
                if not self.zu_tied:
                    psi1 = compute_psi1(2*ls_i, 2*sf_i, mx_previous, vx_previous, self.zu[i][d])
                    psi2 = compute_psi2(2*ls_i, 2*sf_i, mx_previous, vx_previous, self.zu[i][d])

                Aid = self.A[i][d]
                Bid = self.B[i][d]
                moutid = np.sum(np.dot(psi1, Aid))
                J = Bid * psi2
                sumJ = np.sum(J)
                voutid = sn2 + psi0 + sumJ - moutid**2
                mout_i[d] = moutid
                vout_i[d] = voutid

            mx_previous = mout_i
            vx_previous = vout_i

        mout_i = mout_i.reshape((1, mout_i.shape[0]))
        vout_i = vout_i.reshape((1, vout_i.shape[0]))
        return mout_i, vout_i

    def output_of_a_layer(self, x, layer_no):
        mx_previous = x

        i = layer_no - 1
        ls_i = self.ls[i]
        sf_i = self.sf[i]
        Mi = self.no_pseudos[i]
        Dout_i = self.layer_sizes[i+1]
        # create place holders
        mout_i = np.zeros((1, Dout_i))
        vout_i = np.zeros((1, Dout_i))
        # compute the psi terms
        psi0 = np.exp(2.0*sf_i)
        if not (self.no_output_noise and i == self.no_layers-1):
            sn2 = np.exp(2.0*self.sn[i])
        else:
            sn2 = 0
        if self.zu_tied:
            psi1 = compute_kernel(2*ls_i, 2*sf_i, mx_previous, self.zu[i])

        for d in range(Dout_i):
            if not self.zu_tied:
                psi1 = compute_kernel(2*ls_i, 2*sf_i, mx_previous, self.zu[i][d])

            psi1 = psi1.reshape((Mi, ))
            Aid = self.A[i][d]
            Bid = self.B[i][d]
            moutid = np.sum(np.dot(psi1, Aid))
            Bid_psi1 = np.dot(Bid, psi1)
            sumJ = np.sum(np.dot(psi1.T, Bid_psi1))
            voutid = sn2 + psi0 + sumJ
            mout_i[0, d] = moutid
            vout_i[0, d] = voutid

        return mout_i, vout_i

    def compute_logZ_and_gradients(self, x, y, epsilon=None):
        zu_tied = self.zu_tied
        no_layers = self.no_layers
        # variables to hold gradients of logZ
        grad_names = ['ls', 'sf', 'sn', 'zu', 'Ahat', 'Bhat']
        grads = {}
        for name in grad_names:
            grads[name] = [[] for _ in range(no_layers)]
            grads[name] = [[] for _ in range(no_layers)]

        # FORWARD PROPAGATION
        # variables to hold gradients of output means and variances
        mout = [[] for _ in range(no_layers)]
        vout = [[] for _ in range(no_layers)]
        dmout = {}
        dvout = {}
        output_grad_names = ['ls', 'sf', 'sn', 'zu', 'mx', 'vx', 'Ahat', 'Bhat']
        for name in output_grad_names:
            dmout[name] = [[] for _ in range(no_layers)]
            dvout[name] = [[] for _ in range(no_layers)]
        
        # ************************** FIRST LAYER **************************
        # deal with the first layer separately since there is no need for psi2
        Dini = self.layer_sizes[0]
        Douti = self.layer_sizes[1]
        Mi = self.no_pseudos[0]
        mx = x.reshape((Dini,))
        lsi = self.ls[0]
        sfi = self.sf[0]
        ls2 = np.exp(2*lsi)
        psi0 = np.exp(2.0*sfi)
        mouti = np.zeros((Douti,))
        vouti = np.zeros((Douti,))
        ones_Di = self.ones_D[0]
        ones_Mi = self.ones_M[0]
        # todo: deal with no noise at the last layer for certain liks
        if self.lik != 'Gaussian' and no_layers == 1:
            sn2 = 0
        else:
            sn2 = np.exp(2.0*self.sn[0])

        if zu_tied:
            zui = self.zu[0]
            psi1 = compute_kernel(2*lsi, 2*sfi, mx, zui)

        for d in range(Douti):
            if not zu_tied:
                zuid = self.zu[0][d]
                psi1 = compute_kernel(2*lsi, 2*sfi, mx, zuid)
            else:
                zuid = zui

            psi1 = psi1.reshape((Mi, ))
            Ahatid = self.Ahat[0][d]
            Bhatid = self.Bhat[0][d]

            moutid = np.sum(np.dot(psi1, Ahatid))
            Bhatid_psi1 = np.dot(Bhatid, psi1)
            sumJ = np.sum(np.dot(psi1.T, Bhatid_psi1))
            voutid = sn2 + psi0 + sumJ
            mouti[d] = moutid
            vouti[d] = voutid
            
            # now compute gradients of output mean
            # wrt Ahat and Bhat
            dmout['Ahat'][0].append(psi1.T)
            dmout['Bhat'][0].append(0)
            
            # wrt hypers
            dmout['sf'][0].append(2*moutid)
            mx_minus_zuid = np.outer(ones_Mi, mx) - zuid
            temp1 = np.outer(psi1, ones_Di) * 0.5 * mx_minus_zuid**2
            dmoutid_dls = np.dot(Ahatid, temp1) * 1.0 / ls2
            dmout['ls'][0].append(2*dmoutid_dls)
            # temp2 = mx_minus_zuid * np.outer(ones_Mi, 1.0 / ls2 )
            temp2 = mx_minus_zuid * self.ones_M_ls[0]
            dmoutid_dzu = np.outer(psi1 * Ahatid, ones_Di) * temp2
            dmout['zu'][0].append(dmoutid_dzu)
            dmout['sn'][0].append(0)

            # now compute gradients of the output variance
            # wrt Ahat and Bhat
            dvout['Ahat'][0].append(0)
            dvoutid_dBhat = np.outer(psi1, psi1)
            dvout['Bhat'][0].append(dvoutid_dBhat)

            # wrt sf
            dvoutid_dsf = psi0 + 2*sumJ
            dvout['sf'][0].append(2*dvoutid_dsf)
            # wrt ls
            dvoutid_dls = 2*np.dot(Bhatid_psi1, temp1) * 1.0 / ls2
            dvout['ls'][0].append(2*dvoutid_dls)
            dvoutid_dzu = 2*np.outer(psi1 * Bhatid_psi1, ones_Di) * temp2
            dvout['zu'][0].append(dvoutid_dzu)
            
            # wrt noise
            if self.lik != 'Gaussian' and no_layers == 1:
                dvout['sn'][0].append(0)
            else:
                dvout['sn'][0].append(2*sn2)

        mout[0] = mouti
        vout[0] = vouti

        # ************************** END OF FIRST LAYER **********************


        # ************************** OTHERS LAYER **************************
        for i in range(1, no_layers):
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            Mi = self.no_pseudos[i]
            mx = mout[i-1]
            vx = vout[i-1]
            
            lsi = self.ls[i]
            sfi = self.sf[i]
            
            psi0 = np.exp(2.0*sfi)
            mouti = np.zeros((Douti,))
            vouti = np.zeros((Douti,))
            ones_Di = self.ones_D[i]
            ones_Mi = self.ones_M[i]
            # todo: deal with no noise at the last layer for certain liks
            if self.lik != 'Gaussian' and i == no_layers-1:
                sn2 = 0
            else:
                sn2 = np.exp(2.0*self.sn[i])

            if zu_tied:
                zui = self.zu[i]
                psi1 = compute_psi1(2*lsi, 2*sfi, mx, vx, zui)
                psi2 = compute_psi2(2*lsi, 2*sfi, mx, vx, zui)

            for d in range(Douti):
                if not zu_tied:
                    zuid = self.zu[i][d]
                    psi1 = compute_psi1(2*lsi, 2*sfi, mx, vx, zuid)
                    psi2 = compute_psi2(2*lsi, 2*sfi, mx, vx, zuid)
                    Kuuinvid = self.Kuuinv[i][d]
                    Kuuid = self.Kuu[i][d]
                else:
                    zuid = zui
                    Kuuinvid = self.Kuuinv[i]
                    Kuuid = self.Kuu[i]
                
                Ahatid = self.Ahat[i][d]
                Bhatid = self.Bhat[i][d]
                muhatid = self.muhat[i][d]
                Suhatid = self.Suhat[i][d]

                moutid = np.sum(np.dot(psi1, Ahatid))
                J = Bhatid * psi2
                sumJ = np.sum(J)
                voutid = sn2 + psi0 + sumJ - moutid**2
                mouti[d] = moutid
                vouti[d] = voutid

                # now compute gradients of output mean
                # wrt Ahat and Bhat
                dmoutid_dAhat = psi1.T
                dmout['Ahat'][i].append(dmoutid_dAhat.reshape((Mi, )))
                dmout['Bhat'][i].append(0)
                
                # wrt xmean
                ls2 = np.exp(2*lsi)
                lspxvar = ls2 + vx
                Ahatid_psi1 = Ahatid * psi1
                dmoutid_dmx = (- np.sum(Ahatid_psi1) * mx + np.dot(Ahatid_psi1, zuid)) * 1.0 / lspxvar
                dmout['mx'][i].append(dmoutid_dmx)
                
                # wrt xvar
                psi1_Kuuinv = np.dot(psi1, Kuuinvid)
                term1 = np.sum(psi1_Kuuinv * muhatid) * 1.0 / lspxvar * -0.5
                term2 = np.dot(Ahatid_psi1, 0.5 * (np.outer(ones_Mi, mx) - zuid)**2) * 1.0 / (lspxvar**2)
                dmoutid_dvx = term1 + term2
                dmout['vx'][i].append(dmoutid_dvx)
                
                # wrt hypers
                dmoutid_dsf = moutid
                dmout['sf'][i].append(2*moutid)
                mx_minus_zuid = np.outer(ones_Mi, mx) - zuid
                temp1 = np.outer(psi1, ones_Di) * 0.5 * mx_minus_zuid**2
                dmoutid_dls = moutid * 0.5 * (1 - ls2 / (ls2 + vx)) + \
                    np.dot(Ahatid, temp1) * 1.0 / ((ls2 + vx)**2) * ls2
                dmout['ls'][i].append(2*dmoutid_dls)
                temp2 = mx_minus_zuid * np.outer(ones_Mi, 1.0 / (ls2 + vx))
                dmoutid_dzu = np.outer(psi1 * Ahatid, ones_Di) * temp2
                dmout['zu'][i].append(dmoutid_dzu)
                dmout['sn'][i].append(0)

                # now compute gradients of the output variance
                # wrt Ahat and Bhat
                dvoutid_dAhat = - 2.0 * moutid * dmoutid_dAhat
                dvout['Ahat'][i].append(dvoutid_dAhat.reshape((Mi, )))
                dvout['Bhat'][i].append(psi2)

                # wrt xmean
                D = ls2
                Dhalf = ls2 / 2.0
                Btilde = 1.0 / (Dhalf + vx)
                H = 0.5 * zuid * np.outer(ones_Mi, Btilde)
                dvoutid_dmx = 2.0 * np.dot(np.dot(ones_Mi, J), H) - sumJ * Btilde * mx \
                    - 2.0 * moutid * dmoutid_dmx
                dvout['mx'][i].append(dvoutid_dmx)

                # wrt xvar
                term1 = - sumJ * 1.0 / (Dhalf + vx) * 0.5
                dBtilde = -(Btilde**2)
                dVtilde = dBtilde
                dQtilde = 0.25 * dVtilde
                M1 = - 0.5 * np.outer(ones_Mi, dQtilde) * (zuid**2)
                M2 = + 0.5 * np.outer(ones_Mi, dBtilde * mx) * zuid
                M3 = - 0.25 * np.outer(ones_Mi, dVtilde) * zuid
                term2 = 2.0 * np.dot(np.dot(ones_Mi, J), M1) + \
                    2.0 * np.dot(np.dot(ones_Mi, J), M2) \
                    - 0.5 * sumJ * (mx**2) * dBtilde \
                    + np.dot(ones_Mi, zuid * np.dot(J, M3))
                dvoutid_dvx = term1 + term2 - 2.0 * moutid * dmoutid_dvx
                dvout['vx'][i].append(dvoutid_dvx)

                # wrt sf
                dvoutid_dsf = psi0 + 2*np.sum(Bhatid * psi2) - 2.0 * moutid * dmoutid_dsf
                dvout['sf'][i].append(2*np.sum(dvoutid_dsf))

                # wrt to ls
                Vtilde = Btilde - 1.0 / Dhalf
                Qtilde = 1.0 / D + 0.25 * Vtilde
                dBtilde = -(Btilde**2) * Dhalf
                dVtilde = dBtilde + (Dhalf**(-2)) * Dhalf
                dQtilde = -(D**(-2)) * D + 0.25 * dVtilde
                term1 = sumJ * Dhalf * 0.5 * (1.0 / Dhalf - 1.0 / (Dhalf + vx))
                M1 = - 0.5 * np.outer(ones_Mi, dQtilde) * (zuid**2)
                M2 = + 0.5 * np.outer(ones_Mi, dBtilde * mx) * zuid
                M3 = - 0.25 * np.outer(ones_Mi, dVtilde) * zuid
                term2 = 2.0 * np.dot(np.dot(ones_Mi, J), M1) + \
                    2.0 * np.dot(np.dot(ones_Mi, J), M2) \
                    - 0.5 * sumJ * (mx**2) * dBtilde \
                    + np.dot(ones_Mi, zuid * np.dot(J, M3))
                dvoutid_dls = term1 + term2 - 2*moutid*dmoutid_dls
                dvout['ls'][i].append(2*dvoutid_dls)

                # wrt zu
                upsilon = np.dot(ones_Mi, J)

                term1 = 2 * np.outer(upsilon, ones_Di) * -0.5 * zuid * np.outer(ones_Mi, Qtilde) * 2
                term2 = 2 * np.outer(upsilon, ones_Di) * +0.5 * np.outer(ones_Mi, mx * Btilde) 
                term3 = 2 * np.dot(J, (zuid * np.outer(np.ones(Mi), Vtilde))) * -0.25
                dvoutid_dzu = term1 + term2 + term3 - 2*moutid*dmoutid_dzu
                dvout['zu'][i].append(dvoutid_dzu)

                # wrt noise
                if self.lik != 'Gaussian' and i == no_layers-1:
                    dvout['sn'][i].append(0)
                else:
                    dvout['sn'][i].append(2*sn2)

            mout[i] = mouti
            vout[i] = vouti

        # ************************** END OF FORWARD PASS **************************

                
        
        # ************************** COMPUTE LOG Z and DO BACKWARD STEP ***********

        if self.lik == 'Gaussian':
            m = mout[-1]
            v = vout[-1]
            # print m, v
            logZ = np.sum( -0.5 * (np.log(v) + (y - m)**2 / v) )

            dlogZ_dmlast = (y - m) / v
            dlogZ_dvlast = -0.5 / v + 0.5 * (y - m)**2 / v**2
            dlogZ_dmlast = np.reshape(dlogZ_dmlast, [self.layer_sizes[-1],])
            dlogZ_dvlast = np.reshape(dlogZ_dvlast, [self.layer_sizes[-1],])
        elif self.lik == 'Probit':
            # binary classification using probit likelihood
            m = mout[-1]
            v = vout[-1]
            t = y * m / np.sqrt(1 + v)
            Z = 0.5 * (1 + math.erf(t / np.sqrt(2)))
            eps = 1e-16
            logZ = np.log(Z + eps)

            dlogZ_dt = 1/(Z + eps) * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
            dt_dm = y / np.sqrt(1 + v)
            dt_dv = -0.5 * y * m / (1 + v)**1.5
            dlogZ_dmlast = dlogZ_dt * dt_dm
            dlogZ_dvlast = dlogZ_dt * dt_dv
            dlogZ_dmlast = np.reshape(dlogZ_dmlast, [1,])
            dlogZ_dvlast = np.reshape(dlogZ_dvlast, [1,])
        elif self.lik == 'Softmax':
            # multiclass classification using softmax likelihood
            m = mout[-1]
            v = vout[-1]
            veps = np.sqrt(v) * epsilon
            samples = m + veps
            y = np.reshape(y, [1, self.n_classes])
            ylabel = np.argmax(y, axis=1)
            # p_y_given_samples = softmax(samples)
            # p_y_sum = np.sum(p_y_given_samples[:, ylabel]) / epsilon.shape[0]

            # p_y_given_samples, dsamples = softmax_onecol(samples, ylabel)
            # p_y_sum = np.sum(p_y_given_samples) / epsilon.shape[0]
            # dviam = np.sum(dsamples, axis=0) / epsilon.shape[0]
            # dsqrtv = np.sum(dsamples * epsilon, axis=0) / epsilon.shape[0]
            # dviav = 0.5 * dsqrtv / np.sqrt(v)
            # logZ = np.log(p_y_sum)
            # dpysum = 1/p_y_sum
            # dm = dpysum * dviam
            # dv = dpysum * dviav
            
            py, dpy = softmax_given_y(samples, ylabel)
            logZ = np.log(py)
            dsamples = 1/py * dpy
            dm = np.sum(dsamples, axis=0)
            dv = np.sum(dsamples * epsilon, axis=0) / 2 / np.sqrt(v)
            
            dlogZ_dmlast = np.reshape(dm, [self.n_classes,])
            dlogZ_dvlast = np.reshape(dv, [self.n_classes,])

        dlogZ_dlast = {}
        dlogZ_dlast['mx'] = dlogZ_dmlast
        dlogZ_dlast['vx'] = dlogZ_dvlast

        # deal with layers in reverse order
        for i in range(self.no_layers-1, -1, -1):
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            dlogZ_din = {}
            dlogZ_din['mx'] = np.zeros((1, Dini))
            dlogZ_din['vx'] = np.zeros((1, Dini))
            for d in range(Douti):
                for name in output_grad_names:
                    if (i > 0) and (name in ['mx', 'vx']):
                        dlogZ_din[name] += dlogZ_dlast['mx'][d] * dmout[name][i][d] + dlogZ_dlast['vx'][d] * dvout[name][i][d]
                    elif name not in ['mx', 'vx']:
                        # print i, d, name                      
                        grad_name_id = dlogZ_dlast['mx'][d] * dmout[name][i][d] + dlogZ_dlast['vx'][d] * dvout[name][i][d]
                        grads[name][i].append(grad_name_id)

            if i > 0:
                dlogZ_din['mx'] = np.reshape(dlogZ_din['mx'], (Dini, ))
                dlogZ_din['vx'] = np.reshape(dlogZ_din['vx'], (Dini, ))
                dlogZ_dlast = dlogZ_din

        return logZ, grads

    def compute_kuu(self):
        # compute Kuu for each layer
        for i in range(self.no_layers):
            ls_i = self.ls[i]
            sf_i = self.sf[i]
            Dout_i = self.layer_sizes[i+1]
            M_i = self.no_pseudos[i]
            if not self.zu_tied:
                for d in range(Dout_i):
                    zu_id = self.zu[i][d]
                    self.Kuu[i][d] = compute_kernel(2*ls_i, 2*sf_i, zu_id, zu_id)  
                    self.Kuu[i][d] += np.diag(self.jitter * np.ones((M_i, )))
                    self.Kuuinv[i][d] = matrixInverse(self.Kuu[i][d])
            else:
                zu_i = self.zu[i]
                self.Kuu[i] = compute_kernel(2*ls_i, 2*sf_i, zu_i, zu_i)
                self.Kuu[i] += np.diag(self.jitter * np.ones((M_i, )))
                self.Kuuinv[i] = matrixInverse(self.Kuu[i])

    def compute_cavity(self):
        # compute the leave one out moments for each layer
        beta = (self.Ntrain - 1.0) * 1.0 / self.Ntrain
        for i in range(self.no_layers):
            Dout_i = self.layer_sizes[i+1]
            for d in range(Dout_i):
                if self.zu_tied:
                    Kuuinvid = self.Kuuinv[i]
                else:
                    Kuuinvid = self.Kuuinv[i][d]

                self.Suhatinv[i][d] = Kuuinvid + beta * self.theta_1[i][d]
                ShatinvMhat = beta * self.theta_2[i][d]
                # Shat = npalg.inv(self.Suhatinv[i][d])
                Shat = matrixInverse(self.Suhatinv[i][d])
                self.Suhat[i][d] = Shat
                mhat = np.dot(Shat, ShatinvMhat)
                self.muhat[i][d] = mhat
                self.Ahat[i][d] = np.dot(Kuuinvid, mhat)
                Smm = Shat + np.outer(mhat, mhat)
                self.Splusmmhat[i][d] = Smm
                if i > 0:
                    self.Bhat[i][d] = np.dot(Kuuinvid, np.dot(Smm, Kuuinvid)) - Kuuinvid
                else:
                    self.Bhat[i][d] = np.dot(Kuuinvid, np.dot(Shat, Kuuinvid)) - Kuuinvid


    def update_posterior(self):
        # compute the posterior approximation
        for i in range(self.no_layers):
            Dout_i = self.layer_sizes[i+1]
            for d in range(Dout_i):
                if self.zu_tied:
                    Kuuinvid = self.Kuuinv[i]
                else:
                    Kuuinvid = self.Kuuinv[i][d]

                Sinv = Kuuinvid + self.theta_1[i][d]
                SinvM = self.theta_2[i][d]
                # S = npalg.inv(Sinv)
                S = matrixInverse(Sinv)
                self.Su[i][d] = S
                m = np.dot(S, SinvM)
                self.mu[i][d] = m

                Smm = S + np.outer(m, m)
                self.Splusmm[i][d] = Smm
                self.B[i][d] = np.dot(Kuuinvid, np.dot(Smm, Kuuinvid)) - Kuuinvid

    def update_posterior_for_prediction(self):
        # compute the posterior approximation
        for i in range(self.no_layers):
            Dout_i = self.layer_sizes[i+1]
            for d in range(Dout_i):
                if self.zu_tied:
                    Kuuinvid = self.Kuuinv[i]
                else:
                    Kuuinvid = self.Kuuinv[i][d]

                Sinv = Kuuinvid + self.theta_1[i][d]
                SinvM = self.theta_2[i][d]
                # S = npalg.inv(Sinv)
                S = matrixInverse(Sinv)
                self.Su[i][d] = S
                m = np.dot(S, SinvM)
                self.mu[i][d] = m

                self.A[i][d] = np.dot(Kuuinvid, m)
                Smm = S + np.outer(m, m)
                self.Splusmm[i][d] = Smm
                if i > 0:
                    self.B[i][d] = np.dot(Kuuinvid, np.dot(Smm, Kuuinvid)) - Kuuinvid
                else:
                    self.B[i][d] = np.dot(Kuuinvid, np.dot(S, Kuuinvid)) - Kuuinvid
                # Smm = S + np.outer(m, m)
                # self.Splusmm[i][d] = Smm
                # self.B[i][d] = np.dot(Kuuinvid, np.dot(Smm, Kuuinvid)) - Kuuinvid

    def init_hypers_Gaussian(self, x_train, y_train):
        # dict to hold hypers, inducing points and parameters of q(U)
        params = {'ls': [],
                  'sf': [],
                  'zu': [],
                  'sn': [],
                  'eta1_R': [],
                  'eta2': []}

        # first layer
        M1 = self.no_pseudos[0]
        Dout1 = self.layer_sizes[1]
        Ntrain = x_train.shape[0]
        if Ntrain < 20000:
            centroids, label = kmeans2(x_train, M1, minit='points')
        else:
            randind = np.random.permutation(Ntrain)
            centroids = x_train[randind[0:M1], :]
        # ls1 = self.estimate_ls(x_train)
        # ls1[ls1 < -5] = -5
        ls1 = self.estimate_ls_temp(x_train)
        sn1 = np.array([np.log(0.1*np.std(y_train))])
        sf1 = np.array([np.log(1*np.std(y_train))])
        sf1[sf1 < -5] = 0
        sn1[sn1 < -5] = np.log(0.1)

        # # for maunaloa only
        # ls1 = ls1 / 20
        # sn1 = np.array([np.log(0.001)])
        # sf1 = np.array([np.log(0.5)])

        params['sf'].append(sf1)
        params['ls'].append(ls1)
        params['sn'].append(sn1)
        eta1_R0 = []
        eta20 = []
        for d in range(Dout1):
            eta1_R0.append(np.random.randn(M1*(M1+1)/2, ))
            eta20.append(np.random.randn(M1, ))
        params['eta1_R'].append(eta1_R0)
        params['eta2'].append(eta20)

        if self.zu_tied:
            zu0 = centroids
        else:
            zu0 = []
            for d in range(Dout1):
                zu0.append(centroids)
        params['zu'].append(zu0)

        # other layers
        for i in range(1, self.no_layers):
            Mi = self.no_pseudos[i]
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            sfi = np.log(np.array([1]))
            sni = np.log(np.array([0.1]))
            lsi = np.ones((Dini, ))
            params['sf'].append(sfi)
            params['ls'].append(lsi)
            params['sn'].append(sni)
            if self.zu_tied:
                zui = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Dini))
            else:
                zui = []
                for d in range(Douti):
                    zuii = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Dini))
                    zui.append(zuii)
            params['zu'].append(zui)

            eta1_Ri = []
            eta2i = []
            for d in range(Douti):
                eta1_Ri.append(np.random.randn(Mi*(Mi+1)/2, ) / 10)
                eta2i.append(np.random.randn(Mi, ) / 10)
            params['eta1_R'].append(eta1_Ri)
            params['eta2'].append(eta2i)
        
        return params

    def init_hypers_Probit(self, x_train):
        # dict to hold hypers, inducing points and parameters of q(U)
        params = {'ls': [],
                  'sf': [],
                  'zu': [],
                  'sn': [],
                  'eta1_R': [],
                  'eta2': []}

        # first layer
        M1 = self.no_pseudos[0]
        Dout1 = self.layer_sizes[1]
        Ntrain = x_train.shape[0]
        if Ntrain < 20000:
            centroids, label = kmeans2(x_train, M1, minit='points')
        else:
            randind = np.random.permutation(Ntrain)
            centroids = x_train[randind[0:M1], :]
        # ls1 = self.estimate_ls(x_train)
        # ls1[ls1 < -5] = -5
        ls1 = self.estimate_ls_temp(x_train)
        sn1 = np.array([np.log(0.01)])
        sf1 = np.array([np.log(1)])
        sf1[sf1 < -5] = 0
        sn1[sn1 < -5] = np.log(0.1)
        params['sf'].append(sf1)
        params['ls'].append(ls1)
        if self.no_layers > 1:
            params['sn'].append(sn1)
        else:
            # TODO
            params['sn'].append(0)
        eta1_R0 = []
        eta20 = []
        for d in range(Dout1):
            eta1_R0.append(np.random.randn(M1*(M1+1)/2, ))
            eta20.append(np.random.randn(M1, ))
        params['eta1_R'].append(eta1_R0)
        params['eta2'].append(eta20)

        if self.zu_tied:
            zu0 = centroids
        else:
            zu0 = []
            for d in range(Dout1):
                zu0.append(centroids)
        params['zu'].append(zu0)

        # other layers
        for i in range(1, self.no_layers):
            Mi = self.no_pseudos[i]
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            sfi = np.log(np.array([1]))
            sni = np.log(np.array([0.1]))
            lsi = np.ones((Dini, ))
            params['sf'].append(sfi)
            params['ls'].append(lsi)
            if i != self.no_layers-1:
                params['sn'].append(sni)
            else:
                # TODO
                params['sn'].append(0)
            if self.zu_tied:
                zui = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Dini))
            else:
                zui = []
                for d in range(Douti):
                    zuii = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Dini))
                    zui.append(zuii)
            params['zu'].append(zui)

            eta1_Ri = []
            eta2i = []
            for d in range(Douti):
                eta1_Ri.append(np.random.randn(Mi*(Mi+1)/2, ) / 10)
                eta2i.append(np.random.randn(Mi, ) / 10)
            params['eta1_R'].append(eta1_Ri)
            params['eta2'].append(eta2i)

        return params

    def init_hypers_Softmax(self, x_train, y_train):
        # dict to hold hypers, inducing points and parameters of q(U)
        params = {'ls': [],
                  'sf': [],
                  'zu': [],
                  'sn': [],
                  'eta1_R': [],
                  'eta2': []}

        # first layer
        # ls1 = self.estimate_ls(x_train)
        # ls1[ls1 < -5] = np.log(0.01)
        ls1 = self.estimate_ls_temp(x_train)
        sn1 = np.array([np.log(0.01)])
        sf1 = np.array([np.log(1)])
        sf1[sf1 < -5] = 0
        sn1[sn1 < -5] = np.log(0.1)
        params['sf'].append(sf1)
        params['ls'].append(ls1)
        if self.no_layers > 1:
            params['sn'].append(sn1)
        else:
            # TODO
            params['sn'].append(0)
        zu0 = []
        eta1_R0 = []
        eta20 = []

        M1 = self.no_pseudos[0]
        Dout1 = self.layer_sizes[1]
        Ntrain = x_train.shape[0]
        if Ntrain < 20000:
            centroids, label = kmeans2(x_train, M1, minit='points')
        else:
            randind = np.random.permutation(Ntrain)
            centroids = x_train[randind[0:M1], :]

        eta1_R0 = []
        eta20 = []
        for d in range(Dout1):
            eta1_R0.append(np.random.randn(M1*(M1+1)/2, ))
            eta20.append(np.random.randn(M1, ))
        params['eta1_R'].append(eta1_R0)
        params['eta2'].append(eta20)

        if self.zu_tied:
            zu0 = centroids
        else:
            zu0 = []
            for d in range(Dout1):
                zu0.append(centroids)
        params['zu'].append(zu0)

        # other layers
        for i in range(1, self.no_layers):
            Mi = self.no_pseudos[i]
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            sfi = np.log(np.array([1]))
            sni = np.log(np.array([0.1]))
            lsi = np.ones((Dini, ))
            params['sf'].append(sfi)
            params['ls'].append(lsi)
            if i != self.no_layers-1:
                params['sn'].append(sni)
            else:
                # TODO
                params['sn'].append(0)
            if self.zu_tied:
                zui = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Dini))
            else:
                zui = []
                for d in range(Douti):
                    zuii = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Dini))
                    zui.append(zuii)
            params['zu'].append(zui)

            eta1_Ri = []
            eta2i = []
            for d in range(Douti):
                eta1_Ri.append(np.random.randn(Mi*(Mi+1)/2, ) / 10)
                eta2i.append(np.random.randn(Mi, ) / 10)
            params['eta1_R'].append(eta1_Ri)
            params['eta2'].append(eta2i)

        return params

    def get_hypers(self):
        params = {'ls': [],
                  'sf': [],
                  'zu': [],
                  'sn': [],
                  'eta1_R': [],
                  'eta2': []}
        for i in range(self.no_layers):
            Mi = self.no_pseudos[i]
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            params['ls'].append(self.ls[i])
            params['sf'].append(self.sf[i])
            if not (self.no_output_noise and (i == self.no_layers-1)):
                params['sn'].append(self.sn[i])
            triu_ind = np.triu_indices(Mi)
            diag_ind = np.diag_indices(Mi)
            params_zu_i = []
            params_eta2_i = []
            params_eta1_Ri = []
            if self.zu_tied:
                params_zu_i = self.zu[i]
            else:
                for d in range(Douti):
                    params_zu_i.append(self.zu[i][d])

            for d in range(Douti):
                params_eta2_i.append(self.theta_2[i][d])
                Rd = self.theta_1_R[i][d]
                Rd[diag_ind] = np.log(Rd[diag_ind])
                params_eta1_Ri.append(Rd[triu_ind].reshape((Mi*(Mi+1)/2,)))

            params['zu'].append(params_zu_i)
            params['eta1_R'].append(params_eta1_Ri)
            params['eta2'].append(params_eta2_i)
        return params

    def estimate_ls(self, X):
        Ntrain = X.shape[0]
        if Ntrain < 10000:
            X1 = np.copy(X)
        else:
            randind = np.random.permutation(Ntrain)
            X1 = X[randind[0:(5*self.no_pseudos[0])], :]
    
        # d2 = compute_distance_matrix(X1)
        D = X1.shape[1]
        N = X1.shape[0]
        triu_ind = np.triu_indices(N)
        ls = np.zeros((D, ))
        for i in range(D):
            X1i = np.reshape(X1[:, i], (N, 1))
            d2i = cdist(X1i, X1i, 'euclidean')
            # d2i = d2[:, :, i]
            d2imed = np.median(d2i[triu_ind])
            # d2imed = 0.01
            # print d2imed, 
            ls[i] = np.log(d2imed + 1e-16)
        return ls

    def estimate_ls_temp(self, X):
        Ntrain = X.shape[0]
        if Ntrain < 10000:
            X1 = np.copy(X)
        else:
            randind = np.random.permutation(Ntrain)
            X1 = X[randind[0:(5*self.no_pseudos[0])], :]

        dist = cdist(X1, X1, 'euclidean')

        # diff = X1[:, None, :] - X1[None, :, :]
        # dist = np.sum(abs(diff), axis=2)
        D = X1.shape[1]
        N = X1.shape[0]
        triu_ind = np.triu_indices(N)
        ls = np.zeros((D, ))
        d2imed = np.median(dist[triu_ind])
        for i in range(D):
            ls[i] = np.log(d2imed + 1e-16)
        return ls

    def update_hypers(self, params):
        for i in range(self.no_layers):
            Mi = self.no_pseudos[i]
            Dini = self.layer_sizes[i]
            Douti = self.layer_sizes[i+1]
            self.ls[i] = params['ls'][i]
            self.ones_M_ls[i] = np.outer(self.ones_M[i], 1.0 / np.exp(2*self.ls[i]))
            self.sf[i] = params['sf'][i]
            if not ((self.no_output_noise) and (i == self.no_layers-1)):
                self.sn[i] = params['sn'][i]
            triu_ind = np.triu_indices(Mi)
            diag_ind = np.diag_indices(Mi)
            if self.zu_tied:
                zi = params['zu'][i]
                self.zu[i] = zi
            else:
                for d in range(Douti):
                    zid = params['zu'][i][d]
                    self.zu[i][d] = zid

            for d in range(Douti):
                theta_m_d = params['eta2'][i][d]
                theta_R_d = params['eta1_R'][i][d]
                R = np.zeros((Mi, Mi))
                R[triu_ind] = theta_R_d.reshape(theta_R_d.shape[0], )
                R[diag_ind] = np.exp(R[diag_ind])
                self.theta_1_R[i][d] = R
                self.theta_1[i][d] = np.dot(R.T, R)
                self.theta_2[i][d] = theta_m_d
