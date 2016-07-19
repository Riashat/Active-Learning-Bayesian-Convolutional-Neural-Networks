import sys
import theano
import theano.tensor as T
import numpy as np
import FITC_layer
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans2


class FITC_Network:
    def __init__(self, Ntrain, layer_sizes, no_pseudos):

        self.nolayers = len(no_pseudos)
        # create layers
        self.layers = []
        for i in range(self.nolayers):
            self.layers.append(
                FITC_layer.FITC_layer(Ntrain, layer_sizes[i:i+2], no_pseudos[i])
            )

        # collect parameters from all layers
        self.params_ls = []
        self.params_sf = []
        self.params_sn = []
        self.params_zu = []
        self.params_Kuuinv = []
        for layer in self.layers:
            self.params_ls.append(layer.ls)
            self.params_sf.append(layer.sf)
            self.params_sn.append(layer.sn)
            params_zui = []
            params_Kuuinvi = []
            for d in range(layer.Dout):
                params_zui.append(layer.zu[d])
                params_Kuuinvi.append(layer.Kuuinv[d])
            self.params_zu.append(params_zui)
            self.params_Kuuinv.append(params_Kuuinvi)

        self.params_all = []
        for i, layer in enumerate(self.layers):
            for d in range(layer.Dout):
                self.params_all.append(layer.muhat[d])
                self.params_all.append(layer.Suhat[d])
            self.params_all.append(layer.sf)
            self.params_all.append(layer.ls)
            self.params_all.append(layer.sn)
            for d in range(layer.Dout):
                self.params_all.append(layer.zu[d])
                self.params_all.append(layer.Kuuinv[d])

    def compute_phi_prior(self):
        logZ_prior = 0
        for i in range(self.nolayers):
            logZ_prior += self.layers[i].compute_phi_prior()
        return logZ_prior

    def compute_phi_posterior(self):
        logZ_posterior = 0
        for i in range(self.nolayers):
            logZ_posterior += self.layers[i].compute_phi_posterior()
        return logZ_posterior

    def compute_phi_cavity(self):
        phi_cavity = 0
        for i in range(self.nolayers):
            phi_cavity += self.layers[i].compute_phi_cavity()
        return phi_cavity

    def output_probabilistic_sep(self, m):
        v = T.zeros_like(m)
        # Recursively compute output
        for layer in self.layers:
            m, v = layer.output_probabilistic_sep(m, v)

        # deal with single output regression for now
        return m[0], v[0]

    def output_probabilistic(self, m):
        v = T.zeros_like(m)

        # Recursively compute output
        for layer in self.layers:
            m, v = layer.output_probabilistic(m, v)

        # deal with single output regression for now
        return m[0], v[0]

    def compute_logZ_sep(self, x, y):
        m, v = self.output_probabilistic_sep(x)
        logZ = -0.5 * (T.log(v) + (y - m)**2 / v)
        return logZ, m, v

    def compute_kuu(self):
        # compute Kuu for each layer
        for layer in self.layers:
            layer.compute_kuu()

    def compute_cavity(self):
        # compute muhat and Suhat for each layer
        for layer in self.layers:
            layer.compute_cavity()

    def update_posterior(self):
        # compute posterior approximation
        for layer in self.layers:
            layer.update_posterior()

    def init_hypers(self, x_train, y_train):
        # dict to hold hypers, inducing points and parameters of q(U)
        params = {'ls': [],
                  'sf': [],
                  'zu': [],
                  'sn': [],
                  'eta1_R': [],
                  'eta2': []}

        # first layer
        M1 = self.layers[0].M
        D1 = self.layers[0].Din
        Ntrain = x_train.shape[0]
        if Ntrain < 20000:
            centroids, label = kmeans2(x_train, M1, minit='points')
        else:
            randind = np.random.permutation(Ntrain)
            centroids = x_train[randind[0:M1], :]
        ls1 = self.estimate_ls(x_train)
        ls1[ls1 < -5] = -5
        sn1 = np.array([np.log(0.1*np.std(y_train))])
        sf1 = np.array([np.log(1*np.std(y_train))])
        sf1[sf1 < -5] = 0
        sn1[sn1 < -5] = np.log(0.1)
        params['sf'].append(sf1)
        params['ls'].append(ls1)
        params['sn'].append(sn1)
        zu0 = []
        eta1_R0 = []
        eta20 = []
        for d in range(self.layers[0].Dout):
            zu0.append(centroids)
            eta1_R0.append(np.random.randn(M1*(M1+1)/2, 1))
            eta20.append(np.random.randn(M1, 1))
        params['zu'].append(zu0)
        params['eta1_R'].append(eta1_R0)
        params['eta2'].append(eta20)

        # other layers
        for i in range(1, self.nolayers):
            Mi = self.layers[i].M
            Di = self.layers[i].Din
            sfi = np.log(np.array([1]))
            sni = np.log(np.array([0.1]))
            # zuii = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Di))
            lsi = np.ones((Di, ))
            params['sf'].append(sfi)
            params['ls'].append(lsi)
            params['sn'].append(sni)
            zui = []
            eta1_Ri = []
            eta2i = []
            for d in range(self.layers[i].Dout):
                zuii = np.tile(np.linspace(-1, 1, Mi).reshape((Mi, 1)), (1, Di))
                zui.append(zuii)
                eta1_Ri.append(np.random.randn(Mi*(Mi+1)/2, 1) / 10)
                eta2i.append(np.random.randn(Mi, 1) / 10)
            params['zu'].append(zui)
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
        for i in range(self.nolayers):
            layeri = self.layers[i]
            Mi = layeri.M
            Dini = layeri.Din
            Douti = layeri.Dout
            params['ls'].append(layeri.ls.get_value())
            params['sf'].append(layeri.sf.get_value())
            params['sn'].append(layeri.sn.get_value())
            triu_ind = np.triu_indices(Mi)
            diag_ind = np.diag_indices(Mi)
            params_zu_i = []
            params_eta2_i = []
            params_eta1_Ri = []
            for d in range(Douti):
                params_zu_i.append(layeri.zu[d].get_value())
                params_eta2_i.append(layeri.theta_2[d])

                Rd = layeri.theta_1_R[d]
                Rd[diag_ind] = np.log(Rd[diag_ind])
                params_eta1_Ri.append(Rd[triu_ind].reshape((Mi*(Mi+1)/2, 1)))

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
            X1 = X[randind[1:10000], :]

        d2 = compute_distance_matrix(X1)
        D = X1.shape[1]
        N = X1.shape[0]
        triu_ind = np.triu_indices(N)
        ls = np.zeros((D, ))
        for i in range(D):
            d2i = d2[:, :, i]
            d2imed = np.median(d2i[triu_ind])
            ls[i] = np.log(d2imed)
        return ls

    def update_hypers(self, params):
        for i in range(self.nolayers):
            layeri = self.layers[i]
            Mi = layeri.M
            Dini = layeri.Din
            Douti = layeri.Dout
            layeri.ls.set_value(params['ls'][i])
            layeri.sf.set_value(params['sf'][i])
            layeri.sn.set_value(params['sn'][i])
            triu_ind = np.triu_indices(Mi)
            diag_ind = np.diag_indices(Mi)
            for d in range(Douti):
                layeri.zu[d].set_value(params['zu'][i][d])
                theta_m_d = params['eta2'][i][d]
                theta_R_d = params['eta1_R'][i][d]
                R = np.zeros((Mi, Mi))
                R[triu_ind] = theta_R_d.reshape(theta_R_d.shape[0], )
                R[diag_ind] = np.exp(R[diag_ind])
                layeri.theta_1_R[d] = R
                layeri.theta_1[d] = np.dot(R.T, R)
                layeri.theta_2[d] = theta_m_d

def is_positive_definite(x):
    try:
        U = np.linalg.cholesky(x)
        return True
    except np.linalg.linalg.LinAlgError as e:
        print(e)
        return False
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

def is_any_nan(x):
    return np.any(np.isnan(x))

def compute_distance_matrix(x):
    diff = (x[:, None, :] - x[None, :, :])
    dist = abs(diff)
    return dist


