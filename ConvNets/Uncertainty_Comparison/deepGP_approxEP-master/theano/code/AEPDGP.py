import sys
import math
import numpy as np
import scipy.linalg as npalg
import theano
import theano.tensor as T
import FITC_network
import copy
import matplotlib.pyplot as plt
import time
import pprint as pp
import random


class AEPDGP:
    def __init__(self, Ntrain, layer_sizes, no_pseudos,
                 mean_y_train, std_y_train):

        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # creat a network
        self.network = FITC_network.FITC_Network(Ntrain,
                                                 layer_sizes, no_pseudos)
        # create input and output variables in theano
        # deal with single output regression for now
        self.x = T.vector('x')
        self.y = T.scalar('y')

        # function to compute log Z and its derivatives
        self.logZ_SEP, self.mout_SEP, self.vout_SEP = self.network.compute_logZ_sep(self.x, self.y)
        derivatives = T.grad(cost=self.logZ_SEP, wrt=self.network.params_all)
        derivatives.append(self.logZ_SEP)
        derivatives.append(self.mout_SEP)
        derivatives.append(self.vout_SEP)
        self.get_grad_all = theano.function(
            [self.x, self.y],
            derivatives
        )

        # function to compute the predictive distributon
        self.predict_probabilistic = theano.function(
            [self.x],
            self.network.output_probabilistic(self.x)
        )

    def predict(self, X_test):
        mean = np.zeros(X_test.shape[0])
        variance = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            Xi = X_test[i, :]
            m, v = self.predict_probabilistic(Xi)
            m = m * self.std_y_train + self.mean_y_train
            v *= self.std_y_train**2
            mean[i] = m
            variance[i] = v

        return mean, variance

    def predict_noscaling(self, X_test):
        mean = np.zeros(X_test.shape[ 0 ])
        variance = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            Xi = X_test[i, :]
            m, v = self.predict_probabilistic(Xi)
            mean[i] = m
            variance[i] = v

        return mean, variance

    def compute_energy(self, params, X_train, y_train, n_per_mb):

        alpha = 1.0
        # update network with new hypers
        self.network.update_hypers(params)
        # update Kuu given new hypers
        self.network.compute_kuu()
        # compute mu and Su for each layer
        self.network.update_posterior()
        # compute muhat and Suhat for each layer
        self.network.compute_cavity()


        N_train = X_train.shape[0]
        D_in = X_train.shape[0]
        n_per_mb = np.min([n_per_mb, N_train])
        # reset gradient placeholders
        nolayers = self.network.nolayers
        grad_Kuuinv = []
        grad_ls = []
        grad_sf = []
        grad_sn = []
        grad_zu = []
        grad_Suhat = []
        grad_muhat = []
        grad_theta1 = []
        grad_theta2 = []
        grad_theta1_R = []
        for i in range(nolayers):
            Din = self.network.layers[i].Din
            M = self.network.layers[i].M
            Dout = self.network.layers[i].Dout
            grad_ls.append(np.zeros([Din, ]))
            grad_sf.append(np.zeros([1, ]))
            grad_sn.append(np.zeros([1, ]))

            grad_Kuuinv_i = [np.zeros([M, M]) for d in range(Dout)]
            grad_zu_i = [np.zeros([M, Din]) for d in range(Dout)]
            grad_Suhat_i = [np.zeros([M, M]) for d in range(Dout)]
            grad_muhat_i = [np.zeros([M, 1]) for d in range(Dout)]
            grad_theta1_i = [np.zeros([M, M]) for d in range(Dout)]
            grad_theta1_R_i = [np.zeros([M * (M + 1) / 2, 1]) for d in range(Dout)]
            grad_theta2_i = [np.zeros([M, 1]) for d in range(Dout)]
            grad_Kuuinv.append(grad_Kuuinv_i)
            grad_zu.append(grad_zu_i)
            grad_muhat.append(grad_muhat_i)
            grad_Suhat.append(grad_Suhat_i)
            grad_theta1.append(grad_theta1_i)
            grad_theta2.append(grad_theta2_i)
            grad_theta1_R.append(grad_theta1_R_i)


        logZi = 0
        #bind = np.random.choice(N_train, n_per_mb, replace=False)

        bind = np.asarray(random.sample(range(0, N_train), N_train))
 
        # permutation = np.asarray(random.sample(range(0,X.shape[0]), X.shape[0]))

        bX = X_train[bind, :]
        bY = y_train[bind]

        # loop through
        # TODO: parallel computation here
        for i in range(len(bX)):
            bXi = bX[i, :]
            bYi = bY[i]
            grad_all = self.get_grad_all(bXi, bYi)
            startind = 0
            for j in range(nolayers):
                Doutj = self.network.layers[j].Dout
                for d in range(Doutj):
                    grad_muhat[j][d] += grad_all[startind + 0]
                    grad_Suhat[j][d] += grad_all[startind + 1]
                    startind += 2
                grad_sf[j] += grad_all[startind + 0]
                grad_ls[j] += grad_all[startind + 1]
                grad_sn[j] += grad_all[startind + 2]
                startind += 3
                for d in range(Doutj):
                    grad_zu[j][d] += grad_all[startind + 0]
                    grad_Kuuinv[j][d] += grad_all[startind + 1]
                    startind += 2
            logZi += grad_all[-3]

        phi_prior = self.network.compute_phi_prior()
        phi_poste = self.network.compute_phi_posterior()
        phi_cavity = self.network.compute_phi_cavity()
        scale = N_train * 1.0 / n_per_mb / alpha

        energy = phi_prior + (N_train / alpha - 1) * phi_poste - N_train / alpha * phi_cavity - scale * logZi


        for i in range(nolayers):
            Douti = self.network.layers[i].Dout
            grad_ls[i] *= -scale
            grad_sf[i] *= -scale
            grad_sn[i] *= -scale
            for d in range(Douti):
                grad_zu[i][d] *= -scale


        factor = (N_train - alpha) * 1.0 / N_train
        for i in range(nolayers):
            layeri = self.network.layers[i]
            Mi = layeri.M
            triu_ind = np.triu_indices(Mi)
            diag_ind = np.diag_indices(Mi)
            for d in range(layeri.Dout):
                mud_val = layeri.mu[d].get_value()
                Sud_val = layeri.Su[d].get_value()
                muhatd = layeri.muhat[d].get_value()
                Suhatd = layeri.Suhat[d].get_value()
                Kuuinvid = layeri.Kuuinv_val[d]
                Kuuid = layeri.Kuu[d]
                # gradients of the prior contribution
                grad_Kuuinv_prior = - 0.5 * Kuuid

                # gradients of the posterior contribution
                theta2d = layeri.theta_2[d]
                grad_Suinv = - 0.5 * Sud_val + 0.5 * np.outer(mud_val, mud_val)
                grad_mu = np.dot(npalg.inv(Sud_val), mud_val)
                grad_Su_via_muhat = np.outer(grad_mu, theta2d)
                grad_Suinv_via_muhat = - np.dot(Sud_val, np.dot(grad_Su_via_muhat, Sud_val))
                grad_theta1_1 = grad_Suinv + grad_Suinv_via_muhat
                grad_theta2_1 = mud_val
                grad_Kuuinv_poste = grad_theta1_1

                # gradients of the cavity contribution
                grad_Suhatinv_cav = - 0.5 * Suhatd + 0.5 * np.outer(muhatd, muhatd)
                grad_muhat_cav = np.dot(npalg.inv(Suhatd), muhatd)
                grad_Suhat_via_muhat = factor * np.outer(grad_muhat_cav, theta2d)
                grad_Suhatinv_via_muhat = - np.dot(Suhatd, np.dot(grad_Suhat_via_muhat, Suhatd))
                grad_Suhatinv_cav_total = grad_Suhatinv_via_muhat + grad_Suhatinv_cav
                grad_theta1_3 = factor * grad_Suhatinv_cav_total
                grad_theta2_3 = factor * muhatd
                grad_Kuuinv_cavity = grad_Suhatinv_cav_total

                # gradient of logZ contribution
                grad_Suhat_via_muhat = factor * np.outer(grad_muhat[i][d], theta2d)
                grad_Suhat_total = grad_Suhat[i][d] + grad_Suhat_via_muhat
                grad_Suhatinv_total = - np.dot(Suhatd, np.dot(grad_Suhat_total, Suhatd))
                grad_theta1_2 = factor * grad_Suhatinv_total
                grad_theta2_2 = factor * np.dot(Suhatd, grad_muhat[i][d])


                grad_theta1[i][d] = (N_train / alpha - 1) * grad_theta1_1 - scale * grad_theta1_2 - N_train / alpha * grad_theta1_3
                grad_theta2[i][d] = (N_train / alpha - 1) * grad_theta2_1 - scale * grad_theta2_2 - N_train / alpha * grad_theta2_3

                # add eps here to avoid nan when factor = 0
                grad_Kuuinv_logZ = grad_Suhatinv_total + grad_Kuuinv[i][d]

                grad_Kuuinv_all = grad_Kuuinv_prior + (N_train / alpha - 1) * grad_Kuuinv_poste - scale * grad_Kuuinv_logZ - N_train / alpha * grad_Kuuinv_cavity
                

                # compute gradients wrt the cholesky factor
                theta1_R = layeri.theta_1_R[d]
                dtheta1 = grad_theta1[i][d]
                dtheta1_R = np.dot(theta1_R, dtheta1 + dtheta1.T)
                dtheta1_R[diag_ind] = dtheta1_R[diag_ind] * theta1_R[diag_ind]
                dtheta1_R = dtheta1_R[triu_ind]
                grad_theta1_R[i][d] = dtheta1_R.reshape((dtheta1_R.shape[0], 1))

                grad_Kuu = - np.dot(Kuuinvid, np.dot(grad_Kuuinv_all, Kuuinvid))
                grad_ls_viak = grad_Kuu * layeri.dKuudls[d]
                grad_sf_viak = grad_Kuu * layeri.dKuudsf[d]
                grad_ls_viak = np.sum(grad_ls_viak, 2).sum(1)
                grad_sf_viak = np.sum(grad_sf_viak, 1).sum(0)
                grad_zu_viak = grad_Kuu * layeri.dKuudzu[d]
                grad_zu_viak = (- np.sum(grad_zu_viak, 1) + np.sum(grad_zu_viak, 2))
                grad_zu_viak = grad_zu_viak.transpose([1, 0])

                # add the grads together
                grad_ls[i] += grad_ls_viak
                grad_sf[i] += grad_sf_viak
                grad_zu[i][d] += grad_zu_viak
            

        grad = {'ls': grad_ls, 'sf': grad_sf,
                'sn': grad_sn, 'zu': grad_zu,
                'eta1_R': grad_theta1_R, 'eta2': grad_theta2}
        
        return energy, grad

    def train(self, X_train, y_train, X_test, y_test, no_iters, n_per_mb,
              lrate=0.001, compute_test=False, reinit_hypers=True):
        adamobj = self.init_adam(adamlrate=lrate)
        if reinit_hypers:
            init_params = self.network.init_hypers(X_train, y_train)
            params = init_params
        else:
            params = self.network.get_hypers()
        ind = 1

        check = False
        test_nll = []
        test_rms = []
        train_ll = []
        try:
            while not check:
                energy, grad = self.compute_energy(params, X_train, y_train, n_per_mb)
                params = self.update_adam(adamobj, params, grad, ind, 1.0)
                # TODO: check convergence
                converged = False
                check = ind > no_iters or converged
                print_every = 10
                if compute_test and ind % print_every == 0:
                    self.network.compute_kuu()
                    my, vy = self.predict(X_test)
                    test_nlli = mll(y_test, my, vy, median=False)
                    test_rmsi = rmse(y_test, my, median=False)
                    print "training iter: %d, logZ: %.5f, test mnll: %.5f, test srms: %.5f" % (ind, energy, test_nlli, test_rmsi)
                    test_nll.append(test_nlli)
                    test_rms.append(test_rmsi)
                    train_ll.append(energy)
                elif ind % print_every == 0:
                    train_ll.append(energy)
                    print "training iter: %d, logZ: %.5f" % (ind, energy)

                ind += 1
        except KeyboardInterrupt:
            print "Keyboard interrupt exception"

        return test_nll, test_rms, train_ll

    def init_adam(self, adamlrate=0.001):
        alpha = {'ls': adamlrate,
                 'sf': adamlrate,
                 'sn': adamlrate,
                 'zu': adamlrate,
                 'eta1_R': adamlrate,
                 'eta2': adamlrate}
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        # init 1st moment and 2nd moment vectors
        v_zu = []
        v_ls = []
        v_sf = []
        v_sn = []
        v_eta1_R = []
        v_eta2 = []
        nolayers = self.network.nolayers
        for i in range(nolayers):
            layeri = self.network.layers[i]
            Mi = layeri.M
            Dini = layeri.Din
            Douti = layeri.Dout

            v_zui = [np.zeros((Mi, Dini)) for d in range(Douti)]
            v_eta1_Ri = [np.zeros((Mi * (Mi+1) / 2, 1)) for d in range(Douti)]
            v_eta2i = [np.zeros((Mi, 1)) for d in range(Douti)]
            v_eta1_R.append(v_eta1_Ri)
            v_eta2.append(v_eta2i)
            v_zu.append(v_zui)
            v_ls.append(np.zeros((Dini, )))
            v_sf.append(np.zeros((1, )))
            v_sn.append(np.zeros((1, )))
        v_all = {'zu': v_zu, 'ls': v_ls,
                 'sf': v_sf, 'sn': v_sn,
                 'eta1_R': v_eta1_R, 'eta2': v_eta2}

        m_all = copy.deepcopy(v_all)
        m_hat = copy.deepcopy(v_all)
        v_hat = copy.deepcopy(v_all)

        adamobj = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'eps': eps,
                   'm': m_all, 'v': v_all, 'm_hat': m_hat, 'v_hat': v_hat}

        return adamobj

    def update_adam(self, adamobj, params, grad, iterno, dec_rate=1.0):
        # update ADAM params and model params
        param_names1 = ['ls', 'sf', 'sn']
        param_names2 = ['zu', 'eta1_R', 'eta2']
        alpha = adamobj['alpha']
        beta1 = adamobj['beta1']
        beta2 = adamobj['beta2']
        eps = adamobj['eps']
        for i in range(self.network.nolayers):

            for name in param_names1:
                # get gradients
                g = grad[name][i]
                # compute running average of grad and grad^2
                # update biased first moment estimate
                adamobj['m'][name][i] = beta1 * adamobj['m'][name][i] + \
                                        (1.0 - beta1) * g
                # update biased second moment estimate
                adamobj['v'][name][i] = beta2 * adamobj['v'][name][i] + \
                                        (1.0 - beta2) * g**2.0
                # compute bias-corrected first and second moment estimates
                adamobj['m_hat'][name][i] = adamobj['m'][name][i] / (1 - beta1**iterno)
                adamobj['v_hat'][name][i] = adamobj['v'][name][i] / (1 - beta2**iterno)
                # update model params
                curval = params[name][i]
                delta = dec_rate * alpha[name] * adamobj['m_hat'][name][i] / \
                    (np.sqrt(adamobj['v_hat'][name][i]) + eps)
                # params[name][i] = curval + delta
                params[name][i] = curval - delta

            for name in param_names2:
                for d in range(self.network.layers[i].Dout):
                    # get gradients
                    g = grad[name][i][d]
                    # update biased first moment estimate
                    adamobj['m'][name][i][d] = beta1 * adamobj['m'][name][i][d] + \
                                            (1.0 - beta1) * g
                    # update biased second moment estimate
                    adamobj['v'][name][i][d] = beta2 * adamobj['v'][name][i][d] + \
                                            (1.0 - beta2) * g**2.0
                    # compute bias-corrected first and second moment estimates
                    adamobj['m_hat'][name][i][d] = adamobj['m'][name][i][d] / (1 - beta1**iterno)
                    adamobj['v_hat'][name][i][d] = adamobj['v'][name][i][d] / (1 - beta2**iterno)
                    # update model params
                    curval = params[name][i][d]
                    delta = dec_rate * alpha[name] * adamobj['m_hat'][name][i][d] / \
                            (np.sqrt(adamobj['v_hat'][name][i][d]) + eps)
                    # params[name][i][d] = curval + delta
                    params[name][i][d] = curval - delta

        return params

def mae(y, predicted):
    y = y.reshape((y.shape[0],))
    return np.mean(abs(y-predicted))


def rmse(y, predicted, trainstd=1.0, median=False):
    y = y.reshape((y.shape[0],))
    if median:
        return np.median((y-predicted)**2)**0.5/trainstd
    else:
        return np.mean((y-predicted)**2)**0.5/trainstd


def rmsle(y, predicted, trainstd=1.0):
    y = y.reshape((y.shape[0],))
    return np.mean((np.log(predicted+1)-np.log(y+1))**2)**0.5/trainstd


def mll(y, my, vy, mtrain=None, vtrain=None, median=False):
    y = y.reshape((y.shape[0],))
    ll = 0.5*np.log(2*math.pi*vy) + 0.5*(y-my)**2/vy
    if mtrain is None or vtrain is None:
        if median:
            return np.median(ll)
        else:
            return np.mean(ll)
    else:
        N = y.shape[0]
        lltrain = N/2.0*np.log(2*np.pi*vtrain) + 0.5*(y-mtrain)**2/vtrain
        if median:
            return np.median(ll - lltrain)
        else:
            return np.mean(ll - lltrain)
