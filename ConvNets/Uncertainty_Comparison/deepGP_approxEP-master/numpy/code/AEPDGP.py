import sys as Sys
# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    Sys.stdout.flush()
    if iteration == total:
        print("\n")


import sys
import math
import numpy as np
import scipy.linalg as npalg
import scipy.stats as stats
import FITC_network
from EQ_kernel import *
import copy
import matplotlib.pyplot as plt
import time
import pprint as pp
import multiprocessing as mp
import os
from tools import *


class AEPDGP:
    def __init__(self, Ntrain, layer_sizes, no_pseudos,
                 mean_y_train=None, std_y_train=None, lik='Gaussian', n_classes=None, n_samples=None, zu_tied=False):

        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train
        self.lik = lik
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.zu_tied = zu_tied

        # creat a network
        self.network = FITC_network.FITC_network(Ntrain,
                                                 layer_sizes, no_pseudos, 
                                                 lik=lik, n_classes=n_classes, 
                                                 zu_tied=self.zu_tied)

    def predict(self, X_test):
        self.network.update_posterior_for_prediction()
        if self.lik == 'Softmax':
            mean = np.zeros((X_test.shape[0], self.network.n_classes))
            variance = np.zeros((X_test.shape[0], self.network.n_classes))
            for i in range(X_test.shape[0]):
                Xi = X_test[i, :]
                m, v = self.network.output_probabilistic(Xi)
                mean[i, :] = m
                variance[i, :] = v
        else:
            mean = np.zeros([X_test.shape[0], self.network.layer_sizes[-1]])
            variance = np.zeros([X_test.shape[0], self.network.layer_sizes[-1]])
            for i in range(X_test.shape[0]):
                Xi = X_test[i, :]
                m, v = self.network.output_probabilistic(Xi)
                if self.lik == 'Gaussian':
                    m = m * self.std_y_train + self.mean_y_train
                    v *= self.std_y_train**2
                mean[i, :] = m
                variance[i, :] = v

        return mean, variance

    def predict_intermediate(self, X_test, layer_no):
        self.network.update_posterior_for_prediction()
        inter_dim = self.network.layer_sizes[layer_no]
        mean = np.zeros((X_test.shape[0], inter_dim))
        variance = np.zeros((X_test.shape[0], inter_dim))
        for i in range(X_test.shape[0]):
            Xi = X_test[i, :]
            m, v = self.network.output_probabilistic(Xi, inter_layer=layer_no)
            mean[i, :] = m
            variance[i, :] = v

        return mean, variance

    def predict_given_inputs(self, X, layer_no):
        self.network.update_posterior_for_prediction()
        inter_dim = self.network.layer_sizes[layer_no]
        mean = np.zeros((X.shape[0], inter_dim))
        variance = np.zeros((X.shape[0], inter_dim))
        for i in range(X.shape[0]):
            Xi = X[i, :]
            m, v = self.network.output_of_a_layer(Xi, layer_no=layer_no)
            mean[i, :] = m
            variance[i, :] = v

        return mean, variance


    def predict_noscaling(self, X_test):
        self.network.update_posterior_for_prediction()
        mean = np.zeros(X_test.shape[ 0 ])
        variance = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            Xi = X_test[i, :]
            m, v = self.network.output_probabilistic(Xi)
            mean[i] = m
            variance[i] = v

        return mean, variance

    def compute_energy(self, params, Xb, yb, N_train, compute_logZ=False):
        toprint = False
        N_batch = Xb.shape[0]
        network = self.network
        alpha = 1.0
        scale_logZ = - N_train * 1.0 / N_batch / alpha
        beta = (N_train - alpha) * 1.0 / N_train
        scale_poste = N_train * 1.0 / alpha - 1.0
        scale_cav = - N_train * 1.0 / alpha
        scale_prior = 1

        # scale_logZ = 1.0
        # scale_prior = 0.0
        # scale_cav = 0.0
        # scale_poste = 0.0

        zu_tied = network.zu_tied

        t0 = time.time()
        # update network with new hypers
        network.update_hypers(params)
        t1 = time.time()
        if toprint: print "update hypers: %f seconds" % (t1-t0)
        t0 = time.time()
        # update Kuu given new hypers
        network.compute_kuu()
        t1 = time.time()
        if toprint: print "compute kuu: %f seconds" % (t1-t0)
        t0 = time.time()
        # compute mu and Su for each layer
        network.update_posterior()
        t1 = time.time()
        if toprint: print "update posterior: %f seconds" % (t1-t0)
        t0 = time.time()
        # compute muhat and Suhat for each layer
        network.compute_cavity()
        t1 = time.time()
        if toprint: print "compute cavity: %f seconds" % (t1-t0)

        t0 = time.time()
        # reset gradient placeholders
        no_layers = network.no_layers
        grad_all = {}
        grad_all_names = {'ls', 'sf', 'sn', 'zu', 'eta1_R', 'eta2'}
        for name in grad_all_names:
            grad_all[name] = [[] for _ in range(no_layers)]

        for i in range(no_layers):
            M = network.no_pseudos[i]
            Din = network.layer_sizes[i]
            Dout = network.layer_sizes[i+1]
            
            grad_all['ls'][i] = np.zeros([Din, ])
            grad_all['sf'][i] = np.zeros([1, ])
            grad_all['sn'][i] = np.zeros([1, ])

            if zu_tied:
                grad_all['zu'][i] = np.zeros([M, Din])
            else:
                grad_all['zu'][i] = [np.zeros([M, Din]) for d in range(Dout)]

            grad_all['eta1_R'][i] = [np.zeros([M*(M+1)/2, ]) for d in range(Dout)]
            grad_all['eta2'][i] = [np.zeros([M, ]) for d in range(Dout)]

        grad_logZ = {}
        grad_logZ_names = {'ls', 'sf', 'sn', 'zu', 'Ahat', 'Bhat'}
        for name in grad_logZ_names:
            grad_logZ[name] = [[] for _ in range(no_layers)]

        for i in range(no_layers):
            M = network.no_pseudos[i]
            Din = network.layer_sizes[i]
            Dout = network.layer_sizes[i+1]
            grad_logZ['ls'][i] = [np.zeros([Din, ]) for d in range(Dout)]
            grad_logZ['sf'][i] = [np.zeros([1, ]) for d in range(Dout)]
            grad_logZ['sn'][i] = [np.zeros([1, ]) for d in range(Dout)]
            grad_logZ['zu'][i] = [np.zeros([M, Din]) for d in range(Dout)]
            grad_logZ['Bhat'][i] = [np.zeros([M, M]) for d in range(Dout)]
            grad_logZ['Ahat'][i] = [np.zeros([M, ]) for d in range(Dout)]
        
        t1 = time.time()
        if toprint: print "prep: %f seconds" % (t1-t0)

        epsilon = None
        if self.lik == 'Softmax':
            epsilon = np.random.normal(0, 1, (self.n_samples, self.n_classes))

        t0 = time.time()
        logZi_vec = []
        gradZi_vec = []
        for i in range(Xb.shape[0]):
            logZ_i, grad_i = network.compute_logZ_and_gradients(Xb[i, :], yb[i, :], epsilon=epsilon)
            logZi_vec.append(logZ_i)
            gradZi_vec.append(grad_i)

        t1 = time.time()
        if toprint: print "sequential minibatch: %f seconds" % (t1-t0)
        t0 = time.time()
        # collect output
        logZi = 0
        for n in range(Xb.shape[0]):
            logZi += logZi_vec[n]
            grad_n = gradZi_vec[n]
            for i in range(no_layers):
                M = network.no_pseudos[i]
                Din = network.layer_sizes[i]
                Dout = network.layer_sizes[i+1]
                for d in range(Dout):
                    for name in grad_logZ_names:
                        # print name, i, d
                        grad_logZ[name][i][d] += grad_n[name][i][d]
        t1 = time.time()
        if toprint: print "collecting output: %f seconds" % (t1-t0)
        t0 = time.time()
        for i in range(no_layers):
            Mi = network.no_pseudos[i]
            Din = network.layer_sizes[i]
            Dout = network.layer_sizes[i+1]
            triu_ind = np.triu_indices(Mi)
            diag_ind = np.diag_indices(Mi)
            Minner_i = 0
            for d in range(Dout):
                mu_id = network.mu[i][d]
                Su_id = network.Su[i][d]
                Spmm_id = network.Splusmm[i][d]
                muhat_id = network.muhat[i][d]
                Suhat_id = network.Suhat[i][d]
                Spmmhat_id = network.Splusmmhat[i][d]
                if zu_tied:
                    Kuuinv_id = network.Kuuinv[i]
                    Kuu_id = network.Kuu[i]
                else:
                    Kuuinv_id = network.Kuuinv[i][d]
                    Kuu_id = network.Kuu[i][d]

                
                dlogZ_dAhat_id = grad_logZ['Ahat'][i][d]
                dlogZ_dBhat_id = grad_logZ['Bhat'][i][d]
                dlogZ_dvcav_id = np.dot(Kuuinv_id, np.dot(dlogZ_dBhat_id, Kuuinv_id))
                if i == 0:
                    dlogZ_dmcav_id = np.dot(Kuuinv_id, dlogZ_dAhat_id)
                else:
                    dlogZ_dmcav_id = 2*np.dot(dlogZ_dvcav_id, muhat_id) + np.dot(Kuuinv_id, dlogZ_dAhat_id)

                
                theta2_id = network.theta_2[i][d]
                dlogZ_dvcav_via_mcav = beta * np.outer(dlogZ_dmcav_id, theta2_id)
                dlogZ_dvcav_id += dlogZ_dvcav_via_mcav
                dlogZ_dvcavinv_id = -np.dot(Suhat_id, np.dot(dlogZ_dvcav_id, Suhat_id))
                dlogZ_dtheta1_id = beta * dlogZ_dvcavinv_id
                dlogZ_dtheta2_id = beta * np.dot(Suhat_id, dlogZ_dmcav_id)
                dlogZ_dKuuinv_id_via_vcav = dlogZ_dvcavinv_id

                # get contribution of Ahat and Bhat to Kuu and add to Minner_id
                dlogZ_dKuuinv_id_via_Ahat = np.outer(dlogZ_dAhat_id, muhat_id)
                if i == 0:
                    Smmid = Suhat_id
                else:
                    Smmid = Spmmhat_id
                KuuinvSmmid = np.dot(Kuuinv_id, Smmid)
                dlogZ_dKuuinv_id_via_Bhat = 2*np.dot(KuuinvSmmid.T, dlogZ_dBhat_id) - dlogZ_dBhat_id
                dlogZ_dKuuinv_id = dlogZ_dKuuinv_id_via_Ahat + dlogZ_dKuuinv_id_via_Bhat + dlogZ_dKuuinv_id_via_vcav
                Minner_id = scale_poste * Spmm_id + scale_cav * Spmmhat_id - 2.0 * scale_logZ * dlogZ_dKuuinv_id
                

                grad_all['sf'][i] += scale_logZ * grad_logZ['sf'][i][d]
                grad_all['ls'][i] += scale_logZ * grad_logZ['ls'][i][d]
                grad_all['sn'][i] += scale_logZ * grad_logZ['sn'][i][d]
                if not zu_tied:
                    M_id = 0.5 * ( scale_prior * Kuuinv_id + np.dot(Kuuinv_id, np.dot(Minner_id, Kuuinv_id)) )
                    dhyp = d_trace_MKzz_dhypers(2*network.ls[i], 2*network.sf[i], network.zu[i][d], M_id, Kuu_id)

                    grad_all['sf'][i] += 2*dhyp[0]
                    grad_all['ls'][i] += 2*dhyp[1]
                    grad_all['zu'][i][d] = dhyp[2] + scale_logZ * grad_logZ['zu'][i][d]
                else:
                    Minner_i += Minner_id
                    grad_all['zu'][i] += scale_logZ * grad_logZ['zu'][i][d]
                    

                dphi_dtheta1_id = scale_poste * -0.5 * Spmm_id + scale_cav * beta * -0.5 * Spmmhat_id
                dphi_dtheta2_id = scale_poste * mu_id + scale_cav * beta * muhat_id


                dtheta1_id = scale_logZ * dlogZ_dtheta1_id + dphi_dtheta1_id
                dtheta2_id = scale_logZ * dlogZ_dtheta2_id + dphi_dtheta2_id
                theta1_R_id = network.theta_1_R[i][d]
                dtheta1_R_id = np.dot(theta1_R_id, dtheta1_id + dtheta1_id.T)
                dtheta1_R_id[diag_ind] = dtheta1_R_id[diag_ind] * theta1_R_id[diag_ind]
                dtheta1_R_id = dtheta1_R_id[triu_ind]
                grad_all['eta1_R'][i][d] = dtheta1_R_id.reshape((dtheta1_R_id.shape[0], ))

                grad_all['eta2'][i][d] = dtheta2_id.reshape((dtheta2_id.shape[0], ))

            if zu_tied:
                M_iall = 0.5 * ( scale_prior * Dout * Kuuinv_id + np.dot(Kuuinv_id, np.dot(Minner_i, Kuuinv_id)) )
                dhyp = d_trace_MKzz_dhypers(2*network.ls[i], 2*network.sf[i], network.zu[i], M_iall, Kuu_id)

                grad_all['sf'][i] += 2*dhyp[0]
                grad_all['ls'][i] += 2*dhyp[1]
                grad_all['zu'][i] += dhyp[2]

        t1 = time.time()
        if toprint: print "merge grads: %f seconds" % (t1-t0)
        if compute_logZ:

            t0 = time.time()
            phi_prior = network.compute_phi_prior()
            phi_poste = network.compute_phi_posterior()
            phi_cavity = network.compute_phi_cavity()
            energy = scale_prior * phi_prior + scale_poste * phi_poste + scale_cav * phi_cavity + scale_logZ * logZi

            # print 'alpha: ', alpha
            # print 'prior: ', scale_prior * phi_prior 
            # print 'poste:', scale_poste * phi_poste 
            # print 'cav: ', scale_cav * phi_cavity
            # print 'logZ:', scale_logZ * logZi
            t1 = time.time()
            if toprint: print "compute energy: %f seconds" % (t1-t0)
        else:
            energy = 0

        return energy, grad_all
    

    def train(self, X_train, y_train, X_test, y_test, no_epochs, n_per_mb,
              lrate=0.001, compute_test=False, reinit_hypers=True, compute_logZ=False):
        adamobj = self.init_adam(adamlrate=lrate)
        if reinit_hypers:
            if self.lik == 'Gaussian':
                init_params = self.network.init_hypers_Gaussian(X_train, y_train)
            elif self.lik == 'Probit':
                init_params = self.network.init_hypers_Probit(X_train)
            elif self.lik == 'Softmax':
                init_params = self.network.init_hypers_Softmax(X_train, y_train)

            params = init_params
        else:
            params = self.network.get_hypers()
        
        ind = 1
        check = False
        test_nll = []
        test_rms = []
        train_ll = []
        N_train = X_train.shape[0]
        batch_idxs = make_batches(N_train, n_per_mb)
        try:
            epoch = 0
            while (not check):
                printProgress(0, len(batch_idxs), prefix = 'Epoch %d:' % epoch, suffix = 'Complete', barLength = 20)
                permutation = np.random.choice(range(N_train), N_train, replace=False)
                for i, idxs in enumerate(batch_idxs):
                    X_mb = X_train[permutation[idxs], :]
                    y_mb = y_train[permutation[idxs], :]

                    energy, grad = self.compute_energy(params, X_mb, y_mb, N_train, compute_logZ=compute_logZ)
                    params = self.update_adam(adamobj, params, grad, ind, 1.0)
                    ind += 1
                    printProgress(i, len(batch_idxs), prefix = 'Epoch %d:' % epoch, suffix = 'Complete', barLength = 20)
                
                epoch += 1
                
                # TODO: check convergence
                converged = False
                check = (epoch >= no_epochs) or converged
                if compute_test:
                    # t0 = time.time()
                    self.network.compute_kuu()
                    my, vy = self.predict(X_test)

                    test_nlli = compute_test_nll(y_test, my, vy, self.lik, median=False)
                    test_rmsi = compute_test_error(y_test, my, self.lik, median=False)
                    print "\t logZ: %.5f, test mnll: %.5f, test rms: %.5f" % (energy, test_nlli, test_rmsi)
                    test_nll.append(test_nlli)
                    test_rms.append(test_rmsi)
                    train_ll.append(energy)
                    # t1 = time.time()
                    # print "prediction: %f seconds" % (t1-t0)
                else:
                    train_ll.append(energy)
                    pass
                    # print "\t logZ: %.5f" % (energy),

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'

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
        nolayers = self.network.no_layers
        for i in range(nolayers):
            Mi = self.network.no_pseudos[i]
            Dini = self.network.layer_sizes[i]
            Douti = self.network.layer_sizes[i+1]
            if self.zu_tied:
                v_zui = np.zeros((Mi, Dini))
            else:
                v_zui = [np.zeros((Mi, Dini)) for d in range(Douti)]
            v_eta1_Ri = [np.zeros((Mi * (Mi+1) / 2, )) for d in range(Douti)]
            v_eta2i = [np.zeros((Mi, )) for d in range(Douti)]
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
        param_names2 = ['eta1_R', 'eta2']
        if self.zu_tied:
            param_names1.append('zu')
        else:
            param_names2.append('zu')

        alpha = adamobj['alpha']
        beta1 = adamobj['beta1']
        beta2 = adamobj['beta2']
        eps = adamobj['eps']
        for i in range(self.network.no_layers):
            Douti = self.network.layer_sizes[i+1]

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
                for d in range(Douti):
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
