import numpy as np
import itertools
from scipy import ndimage
from tqdm import tqdm
import twixtools
from datetime import datetime
from copy import copy
import os
import pickle
import finufft
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.base import BaseEstimator, TransformerMixin  # This function just makes sure that the object is fitted
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
from PIL import Image
import pywt
import dask.array as da
from numpy.lib import stride_tricks

try:
    import cupy as cp
except:
    print("Could not import cupy")

from . import io_twixt
from .dictmodel import Dictionary

np.bool=np.bool_
asca = np.ascontiguousarray


class PCAComplex(BaseEstimator,TransformerMixin):

    def __init__(self, n_components_=None):
        self.n_components_=n_components_

    def fit(self, X):
        mean_X = np.mean(X, axis=0)
        cov_X = np.matmul(np.transpose((X - mean_X).conj()), (X - mean_X))

        if self.n_components_ is None:
            self.n_components_=cov_X.shape[0]

        X_val, X_vect = np.linalg.eigh(cov_X)

        sorted_index_X = np.argsort(X_val)[::-1]
        X_val = X_val[sorted_index_X]
        X_vect = X_vect[:, sorted_index_X]

        explained_variance_ratio = np.cumsum(X_val ** 2) / np.sum(X_val ** 2)
        if self.n_components_ is None:
            self.n_components_ = cov_X.shape[0]
        else :
            if self.n_components_<1:
                self.n_components_ = np.sum(explained_variance_ratio < self.n_components_) + 1
            else:
                self.n_components_=self.n_components_

        self.components_ = X_vect[:, :self.n_components_]
        self.singular_values_ = X_val[:self.n_components_]

        self.explained_variance_ratio_=explained_variance_ratio[:self.n_components_]
        self.mean_ = mean_X
        self.n_features_=X.shape[1]
        self.n_samples_=X.shape[0]

        return self


    def transform(self, X):
        try:
            # print("Checking cupy presence")
            import cupy as cp
            xp=cp.get_array_module(X)
            # print("Checked cupy presence")
        except ImportError:
            # print("Not using cupy in PCA transform")
            xp=np
            cp=None

        # xp = cp.get_array_module(X)
        
        check_is_fitted(self,'explained_variance_ratio_')

        #X = X.copy()  # This is so we do not make changes to the

        if xp==cp:
            components = cp.asarray(self.components_)
        else:
            components = self.components_

        return xp.matmul(X, components.conj())

    def plot_retrieved_signal(self,X,i=0,len=None,figsize=(15,10)):
        X_trans = self.transform(X)
        retrieved_X = np.matmul(X_trans,np.transpose(self.components_))
        plt.figure(figsize=figsize)
        if len is None:
            len = X.shape[-1]
        plt.plot(np.abs(X[i,:len]),label="Original")
        plt.plot(np.abs(retrieved_X[i, :len]), label="Retrieved")
        plt.legend()
        plt.show()




def combine_mrf_dict_components(mrfdict ,FF_list ,aggregate_components=True):
    '''
    Combine the water and fat components of the dictionary with FF_list to create a single dictionary with all the FFs
    '''
    ff = np.zeros(mrfdict.values.shape[:-1 ] +(len(FF_list),))
    ff_matrix =np.tile(np.array(FF_list) ,ff.shape[:-1 ] +(1,))

    water_signal =np.expand_dims(mrfdict.values[: ,: ,0] ,axis=-1 ) *(1-ff_matrix)
    fat_signal =np.expand_dims(mrfdict.values[: ,: ,1] ,axis=-1 ) *(ff_matrix)

    signal =water_signal +fat_signal

    signal_reshaped =np.moveaxis(signal ,-1 ,-2)
    signal_reshaped =signal_reshaped.reshape((-1 ,signal_reshaped.shape[-1]))

    keys_with_ff = list(itertools.product(mrfdict.keys, FF_list))
    keys_with_ff = [(*res, f) for res, f in keys_with_ff]

    return keys_with_ff,signal_reshaped


def build_phi(mrfdict,FFs=np.arange(0.1,1.09,0.1)):

    print("#########################################Building phi matrix from dictionary components")
    #mrfdict = dictsearch.Dictionary()
    print("Generating full dictionary")
    keys,values=combine_mrf_dict_components(mrfdict,FFs)

    import dask.array as da
    print("Performing svd")
    u,s,vh = da.linalg.svd(da.asarray(values))

    print("SVD done")
    vh=np.array(vh)
    s=np.array(s)

    # phi=vh[:L0]
    #phi=vh

    return vh



def match_signals_v2(all_signals,keys,pca_water,pca_fat,array_water_unique,array_fat_unique,transformed_array_water_unique,transformed_array_fat_unique,var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates,verbose,split,useGPU_dictsearch,mask,return_matched_signals=False):

    nb_signals = all_signals.shape[1]

    if remove_duplicates:
        all_signals, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)
        nb_signals = all_signals.shape[1]

    print("There are {} unique signals to match along {} water and {} fat components".format(nb_signals,
                                                                                             array_water_unique.shape[
                                                                                                 0],
                                                                                             array_fat_unique.shape[
                                                                                                 0]))




    num_group = int(nb_signals / split) + 1

    #idx_max_all_unique = []
    #alpha_optim = []

    if not(useGPU_dictsearch):
        idx_max_all_unique = np.zeros(nb_signals,dtype="int64")
        alpha_optim = np.zeros(nb_signals)
    else:
        idx_max_all_unique = cp.zeros(nb_signals,dtype="int64")
        alpha_optim = cp.zeros(nb_signals)


    if return_matched_signals:
        phase_optim = []
        J_optim = []


    for j in tqdm(range(num_group)):
        j_signal = j * split
        j_signal_next = np.minimum((j + 1) * split, nb_signals)

        if j_signal==j_signal_next:
            continue

        if verbose:
            print("PCA transform")
            start = datetime.now()

        if not (useGPU_dictsearch):

            if pca:
                transformed_all_signals_water = np.transpose(
                    pca_water.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))
                transformed_all_signals_fat = np.transpose(
                    pca_fat.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))

                sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                              transformed_all_signals_water.conj())
                sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                              transformed_all_signals_fat.conj())
            else:
                sig_ws_all_unique = np.matmul(array_water_unique, all_signals[:, j_signal:j_signal_next].conj())
                sig_fs_all_unique = np.matmul(array_fat_unique, all_signals[:, j_signal:j_signal_next].conj())


        else:

            if pca:

                transformed_all_signals_water = cp.transpose(
                    pca_water.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next])))).get()
                transformed_all_signals_fat = cp.transpose(
                    pca_fat.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next])))).get()

                sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                               cp.asarray(transformed_all_signals_water).conj())).get()
                sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                               cp.asarray(transformed_all_signals_fat).conj())).get()
            else:

                sig_ws_all_unique = (cp.matmul(cp.asarray(array_water_unique),
                                               cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()
                sig_fs_all_unique = (cp.matmul(cp.asarray(array_fat_unique),
                                               cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Extracting all sig_ws and sig_fs")
            start = datetime.now()

        if index_water_unique is not None:
            current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
            current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

        else:
            current_sig_ws_for_phase=sig_ws_all_unique
            current_sig_fs_for_phase=sig_fs_all_unique

        if verbose:
            end = datetime.now()
            print(end - start)

        if not (useGPU_dictsearch):

            if verbose:
                print("Adjusting Phase")
                print("Calculating alpha optim and flooring")

            A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
            B = (
                        current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

            a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
            b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
            c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

            discr = b ** 2 - 4 * a * c
            alpha1 = (-b + np.sqrt(discr)) / (2 * a)
            alpha2 = (-b - np.sqrt(discr)) / (2 * a)

            del a
            del b
            del c
            del discr

            current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                    1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2

            if verbose:
                start = datetime.now()

            apha_more_0=(current_alpha_all_unique>=0)
            alpha_less_1=(current_alpha_all_unique<=1)
            alpha_out_bounds=(1*(apha_more_0))*(1*(alpha_less_1))==0

            J_0=np.abs(current_sig_ws_for_phase)/np.sqrt(var_w)

            J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype("float")


            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                print("Calculating cost for all signals")
            start = datetime.now()


            J_all = np.abs((
                             1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)

            end = datetime.now()

            all_J = np.stack([J_all, J_0, J_1], axis=0)

            ind_max_J = np.argmax(all_J, axis=0)

            del all_J


            J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
            del J_0
            del J_1

            current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                        ind_max_J == 2) * 1

            idx_max_all_current = np.argmax(J_all, axis=0)
            current_alpha_all_unique_optim=current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]
            idx_max_all_unique[j_signal:j_signal_next]=idx_max_all_current
            alpha_optim[j_signal:j_signal_next]=current_alpha_all_unique_optim


            if return_matched_signals:
                d = (
                            1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current, np.arange(J_all.shape[1])] + current_alpha_all_unique_optim * current_sig_fs_for_phase[idx_max_all_current, np.arange(J_all.shape[1])]
                phase_adj = -np.arctan(d.imag / d.real)
                cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real

                del d

                phase_adj = (phase_adj) * (
                        1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                    1 * (cond) > 0)

                del cond

            if return_matched_signals:
                J_all_optim=J_all[idx_max_all_current, np.arange(J_all.shape[1])]


            del J_all
            del current_alpha_all_unique



        else:
            if verbose:
                print("Calculating alpha optim and flooring")
                start = datetime.now()

            current_sig_ws_for_phase = cp.asarray(current_sig_ws_for_phase)
            current_sig_fs_for_phase = cp.asarray(current_sig_fs_for_phase)

            ### Testing direct phase solving
            A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
            B = (
                        current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

            a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
            b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
            c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

            del A
            del B

            # del beta
            # del delta
            # del gamma
            # del nu

            discr = b ** 2 - 4 * a * c
            alpha1 = (-b + np.sqrt(discr)) / (2 * a)
            alpha2 = (-b - np.sqrt(discr)) / (2 * a)

            #################################################################################################################################""""
            del a
            del b
            del c
            del discr

            current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                    1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2

            # current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
            #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

            del alpha1
            del alpha2

            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                start = datetime.now()

            apha_more_0 = (current_alpha_all_unique >= 0)
            alpha_less_1 = (current_alpha_all_unique <= 1)
            alpha_out_bounds = (1 * (apha_more_0)) * (1 * (alpha_less_1)) == 0



            J_0 = cp.abs(current_sig_ws_for_phase) / cp.sqrt(var_w)
            J_1 = cp.abs(current_sig_fs_for_phase) / cp.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)

            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                print("Calculating cost for all signals")
                start = datetime.now()


            J_all = cp.abs((
                             1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)


            all_J = cp.stack([J_all, J_0, J_1], axis=0)

            ind_max_J = cp.argmax(all_J, axis=0)

            del all_J


            J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
            del J_0
            del J_1

            current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                    ind_max_J == 2) * 1

            idx_max_all_current = cp.argmax(J_all, axis=0)
            current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]

            idx_max_all_unique[j_signal:j_signal_next] = idx_max_all_current
            alpha_optim[j_signal:j_signal_next]=current_alpha_all_unique_optim

            

            if return_matched_signals:
                d = (
                            1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current, cp.arange(J_all.shape[1])] + current_alpha_all_unique_optim * current_sig_fs_for_phase[idx_max_all_current, cp.arange(J_all.shape[1])]
                phase_adj = -cp.arctan(d.imag / d.real)
                cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real

                del d

                phase_adj = (phase_adj) * (
                        1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                    1 * (cond) > 0)
                
                phase_adj=phase_adj.get()

                del cond

            del current_sig_ws_for_phase
            del current_sig_fs_for_phase


            if return_matched_signals:
                J_all_optim = J_all[idx_max_all_current, cp.arange(J_all.shape[1])]
                J_all_optim=J_all_optim.get()


            idx_max_all_current = idx_max_all_current.get()

            del J_all
            del current_alpha_all_unique


            if verbose:
                end = datetime.now()
                print(end - start)



        if verbose:
            print("Extracting index of pattern with max correl")
            start = datetime.now()

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Filling the lists with results for this loop")
            start = datetime.now()





        del current_alpha_all_unique_optim
        del idx_max_all_current


        if return_matched_signals:
            phase_optim.extend(phase_adj)
            J_optim.extend(J_all_optim)


        if verbose:
            end = datetime.now()
            print(end - start)


    if useGPU_dictsearch:
        idx_max_all_unique=idx_max_all_unique.get()
        alpha_optim=alpha_optim.get()

    if return_matched_signals:
        phase_optim = np.array(phase_optim)
        J_optim = np.array(J_optim)




    idx_max_all_unique=idx_max_all_unique.astype(int)
    params_all_unique = np.array(
        [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])

    if remove_duplicates:
        params_all = params_all_unique[index_signals_unique]
    else:
        params_all = params_all_unique

    del params_all_unique

    map_rebuilt = {
        "wT1": params_all[:, 0],
        "fT1": params_all[:, 1],
        "attB1": params_all[:, 2],
        "df": params_all[:, 3],
        "ff": params_all[:, 4]

    }
    



    
    if return_matched_signals:
        matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique, :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_unique, :].T * np.array(
                    alpha_optim).reshape(1, -1)
        matched_signals/=np.linalg.norm(matched_signals,axis=0)
        matched_signals *= J_optim*np.exp(1j*phase_optim)
        return map_rebuilt,None,None,matched_signals.squeeze()
    else:
        return map_rebuilt, None, None



def match_signals_v2_clustered_on_dico(all_signals_current,keys,pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique,var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique,useGPU_dictsearch,unique_keys,d_T1,d_fT1,d_B1,d_DF,labels,split,high_ff=False,return_cost=False):

    nb_clusters = unique_keys.shape[-1]

    keys = np.array(keys)
    unique_keys = np.array(unique_keys)


    nb_signals=all_signals_current.shape[-1]

    if not(useGPU_dictsearch):
        idx_max_all_unique_low_ff = np.zeros(nb_signals)
        alpha_optim_low_ff = np.zeros(nb_signals)
        if return_cost:
            J_optim = np.zeros(nb_signals)
            phase_optim=np.zeros(nb_signals)
    else:
        idx_max_all_unique_low_ff = cp.zeros(nb_signals,dtype="int64")
        alpha_optim_low_ff = cp.zeros(nb_signals)
        if return_cost:
            J_optim = cp.zeros(nb_signals)
            phase_optim=cp.zeros(nb_signals)


    if not (useGPU_dictsearch):
        for cl in tqdm(range(nb_clusters)):

            indices = np.argwhere(labels == cl)
            nb_signals_cluster=len(indices)
            num_group = int(nb_signals_cluster / split) + 1

            keys_T1 = (keys[:, 0] < unique_keys[:, cl][0] + d_T1) & ((keys[:, 0] > unique_keys[:, cl][0] - d_T1))
            keys_fT1 = (keys[:, 1] < unique_keys[:, cl][1] + d_fT1) & ((keys[:, 1] > unique_keys[:, cl][1] - d_fT1))
            keys_B1 = (keys[:, 2] < unique_keys[:, cl][2] + d_B1) & ((keys[:, 2] > unique_keys[:, cl][2] - d_B1))
            keys_DF = (keys[:, 3] < unique_keys[:, cl][3] + d_DF) & ((keys[:, 3] > unique_keys[:, cl][3] - d_DF))
            retained_signals = np.argwhere(keys_T1 & keys_fT1 & keys_B1 & keys_DF).flatten()


            var_w = var_w_total[retained_signals]
            var_f = var_f_total[retained_signals]
            sig_wf = sig_wf_total[retained_signals]

            all_signals_cluster=all_signals_current[:, indices.flatten()]
            idx_max_all_unique_cluster = []
            alpha_optim_cluster = []
            if return_cost:
                J_optim_cluster = np.zeros(nb_signals_cluster)
                phase_optim_cluster = np.zeros(nb_signals_cluster)

            for j in range(num_group):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals_cluster)
                if j_signal==j_signal_next:
                    continue

                transformed_all_signals_water = np.transpose(
                    pca_water.transform(np.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                transformed_all_signals_fat = np.transpose(
                    pca_fat.transform(np.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                              transformed_all_signals_water.conj())
                sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                              transformed_all_signals_fat.conj())
                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :][retained_signals]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :][retained_signals]
                A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                B = (
                            current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase
                a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag
                discr = b ** 2 - 4 * a * c
                alpha1 = (-b + np.sqrt(discr)) / (2 * a)
                alpha2 = (-b - np.sqrt(discr)) / (2 * a)
                del a
                del b
                del c
                del discr
                current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                        1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2

                apha_more_0 = (current_alpha_all_unique >= 0)
                alpha_less_1 = (current_alpha_all_unique <= 1)
                alpha_out_bounds = (1 * (apha_more_0)) * (1 * (alpha_less_1)) == 0

                if not(high_ff):
                    J_0 = np.abs(current_sig_ws_for_phase) / np.sqrt(var_w)
                J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)

                if not(high_ff):
                    current_alpha_all_unique[alpha_out_bounds] = np.argmax(
                    np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype(
                    "float")
                else:
                    current_alpha_all_unique[alpha_out_bounds] = 1

                J_all = np.abs((
                                       1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / np.sqrt(
                    (
                            1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                            1 - current_alpha_all_unique) * sig_wf)


                if not(high_ff):
                    all_J = np.stack([J_all, J_0, J_1], axis=0)
                else:
                    all_J = np.stack([J_all, J_1], axis=0)
                ind_max_J = np.argmax(all_J, axis=0)
                del all_J

                if not(high_ff):
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
                    del J_0
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                            ind_max_J == 2) * 1
                else:
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_1
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 1
                del J_1

                idx_max_all_current_sig = np.argmax(J_all, axis=0)
                current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                idx_max_all_unique_cluster.extend(idx_max_all_current_sig)
                alpha_optim_cluster.extend(current_alpha_all_unique_optim)

                if return_cost:
                    J_optim_cluster[j_signal:j_signal_next] = np.nan_to_num(J_all[idx_max_all_current_sig, np.arange(J_all.shape[1])] / np.linalg.norm(all_signals_cluster[:, j_signal:j_signal_next],axis=0))
                    d = (1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])] + current_alpha_all_unique_optim * \
                        current_sig_fs_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                    phase_adj = -np.arctan(d.imag / d.real)
                    cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
                    del d
                    phase_adj = (phase_adj) * (
                            1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                        1 * (cond) > 0)
                    phase_optim_cluster[j_signal:j_signal_next]=np.nan_to_num(phase_adj)


            if return_cost:
                J_optim[indices.flatten()]=J_optim_cluster
                phase_optim[indices.flatten()] = phase_optim_cluster

            idx_max_all_unique_low_ff[indices.flatten()] = (retained_signals[idx_max_all_unique_cluster])
            alpha_optim_low_ff[indices.flatten()] = (alpha_optim_cluster)

    else:
        for cl in tqdm(range(nb_clusters)):

            indices = cp.argwhere(labels == cl)
            nb_signals_cluster=len(indices)
            num_group = int(nb_signals_cluster / split) + 1


            keys_T1 = (keys[:, 0] < unique_keys[:, cl][0] + d_T1) & ((keys[:, 0] > unique_keys[:, cl][0] - d_T1))
            keys_fT1 = (keys[:, 1] < unique_keys[:, cl][1] + d_fT1) & ((keys[:, 1] > unique_keys[:, cl][1] - d_fT1))
            keys_B1 = (keys[:, 2] < unique_keys[:, cl][2] + d_B1) & ((keys[:, 2] > unique_keys[:, cl][2] - d_B1))
            keys_DF = (keys[:, 3] < unique_keys[:, cl][3] + d_DF) & ((keys[:, 3] > unique_keys[:, cl][3] - d_DF))
            retained_signals = cp.argwhere(keys_T1 & keys_fT1 & keys_B1 & keys_DF).flatten()

            var_w = var_w_total[retained_signals]
            var_f = var_f_total[retained_signals]
            sig_wf = sig_wf_total[retained_signals]

            all_signals_cluster=cp.asarray(all_signals_current[:, (indices.get()).flatten()])
            idx_max_all_unique_cluster = cp.zeros(nb_signals_cluster,dtype="int64")
            alpha_optim_cluster = cp.zeros(nb_signals_cluster)
            if return_cost:
                J_optim_cluster = cp.zeros(nb_signals_cluster)
                phase_optim_cluster = cp.zeros(nb_signals_cluster)

            for j in range(num_group):
                j_signal = j * split
                j_signal_next =cp.minimum((j + 1) * split, nb_signals_cluster)

                if j_signal==j_signal_next:
                    continue


                transformed_all_signals_water = cp.transpose(
                    pca_water.transform(cp.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                transformed_all_signals_fat = cp.transpose(
                    pca_fat.transform(cp.transpose(all_signals_cluster[:, j_signal:j_signal_next])))
                sig_ws_all_unique = cp.matmul(cp.asarray(transformed_array_water_unique),
                                              transformed_all_signals_water.conj())
                sig_fs_all_unique = cp.matmul(cp.asarray(transformed_array_fat_unique),
                                              transformed_all_signals_fat.conj())
                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :][retained_signals]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :][retained_signals]
                A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                B = (
                            current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase
                a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag
                discr = b ** 2 - 4 * a * c
                alpha1 = (-b + cp.sqrt(discr)) / (2 * a)
                alpha2 = (-b - cp.sqrt(discr)) / (2 * a)
                del a
                del b
                del c
                del discr
                current_alpha_all_unique = (1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (
                        1 - (1 * (alpha1 >= 0) & (alpha1 <= 1))) * alpha2
                # current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)
                apha_more_0 = (current_alpha_all_unique >= 0)
                alpha_less_1 = (current_alpha_all_unique <= 1)
                alpha_out_bounds = (1 * (apha_more_0)) * (1 * (alpha_less_1)) == 0

                if not(high_ff):
                    J_0 = cp.abs(current_sig_ws_for_phase) / cp.sqrt(var_w)
                J_1 = cp.abs(current_sig_fs_for_phase) / cp.sqrt(var_f)

                if not(high_ff):
                    current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)
                else:
                    current_alpha_all_unique[alpha_out_bounds] = 1

                J_all = cp.abs((
                                       1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / cp.sqrt(
                    (
                            1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                            1 - current_alpha_all_unique) * sig_wf)


                if not(high_ff):
                    all_J = cp.stack([J_all, J_0, J_1], axis=0)
                else:
                    all_J = cp.stack([J_all, J_1], axis=0)
                ind_max_J = cp.argmax(all_J, axis=0)
                del all_J

                if not(high_ff):
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
                    del J_0
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                            ind_max_J == 2) * 1
                else:
                    J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_1
                    current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 1
                del J_1

                idx_max_all_current_sig = cp.argmax(J_all, axis=0)
                current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current_sig, cp.arange(J_all.shape[1])]

                idx_max_all_unique_cluster[j_signal:j_signal_next]=idx_max_all_current_sig
                alpha_optim_cluster[j_signal:j_signal_next]=current_alpha_all_unique_optim

                if return_cost:
                    J_optim_cluster[j_signal:j_signal_next] = cp.nan_to_num(J_all[idx_max_all_current_sig, cp.arange(J_all.shape[1])] / cp.linalg.norm(all_signals_cluster[:, j_signal:j_signal_next],axis=0))
                    d = (1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current_sig, cp.arange(J_all.shape[1])] + current_alpha_all_unique_optim * \
                        current_sig_fs_for_phase[idx_max_all_current_sig, cp.arange(J_all.shape[1])]
                    phase_adj = -cp.arctan(d.imag / d.real)
                    cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real
                    del d
                    phase_adj = (phase_adj) * (
                            1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                        1 * (cond) > 0)
                    phase_optim_cluster[j_signal:j_signal_next]=cp.nan_to_num(phase_adj)



            idx_max_all_unique_low_ff[indices.flatten()] = (retained_signals[idx_max_all_unique_cluster])
            alpha_optim_low_ff[indices.flatten()] = (alpha_optim_cluster)
            if return_cost:
                J_optim[indices.flatten()]=J_optim_cluster
                phase_optim[indices.flatten()] = phase_optim_cluster



        idx_max_all_unique_low_ff=idx_max_all_unique_low_ff.get()
        alpha_optim_low_ff = alpha_optim_low_ff.get()
        if return_cost:
            J_optim=J_optim.get()
            phase_optim=phase_optim.get()


    if return_cost:
        return idx_max_all_unique_low_ff,alpha_optim_low_ff,J_optim,phase_optim

    return idx_max_all_unique_low_ff,alpha_optim_low_ff

class Optimizer(object):

    def __init__(self,mask=None,verbose=False,useGPU=False,**kwargs):
        self.paramDict=kwargs
        self.paramDict["useGPU"]=useGPU
        self.mask=mask
        self.verbose=verbose


    def search_patterns(self,dictfile,volumes,retained_timesteps=None):
        #takes as input dictionary pattern and an array of images or volumes and outputs parametric maps
        raise ValueError("search_patterns should be implemented in child")

class SimpleDictSearch(Optimizer):

    def __init__(self,seq=None,split=500,pca=True,threshold_pca=15,useGPU_dictsearch=False,remove_duplicate_signals=False,threshold=None,return_matched_signals=True,volumes_type="raw",**kwargs):
        
        super().__init__(**kwargs)
        self.paramDict["split"] = split
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = int(threshold_pca)
        self.paramDict["remove_duplicate_signals"] = remove_duplicate_signals
        self.paramDict["return_matched_signals"] = return_matched_signals


        self.paramDict["useGPU_dictsearch"]=useGPU_dictsearch
        self.paramDict["threshold"]=threshold

        if volumes_type not in ["singular", "raw"]:
            raise ValueError('volumes_type must be either "singular" or "raw".')
        
        self.paramDict["volumes_type"]=volumes_type


    def search_patterns_test_multi(self, dicofull_file, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        volumes_type=self.paramDict["volumes_type"]
        

        verbose = self.verbose
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]

        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]

        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        # if pca and (type()==str):
        #     pca_file = str.split(dictfile, ".dict")[0] + "_{}pca_simple.pkl".format(threshold_pca)
        #     pca_file_name = str.split(pca_file, "/")[-1]

        # if type(dictfile)==str:
        #     vars_file = str.split(dictfile, ".dict")[0] + "_vars_simple.pkl".format(threshold_pca)
        #     vars_file_name = str.split(vars_file, "/")[-1]
        #     path = str.split(os.path.realpath(__file__), "/utils_mrf.py")[0]

        if volumes.ndim > 2:
            all_signals = volumes[:, mask > 0]
        else:  # already masked
            all_signals = volumes

        ntimesteps=volumes.shape[0]

        all_signals=all_signals.astype("complex64")


        del volumes

        with open(dicofull_file, "rb") as file:
            dicofull = pickle.load(file)

        if volumes_type == "raw":
            mrfdict = dicofull["mrfdict_light"]

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        elif volumes_type=="singular":  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dicofull["mrfdict_light_L0{}".format(threshold_pca)][0]
            array_fat = dicofull["mrfdict_light_L0{}".format(threshold_pca)][1]
            keys = dicofull["mrfdict_light_L0{}".format(threshold_pca)][2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        ntimesteps_dico=array_water.shape[-1]

        if not(ntimesteps_dico==ntimesteps):
            raise ValueError("The dictionary and the incoming signal did not have the same number of timesteps: ntimesteps_dico {} != ntimesteps_signal {}".format(ntimesteps_dico,ntimesteps))


        # array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        # array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        # if not(volumes_type=="raw")or("vars_light" not in dicofull.keys()) or ((pca) and ("pca_light_{}".format(threshold_pca) not in dicofull.keys())):

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)



        del array_water
        del array_fat

        if pca:
            if not(volumes_type=="raw") or ("pca_light_{}".format(threshold_pca) not in dicofull.keys()):
                pca_water = PCAComplex(n_components_=threshold_pca)
                pca_fat = PCAComplex(n_components_=threshold_pca)

                pca_water.fit(array_water_unique)
                pca_fat.fit(array_fat_unique)

                transformed_array_water_unique = pca_water.transform(array_water_unique)
                transformed_array_fat_unique = pca_fat.transform(array_fat_unique)
                if volumes_type=="raw":
                    dicofull["pca_light_{}".format(threshold_pca)] = (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)
                    with open(dicofull_file, "wb") as file:
                        pickle.dump(dicofull, file)

            else:
                print("Loading pca")
                (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)=dicofull["pca_light_{}".format(threshold_pca)]
 
        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        if not(volumes_type=="raw") or ("vars_light" not in dicofull.keys()):
            var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
            var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
            sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                            axis=1).real

            var_w = var_w[index_water_unique]
            var_f = var_f[index_fat_unique]

            var_w = np.reshape(var_w, (-1, 1))
            var_f = np.reshape(var_f, (-1, 1))
            sig_wf = np.reshape(sig_wf, (-1, 1))
            if volumes_type == "raw":
                dicofull["vars_light"] = (var_w, var_f, sig_wf, index_water_unique, index_fat_unique)
                with open(dicofull_file, "wb") as file:
                    pickle.dump(dicofull, file)
                
        else:
            print("Loading var w / var f / sig wf")
            (var_w, var_f, sig_wf, index_water_unique, index_fat_unique)=dicofull["vars_light"] 

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(1))
        

        print("Calculating optimal fat fraction and best pattern per signal")
        if not (self.paramDict["return_matched_signals"]):
            map_rebuilt, J_optim, phase_optim = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                 array_water_unique, array_fat_unique,
                                                                 transformed_array_water_unique,
                                                                 transformed_array_fat_unique, var_w, var_f,
                                                                 sig_wf, pca, index_water_unique,
                                                                 index_fat_unique, remove_duplicates, verbose,split, useGPU_dictsearch, mask
                                                                 )
        else:
            map_rebuilt, J_optim, phase_optim, matched_signals = match_signals_v2(all_signals, keys, pca_water,
                                                                                  pca_fat,
                                                                                  array_water_unique,
                                                                                  array_fat_unique,
                                                                                  transformed_array_water_unique,
                                                                                  transformed_array_fat_unique,
                                                                                  var_w, var_f, sig_wf,
                                                                                  pca, index_water_unique,
                                                                                  index_fat_unique,
                                                                                  remove_duplicates, verbose, split,
                                                                                  useGPU_dictsearch, mask,                                                                            
                                                                                  return_matched_signals=True)



        print("Maps built")


        values_results.append((map_rebuilt, mask))



        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))





    def search_patterns_test_multi_2_steps_dico(self, dicofull_file, volumes, signal=None, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        volumes_type=self.paramDict["volumes_type"]

        if "clustering" not in self.paramDict:
            self.paramDict["clustering"]=True

        split = self.paramDict["split"]
        pca = self.paramDict["pca"]

        if volumes.ndim==5:
            ntimesteps=volumes.shape[1]
        else:
            ntimesteps=volumes.shape[0]

        threshold_pca = self.paramDict["threshold_pca"]
        
        threshold_pca=np.minimum(ntimesteps,threshold_pca)

        threshold_ff=self.paramDict["threshold_ff"]
        # dictfile_light=self.paramDict["dictfile_light"]

        if "return_cost" not in self.paramDict:
            self.paramDict["return_cost"]=False
        return_cost = self.paramDict["return_cost"]

        # if "calculate_matched_signals" not in self.paramDict:
        #     self.paramDict["calculate_matched_signals"]=False
        self.paramDict["calculate_matched_signals"]=True
        calculate_matched_signals = self.paramDict["calculate_matched_signals"]

        if "return_matched_signals" not in self.paramDict:
            self.paramDict["return_matched_signals"]=False
        

        return_matched_signals = self.paramDict["return_matched_signals"]




        if calculate_matched_signals:
            return_cost=True

        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]


        # if pca and (type(dictfile)==dict):
        #     pca_file = str.split(dictfile, ".dict")[0] + "_{}pca.pkl".format(threshold_pca)
        #     pca_file_name = str.split(pca_file, "/")[-1]

        # if type(dictfile)==str:
        #     vars_file = str.split(dictfile, ".dict")[0] + "_vars.pkl".format(threshold_pca)
        #     vars_file_name=str.split(vars_file,"/")[-1]
        #     path=str.split(os.path.realpath(__file__),"/utils_mrf.py")[0]

        # print(path)
        # print(vars_file_name)


        if signal is None:

            if volumes.ndim > 2:
                
                all_signals = volumes[:, mask > 0]
                
            else:  # already masked
                all_signals = volumes

        else :
            all_signals = signal
            all_signals=all_signals.reshape(-1,1)

        all_signals=all_signals.astype("complex64")
        nb_signals=all_signals.shape[1]



        del volumes

        with open(dicofull_file, "rb") as file:
                dicofull = pickle.load(file)

        if volumes_type == "raw":
            
            mrfdict = dicofull["mrfdict"]
            # mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]
            keys=np.array(keys)

            del mrfdict
        elif volumes_type=="singular":  # otherwise dictfile contains {"mrfdict":(s_w,s_f,keys),"mrfdict_light":(s_w_light,s_f_light,keys_light)}

            array_water = dicofull["mrfdict_L0{}".format(threshold_pca)][0]
            array_fat = dicofull["mrfdict_L0{}".format(threshold_pca)][1]
            keys = dicofull["mrfdict_L0{}".format(threshold_pca)][2]
            

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]
        
        ntimesteps_dico=array_water.shape[-1]

        if not(ntimesteps_dico==ntimesteps):
            raise ValueError("The dictionary and the incoming signal did not have the same number of timesteps: ntimesteps_dico {} != ntimesteps_signal {}".format(ntimesteps_dico,ntimesteps))



        # if not(volumes_type=="raw")or("vars" not in dicofull.keys()) or ((pca) and ("pca_{}".format(threshold_pca) not in dicofull.keys())) or (calculate_matched_signals):

            # print("Calculating unique dico signals")
        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)


        if not(volumes_type=="raw") or ("vars" not in dicofull.keys()):

            var_w_total = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
            var_f_total = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
            sig_wf_total = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                                  axis=1).real
            var_w_total = var_w_total[index_water_unique]
            var_f_total = var_f_total[index_fat_unique]
            var_w_total = np.reshape(var_w_total, (-1, 1))
            var_f_total = np.reshape(var_f_total, (-1, 1))
            sig_wf_total = np.reshape(sig_wf_total, (-1, 1))
            
            if volumes_type=="raw":
                dicofull["vars"]=(var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique)
                with open(dicofull_file,"wb") as file:
                    pickle.dump(dicofull,file)
        else:
            print("Loading var w / var f / sig wf")
            
            (var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique)=dicofull["vars"]

        if pca:
            if not(volumes_type=="raw") or ("pca_{}".format(threshold_pca) not in dicofull.keys()):
                pca_water = PCAComplex(n_components_=threshold_pca)
                pca_fat = PCAComplex(n_components_=threshold_pca)

                pca_water.fit(array_water_unique)
                pca_fat.fit(array_fat_unique)

                transformed_array_water_unique = pca_water.transform(array_water_unique)
                transformed_array_fat_unique = pca_fat.transform(array_fat_unique)
                if volumes_type=="raw":
                    dicofull["pca_{}".format(threshold_pca)]=(pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique)    
                    with open(dicofull_file,"wb") as file:
                        pickle.dump(dicofull,file)
                    
            else:
                print("Loading pca")
                (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)=dicofull["pca_{}".format(threshold_pca)] 
        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None



        if useGPU_dictsearch:
            var_w_total = cp.asarray(var_w_total)
            var_f_total = cp.asarray(var_f_total)
            sig_wf_total = cp.asarray(sig_wf_total)
            keys=cp.asarray(keys)

        values_results = []
        keys_results = list(range(1))

        print("Calculating optimal fat fraction and best pattern per signal")

        if self.paramDict["clustering"]:
            #Trick to avoid returning matched signals in the coarse dictionary matching step
            return_matched_signals_backup=self.paramDict["return_matched_signals"]
            self.paramDict["return_matched_signals"]=False

            print("Preliminary dictionary matching for clustering")
            all_maps_bc_cf_light = self.search_patterns_test_multi(dicofull_file,all_signals)

            self.paramDict["return_matched_signals"] = return_matched_signals_backup

            ind_high_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] >= threshold_ff)
            ind_low_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] < threshold_ff)
            all_maps_low_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_low_ff] for k in list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
            all_maps_high_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_high_ff] for k in
                                         list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
            

            # if matching one only signal 
            if all_maps_low_ff.ndim == 1:
                all_maps_low_ff = all_maps_low_ff.reshape(-1,1)

            if all_maps_high_ff.ndim == 1:
                all_maps_high_ff = all_maps_low_ff.reshape(-1,1)

            unique_keys, labels = np.unique(all_maps_low_ff, axis=-1, return_inverse=True)
            #nb_clusters = unique_keys.shape[-1]
            unique_keys_high_ff, labels_high_ff = np.unique(all_maps_high_ff, axis=-1, return_inverse=True)



            idx_max_all_unique = np.zeros(nb_signals)
            alpha_optim = np.zeros(nb_signals)
            if return_cost:
                J_optim = np.zeros(nb_signals)
                phase_optim = np.zeros(nb_signals)

            if useGPU_dictsearch:
                unique_keys=cp.asarray(unique_keys)
                labels = cp.asarray(labels)
                unique_keys_high_ff = cp.asarray(unique_keys_high_ff)
                labels_high_ff = cp.asarray(labels_high_ff)

            all_signals_low_ff = all_signals[:, ind_low_ff.flatten()]
            all_signals_high_ff = all_signals[:, ind_high_ff.flatten()]

            d_T1 = 400
            d_fT1 = 101
            d_B1 = 0.2
            d_DF = 0.030  # 0.015

            if return_cost:
                idx_max_all_unique_low_ff, alpha_optim_low_ff,J_optim_low_ff,phase_optim_low_ff = match_signals_v2_clustered_on_dico(all_signals_low_ff,
                                                                                                                                     keys, pca_water,
                                                                                                                                     pca_fat,
                                                                                                                                     transformed_array_water_unique,
                                                                                                                                     transformed_array_fat_unique,
                                                                                                                                     var_w_total,
                                                                                                                                     var_f_total,
                                                                                                                                     sig_wf_total,
                                                                                                                                     index_water_unique,
                                                                                                                                     index_fat_unique,
                                                                                                                                     useGPU_dictsearch,
                                                                                                                                     unique_keys, d_T1,
                                                                                                                                     d_fT1,
                                                                                                                                     d_B1, d_DF, labels,
                                                                                                                                     split, False,return_cost=True)

            else:
                idx_max_all_unique_low_ff,alpha_optim_low_ff=match_signals_v2_clustered_on_dico(all_signals_low_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
                                                                                                transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
                                                                                                index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys, d_T1, d_fT1,
                                                                                                d_B1, d_DF, labels,split,False)

            d_T1 = 400
            d_fT1 = 101
            d_B1 = 0.2
            d_DF = 0.030  # 0.015


            if return_cost:
                idx_max_all_unique_high_ff, alpha_optim_high_ff,J_optim_high_ff,phase_optim_high_ff = match_signals_v2_clustered_on_dico(
                    all_signals_high_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
                    transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
                    index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys_high_ff, d_T1, d_fT1,
                    d_B1, d_DF, labels_high_ff, split, True,return_cost=True)
            else:
                idx_max_all_unique_high_ff,alpha_optim_high_ff=match_signals_v2_clustered_on_dico(all_signals_high_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
                                                                                                  transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
                                                                                                  index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys_high_ff, d_T1, d_fT1,
                                                                                                  d_B1, d_DF, labels_high_ff,split,True)



            idx_max_all_unique[ind_low_ff.flatten()] = idx_max_all_unique_low_ff
            idx_max_all_unique[ind_high_ff.flatten()] = idx_max_all_unique_high_ff

            alpha_optim[ind_low_ff.flatten()] = alpha_optim_low_ff
            alpha_optim[ind_high_ff.flatten()] = alpha_optim_high_ff

            if return_cost:
                J_optim[ind_low_ff.flatten()] = J_optim_low_ff
                J_optim[ind_high_ff.flatten()] = J_optim_high_ff

                phase_optim[ind_low_ff.flatten()] = phase_optim_low_ff
                phase_optim[ind_high_ff.flatten()] = phase_optim_high_ff
                matched_signals = array_water_unique[index_water_unique, :][idx_max_all_unique.astype(int), :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_unique.astype(int),
                                                                    :].T * np.array(alpha_optim).reshape(1, -1)
                rho_optim= J_optim*np.linalg.norm(all_signals,axis=0)/np.linalg.norm(matched_signals, axis=0)
                np.save("rho_optim.npy", rho_optim)

            if calculate_matched_signals:
                matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique.astype(int), :].T * (1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][idx_max_all_unique.astype(int), :].T * np.array(alpha_optim).reshape(1, -1)
                matched_signals *=np.linalg.norm(all_signals,axis=0)/np.linalg.norm(matched_signals, axis=0)
                matched_signals *= J_optim * np.exp(1j * phase_optim)
                np.save("matched_signals_coarse_dico.npy", matched_signals)
                print("Matched signals saved to matched_signals_coarse_dico.npy")



            if useGPU_dictsearch:
                keys=keys.get()

            keys_for_map = [tuple(k) for k in keys]

            params_all_unique = np.array(
                [keys_for_map[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique.astype(int))])
            
            map_rebuilt = {
                "wT1": params_all_unique[:, 0],
                "fT1": params_all_unique[:, 1],
                "attB1": params_all_unique[:, 2],
                "df": params_all_unique[:, 3],
                "ff": params_all_unique[:, 4]

            }
            if return_cost:
                if not(return_matched_signals):
                    values_results.append((map_rebuilt, mask,J_optim,phase_optim,rho_optim))
                else:
                    values_results.append((map_rebuilt, mask,J_optim,phase_optim,rho_optim,matched_signals))
            else:
                values_results.append((map_rebuilt, mask))

        else:
            #Trick to avoid returning matched signals in the coarse dictionary matching step
            return_matched_signals_backup=self.paramDict["return_matched_signals"]



            if calculate_matched_signals:
                all_maps,matched_signals = self.search_patterns_test_multi(dicofull_file,all_signals)

            else:
                all_maps = self.search_patterns_test_multi(dicofull_file,all_signals)

            map_rebuilt=all_maps[0][0]
            mask=all_maps[0][1]

            if return_cost:
                if not(return_matched_signals):
                    values_results.append((map_rebuilt, mask,None,None))
                else:
                    values_results.append((map_rebuilt, mask,None,None,matched_signals))
            else:
                values_results.append((map_rebuilt, mask))

        print("Maps built")

        return dict(zip(keys_results, values_results))



def build_dico_seqParams(filename,index=-1):
    
    hdr = io_twixt.parse_twixt_header(filename)
 
    alFree = get_specials(hdr, type="alFree",index=index)
    adFree = get_specials(hdr, type="adFree",index=index)
    geometry, is3D, orientation, offset = get_volume_geometry(hdr,index=index)
    
    protocol=hdr[index]["tProtocolName"]
    print(protocol)
    
    if (protocol=="T1_mapping")or("raFin_1400Seg_1400Interleaved" in protocol)or("T1MAP" in protocol)or("customIR_Reco" in protocol)or("T1_MAP" in protocol)or("MRF2D" in protocol)or("raFin_FreeMax" in protocol):
        nb_segments=alFree[3]
    else:
        nb_segments=alFree[4]

    # print(nb_segments)
    if is3D:
        print("3D data")
        x_FOV = hdr[index]['sSliceArray.asSlice[0].dReadoutFOV']
        y_FOV = hdr[index]['sSliceArray.asSlice[0].dReadoutFOV']
        z_FOV = hdr[index]['sSliceArray.asSlice[0].dThickness']
        nb_part = hdr[index]['sKSpace.lPartitions']
        minTE = hdr[index]["alTE[0]"] / 1e3
        echoSpacing = adFree[1]
        dTR = echoSpacing - minTE
        total_TR = hdr[index]["alTR[0]"] / 1e6
        invTime = adFree[0]

        if np.max(np.argwhere(alFree> 0)) >= 16:
            use_navigator_dll = True
        else:
            use_navigator_dll = False
        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV,"z_FOV": z_FOV, "TI": invTime, "total_TR": total_TR,
                            "dTR": dTR, "is3D": is3D, "orientation": orientation, "nb_part": nb_part, "offset": offset,"use_navigator_dll":use_navigator_dll,"nb_segments":nb_segments}
        dico_seqParams.update(geometry)
        
    else:
        # print("2D data")

        
        

        x_FOV = hdr[index]['sSliceArray.asSlice[0].dReadoutFOV']
        y_FOV = hdr[index]['sSliceArray.asSlice[0].dReadoutFOV']
        minTE=hdr[index]["alTE[0]"]/1e3
        echoSpacing=adFree[0]
        dTR=echoSpacing-minTE
        total_TR=hdr[index]["alTR[0]"]/1e6
        invTime=adFree[1]

        
        
        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV,"TI":invTime,"total_TR":total_TR,"dTR":dTR,"offset": offset,"is3D": is3D, "orientation": orientation,"nb_segments":nb_segments}
        dico_seqParams.update(geometry)

    return dico_seqParams



def get_specials(hdr,type="alFree",index=-1):
    if type=="alFree":
        dtype=int
    elif type=="adFree":
        dtype=float
    else:
        raise ValueError("type should be alFree or adFree")


    sWipMemBlock_keys=list_hdr_keys(hdr,"sWipMemBlock.{}[".format(type),index=index)
    sWipMemBlock_idx=[int(str.split(str.split(k,"[")[-1],"]")[0]) for k in sWipMemBlock_keys]

    count_sWimMemBlock=hdr[index]['sWipMemBlock.{}.__attribute__.size'.format(type)]
    sWipMemBlock=np.zeros(count_sWimMemBlock,dtype=dtype)
    for i,idx in enumerate(sWipMemBlock_idx):
        sWipMemBlock[idx]=hdr[index][sWipMemBlock_keys[i]]


    return sWipMemBlock

def get_volume_geometry(hdr_input, index=-1,is_spherical=False):
    '''
    'sSliceArray.asSlice[0].dThickness',
    'sSliceArray.asSlice[0].dPhaseFOV',
    'sSliceArray.asSlice[0].dReadoutFOV',
    'sSliceArray.asSlice[0].sPosition.dSag',
    'sSliceArray.asSlice[0].sPosition.dCor',
    'sSliceArray.asSlice[0].sPosition.dTra',
    'sSliceArray.asSlice[0].sNormal.dTra',
    '''
    if type(hdr_input) == str:
        print("Getting geometry info for {}".format(hdr_input))
        headers = io_twixt.parse_twixt_header(hdr_input)
    else:
        # print("Input is not a file - assuming the header was passed directly")
        headers = hdr_input

    header=headers[index]
    if not has_volume_geometry(header):
        raise ValueError("header has no geometry information")

    # center of first slice

    pos_dSag=header['sSliceArray.asSlice[0].sPosition.dSag'] if 'sSliceArray.asSlice[0].sPosition.dSag' in header else 0.
    pos_dCor=header['sSliceArray.asSlice[0].sPosition.dCor'] if 'sSliceArray.asSlice[0].sPosition.dCor' in header else 0.
    pos_dTra=header['sSliceArray.asSlice[0].sPosition.dTra'] if 'sSliceArray.asSlice[0].sPosition.dTra' in header else 0.

    offset=header['lScanRegionPosTra']

    position = (
        pos_dSag,
        pos_dCor,
        pos_dTra#+offset,
    )

    # volume shape in pixel
    protocol_name=header["tProtocolName"]
    #print(protocol_name)
    is3D="3D" in protocol_name
    if is3D:
        if is_spherical:
            print("Spherical : assuming isotropic resolution")
            nb_slices=header['sKSpace.lBaseResolution']
        else:
            nb_slices=header["sKSpace.lPartitions"]

    else:
        nb_slices=header['sSliceArray.lSize']
    shape = (
        header['sKSpace.lBaseResolution'],
        header['sKSpace.lBaseResolution'],
        nb_slices,
    )

    # print("Readout FOV: {}".format(header['sSliceArray.asSlice[0].dReadoutFOV']))
    # print("Phase FOV : {}".format(header['sSliceArray.asSlice[0].dPhaseFOV']))

    
    if shape[-1]==1:
        spacing_z=header['sSliceArray.asSlice[0].dThickness']
    elif is3D:
        if is_spherical:
            print("Spherical : assuming isotropic resolution")
            spacing_z=header['sSliceArray.asSlice[0].dReadoutFOV']/shape[2]
        else:
            spacing_z=header["sSliceArray.asSlice[0].dThickness"]/nb_slices
    else:    
        spacing_z=header['sSliceArray.asSlice[1].sPosition.dTra']-header['sSliceArray.asSlice[0].sPosition.dTra']



    spacing = (
        header['sSliceArray.asSlice[0].dReadoutFOV']/shape[0], # which field to use ???
        header['sSliceArray.asSlice[0].dReadoutFOV']/shape[1], #??
        spacing_z,
    )
    
    if ('sSliceArray.lCor' in header.keys()) and header['sSliceArray.lCor']==1:
        order_axis=[0,2,1]
        orientation="coronal"
    elif ('sSliceArray.lSag' in header.keys()) and header['sSliceArray.lSag']==1:
        order_axis=[2,0,1]
        orientation="sagittal"
    else:
        order_axis=[0,1,2]
        orientation="transversal"
    
    spacing=tuple(np.array(spacing)[order_axis])
    shape=tuple(np.array(shape)[order_axis])


    if is3D:
        origin = (
            position[0] - shape[0]/2 * spacing[0],
            position[1] - shape[1]/2 * spacing[1],
            position[2] - shape[2]/2 * spacing[2],#- spacing[2] / 2,
        )
    else:

        origin = (
            position[0] - shape[0]/2 * spacing[0],
            position[1] - shape[1]/2 * spacing[1],
            position[2] ,#- spacing[2] / 2,
        )

    geom={
        'origin': origin,
        'spacing': spacing,
        # 'transform': transform, # TODO
    }
    return geom,is3D,orientation,offset

def has_volume_geometry(header):
    fields = ['sKSpace.lBaseResolution']
    return all(field in header for field in fields)

def read_rawdata_2D(filename):
    dico_seqParams=build_dico_seqParams(filename)
    nb_segments=dico_seqParams["nb_segments"]
    print("Number of segments: {}".format(nb_segments))

    twix = twixtools.read_twix(filename,parse_pmu=False)
    mdb_list = twix[-1]['mdb']
    data= []
    for i, mdb in enumerate(mdb_list):
        if mdb.is_image_scan():
            data.append(mdb)

    data = np.array([mdb.data for mdb in data])

    data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
    data=np.moveaxis(data,2,1)

    return data,dico_seqParams


def build_single_image_multichannel(kdata,trajectory,size,density_adj=True,eps=1e-6,b1=None,useGPU=False,light_memory_usage=False,is_theta_z_adjusted=False,normalize_volumes=False):
    '''

    :param kdata: shape nchannels*ntimesteps*point_per_timestep
    :param trajectory: shape ntimesteps * point_per_timestep * ndim (2 or 3)
    :param size: image size
    :param density_adj:
    :param eps:
    :param b1: coil sensitivity map
    :return: mask of size size
    '''
    volume_rebuilt=simulate_radial_undersampled_images_multi(kdata,trajectory,size,density_adj,eps,is_theta_z_adjusted,b1,1,useGPU,None,light_memory_usage,True)[0]
    return volume_rebuilt



def simulate_radial_undersampled_images_multi(kdata, trajectory, size, density_adj=True, eps=1e-6,
                                              is_theta_z_adjusted=False, b1=None, ntimesteps=175, useGPU=False,
                                              memmap_file=None, light_memory_usage=False,
                                              normalize_volumes=True,normalize_iterative=False):
    # Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)

    # if light_memory_usage and not(useGPU):
    #    print("Warning : light memory usage is not used without GPU")
    traj = trajectory.get_traj_for_reconstruction(ntimesteps)

    nb_channels = len(kdata)

    if not (len(kdata[0]) == len(traj)):
        kdata = kdata.reshape(nb_channels, len(traj), -1)


    #print(traj[0].shape)
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes = trajectory.paramDict["total_nspokes"]

    #print(kdata.shape)
    nspoke = int(nb_allspokes / ntimesteps)

    num_samples=traj.shape[1]

    if not (is_theta_z_adjusted):
        dtheta = np.pi / nspoke
        dz = 1 / trajectory.paramDict["nb_rep"]

    else:
        dtheta = 1
        dz = 1/(2*np.pi)



    if type(density_adj) is bool:
        if density_adj:
            density_adj="Radial"

    if density_adj=="Radial":
        density = np.abs(np.linspace(-1, 1, npoint))
        #density=np.expand_dims(axis=0)
        for j in tqdm(range(nb_channels)):
            kdata[j] =[(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata[j]]

    # elif density_adj=="Voronoi":
    #     print("Calculating Voronoi Density Adj")
    #     density=[]
    #     for i in tqdm(range(len(traj))):
    #         curr_dens=voronoi_volumes_freud(traj[i])
    #         curr_dens_shape=curr_dens.shape
    #         curr_dens=curr_dens.reshape(-1,npoint)
    #         curr_dens[:,0]=curr_dens[:,1]
    #         curr_dens[:, npoint-1] = curr_dens[:, npoint-2]
    #         curr_dens=curr_dens.reshape(curr_dens_shape)
    #         curr_dens /= curr_dens.sum()
    #         density.append(curr_dens)

    #     # density = [
    #     #     voronoi_volumes_freud(traj[i]) for i in
    #     #     tqdm(range(len(traj)))]
    #     for j in tqdm(range(nb_channels)):
    #         kdata[j] = [k * density[i] for i, k in enumerate(kdata[j])]

    if kdata[0][0].dtype == "complex64":
        try:
            traj=traj.astype("float32")
        except:
            for i in range(traj.shape[0]):
                traj[i] = traj[i].astype("float32")
        print(traj[0].dtype)

    if not(normalize_iterative):
        for i in tqdm(range(nb_channels)):
            kdata[i] *= dz * dtheta / (2*npoint)
    else:
        for i in tqdm(range(nb_channels)):
            kdata[i]/=num_samples

    # kdata = (normalize_image_series(np.array(kdata)))

    output_shape = (ntimesteps,) + size

    flushed = False

    if memmap_file is not None:
        from tempfile import mkdtemp
        import os.path as path
        file_memmap = path.join(mkdtemp(), "memmap_volumes.dat")
        images_series_rebuilt = np.memmap(file_memmap, dtype="complex64", mode="w+", shape=output_shape)

    else:
        images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    print("Performing NUFFT")
    if traj[0].shape[-1] == 2:  # 2D

        for i, t in tqdm(enumerate(traj)):
            fk = finufft.nufft2d1(asca(t[:, 0]), asca(t[:, 1]), asca(np.squeeze(kdata[:, i, :])), size)

            # images_series_rebuilt = np.moveaxis(images_series_rebuilt, 0, 1)
            if b1 is None:
                print(fk.shape)
                if fk.ndim>2:
                    images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                else:
                    print('Taking abs of image')
                    images_series_rebuilt[i]=np.abs(fk)
            else:
                images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

    elif traj[0].shape[-1] == 3:  # 3D
        if not (useGPU):

            for i, t in tqdm(enumerate(traj)):
                if not (light_memory_usage):
                    fk = finufft.nufft3d1(asca(t[:, 2]), asca(t[:, 0]), asca(t[:, 1]), asca(kdata[:, i, :]), size)
                    if b1 is None:
                        images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                    else:
                        images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

                else:
                    flush_condition = (memmap_file is not None) and (
                                (psutil.virtual_memory().cached + psutil.virtual_memory().free) / 1e9 < 2) and (
                                          not (flushed))
                    if flush_condition:
                        print("Flushed Memory")
                        offset = i * images_series_rebuilt.itemsize * i * np.prod(images_series_rebuilt.shape[1:])
                        i0 = i
                        new_shape = (output_shape[0] - i,) + output_shape[1:]
                        images_series_rebuilt.flush()
                        del images_series_rebuilt
                        flushed = True
                        normalize_volumes = False
                        images_series_rebuilt = np.memmap(memmap_file, dtype="complex64", mode="r+", shape=new_shape,
                                                          offset=offset)

                    if flushed:
                        for j in tqdm(range(nb_channels)):
                            print(t.shape)
                            print(kdata[j][i].shape)
                            fk = finufft.nufft3d1(
                                asca(t[:, 2]), asca(t[:, 0]), asca(t[:, 1]), 
                                asca(kdata[j][i]), size)
                            if b1 is None:
                                images_series_rebuilt[i - i0] += np.abs(fk) ** 2
                            else:
                                images_series_rebuilt[i - i0] += b1[j].conj() * fk

                        if b1 is None:
                            images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])

                    else:
                        for j in tqdm(range(nb_channels)):

                            #index_non_zero_kdata=np.nonzero(kdata[j][i])
                            #kdata_current=kdata[j][i][index_non_zero_kdata]
                            #t_current=t[index_non_zero_kdata]
                            kdata_current = asca(kdata[j][i])
                            #print(t_current.shape)
                            #print(kdata_current.shape)
                            fk = finufft.nufft3d1(asca(t[:, 2]), asca(t[:, 0]), asca(t[:, 1]), kdata_current, size)
                            if b1 is None:
                                images_series_rebuilt[i] += np.abs(fk) ** 2
                            else:
                                images_series_rebuilt[i] += b1[j].conj() * fk

                        if b1 is None:
                            images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])
        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            for i in tqdm(list(range(kdata[0].shape[0]))):
                if not (light_memory_usage):
                    fk_gpu = GPUArray((nb_channels, N1, N2, N3), dtype=complex_dtype)
                    c_retrieved = kdata[:, i, :]
                    kx = traj[i][:, 0]
                    ky = traj[i][:, 1]
                    kz = traj[i][:, 2]

                    # Cast to desired datatype.
                    kx = kx.astype(dtype)
                    ky = ky.astype(dtype)
                    kz = kz.astype(dtype)
                    c_retrieved = c_retrieved.astype(complex_dtype)

                    # Allocate memory for the uniform grid on the GPU.
                    c_retrieved_gpu = to_gpu(c_retrieved)

                    # Initialize the plan and set the points.
                    plan = cufinufft(1, (N1, N2, N3), nb_channels, eps=eps, dtype=dtype)
                    plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

                    # Execute the plan, reading from the strengths array c and storing the
                    # result in fk_gpu.
                    plan.execute(c_retrieved_gpu, fk_gpu)

                    fk = np.squeeze(fk_gpu.get())

                    fk_gpu.gpudata.free()
                    c_retrieved_gpu.gpudata.free()

                    if b1 is None:
                        images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                    else:
                        images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

                    plan.__del__()
                else:
                    # fk = np.zeros(output_shape,dtype=complex_dtype)
                    for j in tqdm(range(nb_channels)):
                        fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                        index_non_zero_kdata=np.nonzero(kdata[j][i])
                        c_retrieved = kdata[j][i][index_non_zero_kdata]
                        kx = traj[i][index_non_zero_kdata][:, 0]
                        ky = traj[i][index_non_zero_kdata][:, 1]
                        kz = traj[i][index_non_zero_kdata][:, 2]

                        # Cast to desired datatype.
                        kx = kx.astype(dtype)
                        ky = ky.astype(dtype)
                        kz = kz.astype(dtype)
                        c_retrieved = c_retrieved.astype(complex_dtype)

                        # Allocate memory for the uniform grid on the GPU.
                        c_retrieved_gpu = to_gpu(c_retrieved)

                        # Initialize the plan and set the points.
                        plan = cufinufft(1, (N1, N2, N3), 1, eps=eps, dtype=dtype)
                        plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

                        # Execute the plan, reading from the strengths array c and storing the
                        # result in fk_gpu.
                        plan.execute(c_retrieved_gpu, fk_gpu)

                        fk = np.squeeze(fk_gpu.get())

                        fk_gpu.gpudata.free()
                        c_retrieved_gpu.gpudata.free()

                        if b1 is None:
                            images_series_rebuilt[i] += np.abs(fk) ** 2
                        else:
                            images_series_rebuilt[i] += b1[j].conj() * fk
                        plan.__del__()

                    if b1 is None:
                        images_series_rebuilt[i] = np.sqrt(images_series_rebuilt[i])

        del kdata
        gc.collect()

        if flushed:
            images_series_rebuilt.flush()
            del images_series_rebuilt
            images_series_rebuilt = np.memmap(memmap_file, dtype="complex64", mode="r", shape=output_shape)

        if (normalize_volumes) and (b1 is not None) and not(normalize_iterative):
            print("Normalizing by Coil Sensi")
            if light_memory_usage:
                b1_norm = np.sum(np.abs(b1) ** 2,axis=0)
                for i in tqdm(range(images_series_rebuilt.shape[0])):
                    images_series_rebuilt[i] /= b1_norm
            else:
                images_series_rebuilt /= np.expand_dims(np.sum(np.abs(b1) ** 2,axis=0),axis=0)


    # images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return images_series_rebuilt




def groupby(arr, n, axis=0, mode="edge"):
    """ group array into groups of size 'n' """

    ngroup = -(-arr.shape[axis] // n)
    if arr.shape[axis] % n != 0:
        # pad array
        padding = [(0,0)] * arr.ndim
        nzero = n - np.mod(arr.shape[axis], n)
        padding[axis] = (nzero//2, -(-nzero//2))
        arr = np.pad(arr, padding, mode=mode)
    arr = np.moveaxis(arr, axis, 0)
    arr = arr.reshape((ngroup, -1) + arr.shape[1:])
    return list(np.moveaxis(arr, 1, axis + 1))


def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new


def build_mask_from_volume(volumes,threshold_factor=0.05,iterations=2):
    mask = False
    unique = np.histogram(np.abs(volumes), 1000)[1]
    mask = mask | (np.abs(volumes) > unique[int(len(unique) * threshold_factor)])
    if iterations>0:
        mask = safe_binary_closing(mask,iterations=iterations)
    
    # print(mask.shape)
    return mask*1


def safe_binary_closing(volume, structure=None, iterations=0):
    """
    Applies binary closing to an nD volume while preserving the border slices by using edge padding.

    Parameters:
    - volume (ndarray): The input binary array (n-dimensional).
    - structure (ndarray or None): Structuring element. If None, a cross-shaped element is used.
    - iterations (int): Number of times to apply the closing operation.

    Returns:
    - ndarray: The binary closed volume with preserved borders.
    """
    # Determine the number of dimensions
    n_dims = volume.ndim

    # Define padding: 1 layer on each side for all axes
    pad_width = [(1, 1)] * n_dims  # [(1,1), (1,1), ..., (1,1)] for nD

    # Pad the volume using edge values to prevent border artifacts
    padded_volume = np.pad(volume, pad_width=pad_width, mode='edge')
    # Apply binary closing
    closed_padded = ndimage.binary_closing(padded_volume, structure=structure, iterations=iterations)

    # Remove the padding to restore the original shape
    slices = tuple(slice(1, -1) for _ in range(n_dims))  # (slice(1,-1), slice(1,-1), ...)
    closed_volume = closed_padded[slices]

    return closed_volume


def list_hdr_keys(hdr,filter,index=-1):
    return [k for k in hdr[index] if filter in k]




def plot_image_grid(list_images,nb_row_col,figsize=(10,10),title="",cmap=None,save_file=None,same_range=False,aspect=None):
    fig = plt.figure(figsize=figsize)

    plt.title(title)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nb_row_col,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    if same_range:
        vmin=np.min(np.array(list_images))
        vmax=np.max(np.array(list_images))
        for ax, im in zip(grid, list_images):
            ax.imshow(im,cmap=cmap,vmin=vmin,vmax=vmax,aspect=aspect)
    else:
        for ax, im in zip(grid, list_images):
            ax.imshow(im,cmap=cmap,aspect=aspect)

    if save_file is not None:
        plt.savefig(save_file)
    else:
        # plt.show()
        return fig


def volumes_to_gif(volumes, index=None):
    index = index or slice(None)
    volslices = [np.abs(vol[index]) for vol in volumes]
    images = []
    for i in range(len(volslices)):
        img = Image.fromarray((volslices[i] / volslices[i].max() * 255).astype('uint8'), 'L')
        img = img.convert("P")
        images.append(img)
    return images  



def build_mask_single_image_multichannel(kdata,trajectory,image_size,b1=None,density_adj=False,eps=1e-6,threshold_factor=1/25,useGPU=False,light_memory_usage=False,is_theta_z_adjusted=False,selected_spokes=None,normalize_volumes=True):
    '''

    :param kdata: shape nchannels*ntimesteps*point_per_timestep
    :param density_adj:
    :param eps:
    :param b1: coil sensitivity map
    :return: mask of size size
    '''

    mask = False

    if (selected_spokes is not None):
        trajectory_for_mask = copy(trajectory)
        #selected_spokes = np.r_[20:800,1200:1400]
        trajectory_for_mask.traj = trajectory.get_traj()[selected_spokes]
    else:
        trajectory_for_mask = trajectory

    if (selected_spokes is not None):
        volume_rebuilt = build_single_image_multichannel(kdata[:,selected_spokes,:,:],trajectory_for_mask,image_size,density_adj,eps,b1,useGPU=useGPU,light_memory_usage=light_memory_usage,is_theta_z_adjusted=is_theta_z_adjusted,normalize_volumes=normalize_volumes)
    else:
        volume_rebuilt = build_single_image_multichannel(kdata, trajectory_for_mask, image_size,
                                                         density_adj, eps, b1, useGPU=useGPU,
                                                         light_memory_usage=light_memory_usage,
                                                         is_theta_z_adjusted=is_theta_z_adjusted,normalize_volumes=normalize_volumes)

    traj = trajectory.get_traj_for_reconstruction(timesteps=1)


    if traj.shape[-1]==2: # For slices

        # if threshold_factor is None:
        #     threshold_factor = 1/7

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique) *threshold_factor)])
        #mask = ndimage.binary_closing(mask, iterations=3)


    elif traj.shape[-1]==3: # For volumes

        # if threshold_factor is None:
        #     threshold_factor = 1/20

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique) *threshold_factor)])
        mask = ndimage.binary_closing(mask, iterations=3)

    return mask




def convertArrayToImageHelper(dico,data,apply_offset=False,reorient=True):

    spacing=dico["spacing"]
    origin=dico["origin"]
    orientation=dico["orientation"]
    is3D=dico["is3D"]

    if apply_offset:

        offset=dico["offset"]
        print("Applying offset {}".format(offset))
        
        origin=np.array(origin)
        origin[-1]=origin[-1]+offset
        origin=tuple(origin)
    

    geom={"origin":origin,"spacing":spacing}
    # print(geom)

    if data.ndim==2:
        data=data[None,...]
    # print(data.shape)
    
    # curr_map=np.flip(np.moveaxis(curr_map,0,2),axis=(0,1,2))

    if reorient:
        print("Reorienting input volume")
        offset=data.ndim-3
        if orientation=="coronal":
            print("WARNING: coronal orientation not tested - should be checked")
            data=np.flip(np.moveaxis(data,(offset,offset+1,offset+2),(offset+1,offset+2,offset)),axis=(offset,offset+1))
                #data=np.moveaxis(data,0,2)
        elif orientation=="transversal":
                # data=np.moveaxis(data,offset,offset+2)
            data=np.flip(np.moveaxis(data,offset+1,offset+2),axis=(offset+1))
            # data=np.flip(np.moveaxis(data,offset+1,offset+2))
        elif orientation=="sagittal":
                # data=np.moveaxis(data,offset,offset+2)
            print("WARNING: sagittal orientation not tested - should be checked")
            data=np.flip(np.moveaxis(data,(offset,offset+1,offset+2),(offset,offset+2,offset+1)))
    
    return data,geom



def cutup(data, blck, strd):
    """
    Extracts overlapping patches from an ND array using strided views, handling non-multiple strides,
    and returns the applied padding.

    Parameters:
        data : ndarray
            Input ND array.
        blck : tuple or array-like
            Block (patch) size for each dimension.
        strd : tuple or array-like
            Stride (step size) for each dimension.

    Returns:
        patches : ndarray
            Overlapping patches of shape (num_patches_dim1, ..., num_patches_dimN, *blck).
        pad_widths : list of tuples
            Padding applied to each dimension.
    """
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)

    # Compute the number of blocks in each dimension, rounding up
    nbl = np.ceil((sh - blck) / strd).astype(int) + 1  # Ensure we cover the full range

    # Compute required padding for each dimension
    pad_widths = np.maximum((nbl - 1) * strd + blck - sh, 0)

    # Apply padding if needed
    if np.any(pad_widths > 0):
        pad_widths = [(0, int(p)) for p in pad_widths]  # Pad only at the end
        data = np.pad(data, pad_widths, mode='constant')
    else:
        pad_widths = [(0, 0) for _ in range(len(sh))]  # No padding needed

    # Compute new strides
    strides = tuple(data.strides * strd) + tuple(data.strides)

    # Define output shape (patch grid dimensions + block size)
    new_shape = tuple(nbl) + tuple(blck)

    # Create the strided view
    patches = stride_tricks.as_strided(data, shape=new_shape, strides=strides)
    
    return patches, pad_widths

# def stuff_patches_3D(sh,patches,strd,blck):
#     out = np.zeros(sh, patches.dtype)
#     sh = np.asanyarray(sh)
#     blck = np.asanyarray(blck)
#     strd = np.asanyarray(strd)
#     nbl = (sh - blck) // strd + 1
#     strides = np.r_[out.strides * strd, out.strides]
#     dims = np.r_[nbl, blck]
#     data6 = stride_tricks.as_strided(out, strides=strides, shape=dims)
#     data6[...]=patches.reshape(data6.shape)
#     return out

def stuff_patches_3D(orig_sh, patches, blck, strd, pad_widths):
    """
    Recombines 3D overlapping patches into the original image/volume, handling padding and stride mismatches.

    Parameters:
        orig_sh : tuple of int
            Original shape of the volume before padding.
        patches : ndarray
            Overlapping patches of shape (num_patches_x, num_patches_y, num_patches_z, *blck).
        strd    : tuple of int
            Stride (step size) used to extract patches.
        blck    : tuple of int
            Patch size.
        pad_widths : list of tuples
            Padding applied to each dimension.

    Returns:
        out : ndarray
            Reconstructed volume, properly cropped to original dimensions.
    """
    padded_sh = tuple(s + sum(pad) for s, pad in zip(orig_sh, pad_widths))
    
    # Initialize padded output volume and weight map
    out = np.zeros(padded_sh, dtype=patches.dtype)
    weight = np.zeros(padded_sh, dtype=np.float32)

    strd = np.asanyarray(strd)
    blck = np.asanyarray(blck)

    # Compute number of patches in each dimension
    nbl = np.ceil((orig_sh - blck) / strd).astype(int) + 1

    # print(nbl)
    

    # Iterate over patch positions
    for i in range(nbl[0]):
        for j in range(nbl[1]):
            for k in range(nbl[2]):
                for l in range(nbl[3]):
                    # Compute slice indices
                    slices = (
                        slice(i * strd[0], min(i * strd[0] + blck[0], padded_sh[0])),
                        slice(j * strd[1], min(j * strd[1] + blck[1], padded_sh[1])),
                        slice(k * strd[2], min(k * strd[2] + blck[2], padded_sh[2])),
                        slice(l* strd[3], min(l * strd[3] + blck[3], padded_sh[3]))
                    )

                    # print(slices)

                    # Add patch values to the output image
                    # print(patches[i,j,k,l].shape)
                    out[slices] += patches[i, j, k,l]
                    weight[slices] += 1  # Accumulate weight for normalization

    # Normalize overlapping areas
    weight[weight == 0] = 1  # Avoid division by zero
    out /= weight

    # Crop to original shape (remove padding)
    crop_slices = tuple(slice(0, orig_sh[d]) for d in range(4))
    return out[crop_slices]

def proj_LLR(vol, blck, strd, threshold):
    if strd is None:
        strd=blck

    blck=np.array(list(vol.shape[:(vol.ndim - len(blck))])+list(blck))
    strd=np.array(list(vol.shape[:(vol.ndim - len(strd))])+list(strd))


    x_patches,padding = cutup(vol, blck, strd)
    patch_shape = x_patches.shape
    # print(patch_shape)
    x_patches = x_patches.reshape((np.prod(patch_shape[:len(blck)]), -1))

    a, s, bh = da.linalg.svd(da.asarray(x_patches))
    bh = np.array(bh)
    s = np.array(s)
    a = np.array(a)

    sig = pywt.threshold(s, threshold)

    print("Retained comp % {}".format(np.sum(sig > 0) / sig.shape[0] * 100))
    x_patches_lr = a @ np.diag(sig) @ bh
    u = stuff_patches_3D(vol.shape, x_patches_lr.reshape(patch_shape), blck, strd,padding)
    return u



def undersampling_operator_singular(volumes, trajectory, b1_all_slices=None, density_adj=True):
    """
    Computes A.H @ W @ A @ volumes where A is the Fourier + sampling operator 
    and W is the radial density adjustment.

    Parameters:
        volumes (ndarray): Image volumes of shape (nb_temporal_components, nb_slices, img_size_x, img_size_y)
        trajectory: Trajectory object with radial k-space coordinates
        b1_all_slices (ndarray): Coil sensitivity maps of shape (nb_slices, nb_channels, img_size_x, img_size_y)
        density_adj (bool): Whether to apply density compensation (default: True)

    Returns:
        images_series_rebuilt (ndarray): Reconstructed images of shape (nb_temporal_components, nb_slices, img_size_x, img_size_y)
    """
    
    L0, nb_slices, img_size_x, img_size_y = volumes.shape
    img_size = (img_size_x, img_size_y)
    volumes=volumes.astype("complex64")
    if b1_all_slices is None:
        b1_all_slices = np.ones((nb_slices, 1, img_size_x, img_size_y), dtype="complex64")

    nb_channels = b1_all_slices.shape[1]
    traj = trajectory.get_traj_for_reconstruction(1).astype("float32").reshape(-1, 2)
    npoint = trajectory.paramDict["npoint"]
    num_k_samples = traj.shape[0]

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        # density /=np.sum(density)

    # Initialize output image
    images_series_rebuilt = np.zeros((L0, nb_slices, img_size_x, img_size_y), dtype=np.complex64)

    for k in range(nb_channels):
        # Apply coil sensitivity maps
        curr_volumes = volumes * np.expand_dims(b1_all_slices[:, k], axis=0)

        # NUFFT Forward (Image → k-space)
        curr_kdata = finufft.nufft2d2(asca(traj[:, 0]), asca(traj[:, 1]), asca(curr_volumes.reshape(L0 * nb_slices, img_size_x, img_size_y)))
        curr_kdata = curr_kdata.reshape(L0, nb_slices, -1).astype("complex64")

        if density_adj:
            curr_kdata = curr_kdata.reshape(L0, -1, npoint)
              # Radial density compensation
            curr_kdata *= np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
            curr_kdata = curr_kdata.reshape(L0, nb_slices, traj.shape[0])

        # NUFFT Adjoint (k-space → Image)
        recon_images = finufft.nufft2d1(asca(traj[:, 0]), asca(traj[:, 1]), asca(curr_kdata.reshape(L0 * nb_slices, -1)), img_size)
        recon_images = recon_images.reshape((L0, nb_slices, img_size_x, img_size_y))

        # Apply conjugate coil sensitivities
        images_series_rebuilt += np.expand_dims(b1_all_slices[:, k].conj(), axis=0) * recon_images

    images_series_rebuilt /= num_k_samples  # Normalize
    return images_series_rebuilt





def fista_reconstruction(volumes, b1, radial_traj, dens_adj, niter, lambd, mu, 
                         prox_operator, **kwargs_prox):
    
    """
    Performs FISTA reconstruction with a given proximal operator for denoising.
    
    Parameters:
    - volumes: Input volume data.
    - b1: Sensitivity maps.
    - radial_traj: Radial trajectory for sampling.
    - dens_adj: Density compensation.
    - niter: Number of iterations.
    - lambd: Regularization parameter.
    - mu: Step size parameter.
    - prox_operator: Function handle for the proximal operator (e.g., wavelet or LLR).
    - **kwargs_prox: Additional arguments for the proximal operator.
    
    Returns:
    - Denoised and reconstructed volume.

    Example Usage:
    Using wavelet-based denoising
    result = fista_reconstruction(volumes, b1, radial_traj, dens_adj, niter, lambd, mu, 
                               prox_wavelet, wav_type='db4', wav_level=3, axes=(0,1))

    Using LLR-based denoising
    result = fista_reconstruction(volumes, b1, radial_traj, dens_adj, niter, lambd, mu, 
                               prox_LLR, blck=(8,8,8), strd=(4,4,4))
    """
    
    if niter == 0:
        return volumes
    
    # Initialize variables
    u = mu * volumes
    u0 = volumes
    y = u
    t = 1
    
    # Apply the proximal operator for initial denoising
    u = prox_operator(y, lambd * mu, **kwargs_prox)
    t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
    y = u
    t = t_next
    
    for i in tqdm(range(1, niter),desc="FISTA iteration"):
        u_prev = u
        
        # Compute undersampled volume
        volumesi = undersampling_operator_singular(y, radial_traj, b1, density_adj=dens_adj)
        
        # Gradient update
        grad = volumesi - u0 / mu
        y = y - mu * grad
        
        # Apply proximal operator
        u = prox_operator(y, lambd * mu, **kwargs_prox)
        
        # Update momentum
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u + (t - 1) / t_next * (u - u_prev)
        t = t_next
        
    return np.array(u).squeeze()

def prox_wavelet(volumes, threshold, wav_type="db4", wav_level=None, axes=(1,2,3)):
    """
    Applies wavelet-based thresholding as a proximal operator.
    
    Parameters:
    - volumes: Input volume data.
    - threshold: Threshold for wavelet shrinkage.
    - wav_type: Type of wavelet transform.
    - wav_level: Number of decomposition levels.
    - axes: Axes along which the wavelet transform is applied.
    
    Returns:
    - Denoised volume after wavelet reconstruction.
    """

    coefs = pywt.wavedecn(volumes, wav_type, level=wav_level, mode="periodization", axes=axes)
    u, slices = pywt.coeffs_to_array(coefs, axes=axes)
    u = pywt.threshold(u, threshold)

    return pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization", axes=axes)

def prox_LLR(volumes, threshold, blck, strd):
    """
    Applies locally low-rank (LLR) denoising as a proximal operator.
    
    Parameters:
    - volumes: Input volume data.
    - threshold: Threshold for low-rank approximation.
    - blck: Block size for LLR.
    - strd: Stride size for LLR.
    
    Returns:
    - Denoised volume after LLR processing.
    """

    return proj_LLR(volumes, blck, strd, threshold)



def add_temporal_basis(dico,L0=None):
    if "phi" not in dico.keys():
        print("Building temporal basis from dictionary")
        mrfdict=dico["mrfdict"]
        phi=build_phi(mrfdict)
        dico["phi"]=phi

    if (L0 is not None)and(("mrfdict_light_L0{}".format(L0) not in dico.keys())or("mrfdict_L0{}".format(L0) not in dico.keys())):
        phi=dico["phi"]
        print("Projecting dictionaries on subspace formed by first {} temporal components".format(L0))
        dico=compress_dictionary(dico,phi,L0)
    return dico
    


def compress_dictionary(dico,phi,L0):
    phi=phi[:L0]
    mrfdict=dico["mrfdict"]
    keys = mrfdict.keys
    array_water = mrfdict.values[:, :, 0]
    array_fat = mrfdict.values[:, :, 1]
    array_water_projected=array_water@phi.T.conj()
    array_fat_projected=array_fat@phi.T.conj()

    mrfdict_light=dico["mrfdict_light"]
    keys_light = mrfdict_light.keys
    array_water = mrfdict_light.values[:, :, 0]
    array_fat = mrfdict_light.values[:, :, 1]
    array_water_light_projected=array_water@phi.T.conj()
    array_fat_light_projected=array_fat@phi.T.conj()

    dico["mrfdict_light_L0{}".format(L0)]=(array_water_light_projected,array_fat_light_projected,keys_light)
    dico["mrfdict_L0{}".format(L0)]=(array_water_projected,array_fat_projected,keys)
    return dico

