import numpy as np
from scipy import ndimage
from scipy.ndimage import affine_transform
from mrftools.utils_mrf_simu import makevol,build_mask_from_volume,simulate_radial_undersampled_images,create_cuda_context,correct_mvt_kdata,simulate_radial_undersampled_images_multi,simulate_radial_undersampled_singular_images_multi,generate_kdata_multi,generate_kdata_singular_multi,undersampling_operator_singular,undersampling_operator,grad_J_TV,J_TV,generate_kdata_multi_new,undersampling_operator_new,undersampling_operator_singular_new
from mrftools.utils_reco import *
from mrftools.Transformers import PCAComplex
# from mutools.optim.dictsearch import dictsearch
from tqdm import tqdm
# from mrfsim import makevol
from mrftools.image_series import MapFromDict
from datetime import datetime
from skimage.restoration import denoise_tv_chambolle
from copy import copy
import os
#import cupy as cp
try:
    #from pycuda.autoinit import _finish_up
    import cupy as cp
except:
    print("Could not import cupy")
    pass

try :
    from SPIJN import *
except:
    print("Could not import SPIJN")
    pass

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
try:
    from numba import cuda
except:
    pass
import gc
import pickle

class GaussianWeighting(object):
    def __init__(self, sig=np.pi/2):
        self.sig=sig

    def apply(self,traj):
        return np.exp(-np.linalg.norm(traj,axis=-1)**2/(2*self.sig**2))

def match_signals_v2(all_signals,keys,pca_water,pca_fat,array_water_unique,array_fat_unique,transformed_array_water_unique,transformed_array_fat_unique,var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates,verbose,niter,split,useGPU_dictsearch,mask,tv_denoising_weight,log_phase=False,return_matched_signals=False,n_clusters_dico=None,pruning=None):

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

    if n_clusters_dico is not None:
        niter=0
        return_matched_signals=False
        log_phase=False

    if niter > 0 or return_matched_signals:
        phase_optim = []
        J_optim = []

    elif log_phase:
        phase_optim = []

    for j in tqdm(range(num_group)):
        j_signal = j * split
        j_signal_next = np.minimum((j + 1) * split, nb_signals)

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
                print(array_water_unique.shape)
                print(all_signals.shape)
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

        # current_sig_ws = current_sig_ws_for_phase.real
        # current_sig_fs = current_sig_fs_for_phase.real

        if verbose:
            end = datetime.now()
            print(end - start)

        if not (useGPU_dictsearch):
            # if adj_phase:
            if verbose:
                print("Adjusting Phase")
                print("Calculating alpha optim and flooring")

                ### Testing direct phase solving
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

            # current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
            #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

            #del alpha1
            #del alpha2

            if verbose:
                start = datetime.now()



            #current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

            apha_more_0=(current_alpha_all_unique>=0)
            alpha_less_1=(current_alpha_all_unique<=1)
            alpha_out_bounds=(1*(apha_more_0))*(1*(alpha_less_1))==0

            J_0=np.abs(current_sig_ws_for_phase)/np.sqrt(var_w)

            J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype("float")


            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
            start = datetime.now()

            #current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
            #current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

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

            if n_clusters_dico is None:
                idx_max_all_current = np.argmax(J_all, axis=0)
                current_alpha_all_unique_optim=current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]
                idx_max_all_unique[j_signal:j_signal_next]=idx_max_all_current
                alpha_optim[j_signal:j_signal_next]=current_alpha_all_unique_optim
            else:
                idx_max_all_current = np.argsort(J_all, axis=0)[-int(pruning * n_clusters_dico):]


            if niter>0 or log_phase or return_matched_signals:
                d = (
                            1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current, np.arange(J_all.shape[1])] + current_alpha_all_unique_optim * current_sig_fs_for_phase[idx_max_all_current, np.arange(J_all.shape[1])]
                phase_adj = -np.arctan(d.imag / d.real)
                cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real

                del d

                phase_adj = (phase_adj) * (
                        1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                    1 * (cond) > 0)

                del cond

            if niter>0 or return_matched_signals:
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

            # print(current_alpha_all_unique.shape)
            # print(J_1.shape)
            # print(J_0.shape)
            # print(alpha_out_bounds.shape)

            current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)
            # phase_adj = np.angle((
            #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)



            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
                start = datetime.now()


            # del phase_adj
            #del current_sig_ws_for_phase
            #del current_sig_fs_for_phase

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

            if n_clusters_dico is None:
                idx_max_all_current = cp.argmax(J_all, axis=0)
                current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]


                #current_alpha_all_unique_optim = current_alpha_all_unique_optim.get()
                idx_max_all_unique[j_signal:j_signal_next] = idx_max_all_current
                alpha_optim[j_signal:j_signal_next]=current_alpha_all_unique_optim

            else:
                idx_max_all_current = cp.argsort(J_all, axis=0)[-int(pruning * n_clusters_dico):]



            

            if niter>0 or log_phase or return_matched_signals:
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


            if niter > 0 or return_matched_signals:
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

        #idx_max_all_current = np.argmax(J_all, axis=0)
        # check_max_correl=np.max(J_all,axis=0)

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Filling the lists with results for this loop")
            start = datetime.now()




        if n_clusters_dico is None:
            pass
            #if not(useGPU_dictsearch):
            #    idx_max_all_unique.extend(idx_max_all_current)
            #    alpha_optim.extend(current_alpha_all_unique_optim)
        else:
            if len(idx_max_all_unique)==0:
                idx_max_all_unique=idx_max_all_current
            else:
                idx_max_all_unique=np.append(idx_max_all_unique,idx_max_all_current,axis=1)

        del current_alpha_all_unique_optim
        del idx_max_all_current


        if (niter > 0) or return_matched_signals:
            phase_optim.extend(phase_adj)
            J_optim.extend(J_all_optim)

        elif log_phase:
            phase_optim.extend(phase_adj)

        #if not (return_matched_signals):
            #del phase_adj


        if verbose:
            end = datetime.now()
            print(end - start)


    if useGPU_dictsearch:
        idx_max_all_unique=idx_max_all_unique.get()
        alpha_optim=alpha_optim.get()
    # idx_max_all_unique = np.argmax(J_all, axis=0)
    #del J_all
    #del current_alpha_all_unique

    if (niter > 0) or return_matched_signals:
        phase_optim = np.array(phase_optim)
        J_optim = np.array(J_optim)
    elif log_phase:
        phase_optim = np.array(phase_optim)



    # del sig_ws_all_unique
    # del sig_fs_all_unique
    if n_clusters_dico is not None:
        return idx_max_all_unique

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

    if tv_denoising_weight is not None:
        for i,k in enumerate(map_rebuilt.keys()):
            map_rebuilt[k]=denoise_tv_chambolle(makevol(map_rebuilt[k],mask>0),weight=tv_denoising_weight)[mask>0]
            if k!="ff":#Projection back on dictionary parameters
                print("Projection back to dictionary values for {} after denoising".format(k))
                curr_values = np.unique(np.array(keys)[:,i])
                map_rebuilt[k]=curr_values[np.argmin(np.abs(map_rebuilt[k].reshape(-1, 1) - curr_values), axis=-1)]




    if not(log_phase):
        if return_matched_signals:
            print(phase_optim.shape)
            print(J_optim.shape)

            matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique, :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_unique, :].T * np.array(
                    alpha_optim).reshape(1, -1)
            print(matched_signals.shape)
            matched_signals/=np.linalg.norm(matched_signals,axis=0)
            matched_signals *= J_optim*np.exp(1j*phase_optim)
            return map_rebuilt,None,None,matched_signals.squeeze()
        else:
            return map_rebuilt, None, None
    else:
        return map_rebuilt, None, phase_optim



def match_signals_v2_clustered_on_dico(all_signals_current,keys,pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique,var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique,useGPU_dictsearch,unique_keys,d_T1,d_fT1,d_B1,d_DF,labels,split,high_ff=False,return_cost=False):
    #nb_signals_low_ff = len(ind_low_ff)
    #nb_signals_high_ff = len(ind_high_ff)
    nb_clusters = unique_keys.shape[-1]

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

            # if j_signal==nb_signals:
            #    break
            keys_T1 = (keys[:, 0] < unique_keys[:, cl][0] + d_T1) & ((keys[:, 0] > unique_keys[:, cl][0] - d_T1))
            keys_fT1 = (keys[:, 1] < unique_keys[:, cl][1] + d_fT1) & ((keys[:, 1] > unique_keys[:, cl][1] - d_fT1))
            keys_B1 = (keys[:, 2] < unique_keys[:, cl][2] + d_B1) & ((keys[:, 2] > unique_keys[:, cl][2] - d_B1))
            keys_DF = (keys[:, 3] < unique_keys[:, cl][3] + d_DF) & ((keys[:, 3] > unique_keys[:, cl][3] - d_DF))
            retained_signals = np.argwhere(keys_T1 & keys_fT1 & keys_B1 & keys_DF).flatten()

            #print(retained_signals.shape)


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
                # current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)
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

                # current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                # current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real
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

            # if j_signal==nb_signals:
            #    break
            keys_T1 = (keys[:, 0] < unique_keys[:, cl][0] + d_T1) & ((keys[:, 0] > unique_keys[:, cl][0] - d_T1))
            keys_fT1 = (keys[:, 1] < unique_keys[:, cl][1] + d_fT1) & ((keys[:, 1] > unique_keys[:, cl][1] - d_fT1))
            keys_B1 = (keys[:, 2] < unique_keys[:, cl][2] + d_B1) & ((keys[:, 2] > unique_keys[:, cl][2] - d_B1))
            keys_DF = (keys[:, 3] < unique_keys[:, cl][3] + d_DF) & ((keys[:, 3] > unique_keys[:, cl][3] - d_DF))
            retained_signals = cp.argwhere(keys_T1 & keys_fT1 & keys_B1 & keys_DF).flatten()

            #print(retained_signals.shape)


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

                # current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                # current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real
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
                #idx_max_all_current_sig=idx_max_all_current_sig.get()
                #current_alpha_all_unique_optim = current_alpha_all_unique_optim.get()
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

def match_signals_v2_high_ff(all_signals_high_ff,keys,pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique,var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique,useGPU_dictsearch,split):
    #nb_signals_low_ff = len(ind_low_ff)
    #nb_signals_high_ff = len(ind_high_ff)
    #nb_clusters = unique_keys.shape[-1]

    nb_signals_high_ff=all_signals_high_ff.shape[-1]
    num_group = int(nb_signals_high_ff / split) + 1

    idx_max_all_unique_high_ff = []
    alpha_optim_high_ff = []

    if not (useGPU_dictsearch):
        for j in tqdm(range(num_group)):
            j_signal = j * split
            j_signal_next = np.minimum((j + 1) * split, nb_signals_high_ff)

            var_w = var_w_total
            var_f = var_f_total
            sig_wf = sig_wf_total

            transformed_all_signals_water = np.transpose(
                pca_water.transform(np.transpose(all_signals_high_ff[:, j_signal:j_signal_next])))
            transformed_all_signals_fat = np.transpose(
                pca_fat.transform(np.transpose(all_signals_high_ff[:, j_signal:j_signal_next])))
            sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                          transformed_all_signals_water.conj())
            sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                          transformed_all_signals_fat.conj())
            current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
            current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]
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
            # current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)
            apha_more_0 = (current_alpha_all_unique >= 0)
            alpha_less_1 = (current_alpha_all_unique <= 1)
            alpha_out_bounds = (1 * (apha_more_0)) * (1 * (alpha_less_1)) == 0
            # J_0=np.abs(current_sig_ws_for_phase)/np.sqrt(var_w)
            J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)
            # current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype("float")
            current_alpha_all_unique[alpha_out_bounds] = 1.0

            # current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
            # current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real
            J_all = np.abs((
                                   1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase) / np.sqrt(
                (
                        1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                        1 - current_alpha_all_unique) * sig_wf)

            # all_J = np.stack([J_all, J_0, J_1], axis=0)
            all_J = np.stack([J_all, J_1], axis=0)
            ind_max_J = np.argmax(all_J, axis=0)
            del all_J
            # J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_0 + (ind_max_J == 2) * J_1
            J_all = (ind_max_J == 0) * J_all + (ind_max_J == 1) * J_1
            del J_1
            # del J_1
            # current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
            #            ind_max_J == 2) * 1

            current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 1

            idx_max_all_current_sig = np.argmax(J_all, axis=0)
            current_alpha_all_unique_optim = current_alpha_all_unique[
                idx_max_all_current_sig, np.arange(J_all.shape[1])]
            idx_max_all_unique_high_ff.extend(idx_max_all_current_sig)
            alpha_optim_high_ff.extend(current_alpha_all_unique_optim)
    else:
        raise ValueError("Not implemented yet")

    idx_max_all_unique_high_ff = np.array(idx_max_all_unique_high_ff)
    alpha_optim_high_ff = np.array(alpha_optim_high_ff)

    return idx_max_all_unique_high_ff,alpha_optim_high_ff

def match_signals_v2_on_clusters(all_signals,keys,threshold_pca,array_water,array_fat,var_w_total,var_f_total,sig_wf_total,verbose,niter,useGPU_dictsearch,idx_clusters,model_dico,model_signals,log_phase=False,return_matched_signals=False):

    nb_signals = all_signals.shape[1]

    n_clusters_signals=model_signals.n_clusters

    idx_max_all_unique = np.zeros(nb_signals)
    alpha_optim = np.zeros(nb_signals)


    if niter > 0 or return_matched_signals:
        phase_optim = np.zeros(nb_signals)
        J_optim = np.zeros(nb_signals)

    elif log_phase:
        phase_optim = np.zeros(nb_signals)

    if useGPU_dictsearch:
        var_w_total=cp.asarray(var_w_total)
        var_f_total = cp.asarray(var_f_total)
        sig_wf_total = cp.asarray(sig_wf_total)

    for j in tqdm(range(n_clusters_signals)):
        indices = np.argwhere(model_signals.labels_ == j)
        # if j_signal==nb_signals:
        #    break

        retained_clusters = idx_clusters[:, indices.flatten()].flatten()
        retained_signals = np.where(np.in1d(model_dico.labels_, retained_clusters))[0]
        print(retained_signals.shape)

        array_water_current = array_water[retained_signals]
        array_fat_current = array_fat[retained_signals]

        var_w = var_w_total[retained_signals]
        var_f = var_f_total[retained_signals]
        sig_wf = sig_wf_total[retained_signals]

        array_water_unique, index_water_unique = np.unique(array_water_current, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat_current, axis=0, return_inverse=True)
        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]
        pca_water = PCAComplex(n_components_=threshold_pca)
        pca_fat = PCAComplex(n_components_=threshold_pca)
        pca_water.fit(array_water_unique)
        pca_fat.fit(array_fat_unique)
        print("Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
        print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

        transformed_array_water_unique = pca_water.transform(array_water_unique)
        transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        if not (useGPU_dictsearch):
            transformed_all_signals_water = np.transpose(
                pca_water.transform(np.transpose(all_signals[:, indices.flatten()])))
            transformed_all_signals_fat = np.transpose(
                pca_fat.transform(np.transpose(all_signals[:, indices.flatten()])))
            sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                          transformed_all_signals_water.conj())
            sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                          transformed_all_signals_fat.conj())



        else:
            transformed_all_signals_water = cp.transpose(
                    pca_water.transform(cp.transpose(cp.asarray(all_signals[:, indices.flatten()])))).get()

            transformed_all_signals_fat = cp.transpose(
                    pca_fat.transform(cp.transpose(cp.asarray(all_signals[:, indices.flatten()])))).get()

            sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                               cp.asarray(transformed_all_signals_water).conj())).get()
            sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                               cp.asarray(transformed_all_signals_fat).conj())).get()

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Extracting all sig_ws and sig_fs")
            start = datetime.now()

        current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
        current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

        # current_sig_ws = current_sig_ws_for_phase.real
        # current_sig_fs = current_sig_fs_for_phase.real

        if verbose:
            end = datetime.now()
            print(end - start)

        if not (useGPU_dictsearch):
            # if adj_phase:
            if verbose:
                print("Adjusting Phase")
                print("Calculating alpha optim and flooring")

                ### Testing direct phase solving
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

            # current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
            #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

            #del alpha1
            #del alpha2

            if verbose:
                start = datetime.now()



            #current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

            apha_more_0=(current_alpha_all_unique>=0)
            alpha_less_1=(current_alpha_all_unique<=1)
            alpha_out_bounds=(1*(apha_more_0))*(1*(alpha_less_1))==0

            J_0=np.abs(current_sig_ws_for_phase)/np.sqrt(var_w)

            J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype("float")


            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
            start = datetime.now()

            #current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
            #current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

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

            idx_max_all_current_sig = np.argmax(J_all, axis=0)
            current_alpha_all_unique_optim=current_alpha_all_unique[idx_max_all_current_sig, np.arange(J_all.shape[1])]



            if niter>0 or log_phase or return_matched_signals:
                d = (
                            1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])] + current_alpha_all_unique_optim * current_sig_fs_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                phase_adj = -np.arctan(d.imag / d.real)
                cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real

                del d

                phase_adj = (phase_adj) * (
                        1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                    1 * (cond) > 0)

                del cond

            if niter>0 or return_matched_signals:
                J_all_optim=J_all[idx_max_all_current_sig, np.arange(J_all.shape[1])]


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

            # print(current_alpha_all_unique.shape)
            # print(J_1.shape)
            # print(J_0.shape)
            # print(alpha_out_bounds.shape)

            current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)
            # phase_adj = np.angle((
            #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)



            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
                start = datetime.now()


            # del phase_adj
            #del current_sig_ws_for_phase
            #del current_sig_fs_for_phase

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

            idx_max_all_current_sig = cp.argmax(J_all, axis=0)
            current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current_sig, np.arange(J_all.shape[1])]
            current_alpha_all_unique_optim = current_alpha_all_unique_optim.get()




            del current_sig_ws_for_phase
            del current_sig_fs_for_phase

            if niter > 0 or return_matched_signals:
                J_all_optim = J_all[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                J_all_optim=J_all_optim.get()


            idx_max_all_current_sig = idx_max_all_current_sig.get()

            del J_all
            del current_alpha_all_unique


            if verbose:
                end = datetime.now()
                print(end - start)

        if verbose:
            print("Extracting index of pattern with max correl")
            start = datetime.now()

        #idx_max_all_current = np.argmax(J_all, axis=0)
        # check_max_correl=np.max(J_all,axis=0)

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Filling the lists with results for this loop")
            start = datetime.now()



        idx_max_all_unique[indices.flatten()]=idx_max_all_current_sig
        alpha_optim[indices.flatten()]=current_alpha_all_unique_optim

        if (niter > 0) or return_matched_signals:
            phase_optim[indices.flatten()]=phase_adj
            J_optim[indices.flatten()]=(J_all_optim)

        elif log_phase:
            phase_optim[indices.flatten()]=phase_adj

        #if not (return_matched_signals):
            #del phase_adj


        if verbose:
            end = datetime.now()
            print(end - start)

    # idx_max_all_unique = np.argmax(J_all, axis=0)
    #del J_all
    #del current_alpha_all_unique

    idx_max_all_unique=idx_max_all_unique.astype(int)
    params_all_unique = np.array(
        [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique)])


    map_rebuilt = {
        "wT1": params_all_unique[:, 0],
        "fT1": params_all_unique[:, 1],
        "attB1": params_all_unique[:, 2],
        "df": params_all_unique[:, 3],
        "ff": params_all_unique[:, 4]

    }




    if not(log_phase):
        if return_matched_signals:
            print(phase_optim.shape)
            print(J_optim.shape)

            matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique, :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_unique, :].T * np.array(
                    alpha_optim).reshape(1, -1)
            print(matched_signals.shape)
            matched_signals/=np.linalg.norm(matched_signals,axis=0)
            matched_signals *= J_optim*np.exp(1j*phase_optim)
            return map_rebuilt,None,None,matched_signals.squeeze()
        else:
            return map_rebuilt, None, None
    else:
        return map_rebuilt, None, phase_optim


def match_signals_v2_on_clusters(all_signals,keys,threshold_pca,array_water,array_fat,var_w_total,var_f_total,sig_wf_total,verbose,niter,useGPU_dictsearch,idx_clusters,model_dico,model_signals,log_phase=False,return_matched_signals=False):

    nb_signals = all_signals.shape[1]

    n_clusters_signals=model_signals.n_clusters

    idx_max_all_unique = np.zeros(nb_signals)
    alpha_optim = np.zeros(nb_signals)


    if niter > 0 or return_matched_signals:
        phase_optim = np.zeros(nb_signals)
        J_optim = np.zeros(nb_signals)

    elif log_phase:
        phase_optim = np.zeros(nb_signals)

    if useGPU_dictsearch:
        var_w_total=cp.asarray(var_w_total)
        var_f_total = cp.asarray(var_f_total)
        sig_wf_total = cp.asarray(sig_wf_total)

    for j in tqdm(range(n_clusters_signals)):
        indices = np.argwhere(model_signals.labels_ == j)
        # if j_signal==nb_signals:
        #    break

        retained_clusters = idx_clusters[:, indices.flatten()].flatten()
        retained_signals = np.where(np.in1d(model_dico.labels_, retained_clusters))[0]
        print(retained_signals.shape)

        array_water_current = array_water[retained_signals]
        array_fat_current = array_fat[retained_signals]

        var_w = var_w_total[retained_signals]
        var_f = var_f_total[retained_signals]
        sig_wf = sig_wf_total[retained_signals]

        array_water_unique, index_water_unique = np.unique(array_water_current, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat_current, axis=0, return_inverse=True)
        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]
        pca_water = PCAComplex(n_components_=threshold_pca)
        pca_fat = PCAComplex(n_components_=threshold_pca)
        pca_water.fit(array_water_unique)
        pca_fat.fit(array_fat_unique)
        print("Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
        print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

        transformed_array_water_unique = pca_water.transform(array_water_unique)
        transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        if not (useGPU_dictsearch):
            transformed_all_signals_water = np.transpose(
                pca_water.transform(np.transpose(all_signals[:, indices.flatten()])))
            transformed_all_signals_fat = np.transpose(
                pca_fat.transform(np.transpose(all_signals[:, indices.flatten()])))
            sig_ws_all_unique = np.matmul(transformed_array_water_unique,
                                          transformed_all_signals_water.conj())
            sig_fs_all_unique = np.matmul(transformed_array_fat_unique,
                                          transformed_all_signals_fat.conj())



        else:
            transformed_all_signals_water = cp.transpose(
                    pca_water.transform(cp.transpose(cp.asarray(all_signals[:, indices.flatten()])))).get()

            transformed_all_signals_fat = cp.transpose(
                    pca_fat.transform(cp.transpose(cp.asarray(all_signals[:, indices.flatten()])))).get()

            sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                               cp.asarray(transformed_all_signals_water).conj())).get()
            sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                               cp.asarray(transformed_all_signals_fat).conj())).get()

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Extracting all sig_ws and sig_fs")
            start = datetime.now()

        current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
        current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

        # current_sig_ws = current_sig_ws_for_phase.real
        # current_sig_fs = current_sig_fs_for_phase.real

        if verbose:
            end = datetime.now()
            print(end - start)

        if not (useGPU_dictsearch):
            # if adj_phase:
            if verbose:
                print("Adjusting Phase")
                print("Calculating alpha optim and flooring")

                ### Testing direct phase solving
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

            # current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
            #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

            #del alpha1
            #del alpha2

            if verbose:
                start = datetime.now()



            #current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

            apha_more_0=(current_alpha_all_unique>=0)
            alpha_less_1=(current_alpha_all_unique<=1)
            alpha_out_bounds=(1*(apha_more_0))*(1*(alpha_less_1))==0

            J_0=np.abs(current_sig_ws_for_phase)/np.sqrt(var_w)

            J_1 = np.abs(current_sig_fs_for_phase) / np.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype("float")


            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
            start = datetime.now()

            #current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
            #current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

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

            idx_max_all_current_sig = np.argmax(J_all, axis=0)
            current_alpha_all_unique_optim=current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]



            if niter>0 or log_phase or return_matched_signals:
                d = (
                            1 - current_alpha_all_unique_optim) * current_sig_ws_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])] + current_alpha_all_unique_optim * current_sig_fs_for_phase[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                phase_adj = -np.arctan(d.imag / d.real)
                cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real

                del d

                phase_adj = (phase_adj) * (
                        1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                    1 * (cond) > 0)

                del cond

            if niter>0 or return_matched_signals:
                J_all_optim=J_all[idx_max_all_current_sig, np.arange(J_all.shape[1])]


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

            # print(current_alpha_all_unique.shape)
            # print(J_1.shape)
            # print(J_0.shape)
            # print(alpha_out_bounds.shape)

            current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)
            # phase_adj = np.angle((
            #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)



            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
                start = datetime.now()


            # del phase_adj
            #del current_sig_ws_for_phase
            #del current_sig_fs_for_phase

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

            idx_max_all_current_sig = cp.argmax(J_all, axis=0)
            current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current_sig, np.arange(J_all.shape[1])]
            current_alpha_all_unique_optim = current_alpha_all_unique_optim.get()




            del current_sig_ws_for_phase
            del current_sig_fs_for_phase

            if niter > 0 or return_matched_signals:
                J_all_optim = J_all[idx_max_all_current_sig, np.arange(J_all.shape[1])]
                J_all_optim=J_all_optim.get()


            idx_max_all_current_sig = idx_max_all_current_sig.get()

            del J_all
            del current_alpha_all_unique


            if verbose:
                end = datetime.now()
                print(end - start)

        if verbose:
            print("Extracting index of pattern with max correl")
            start = datetime.now()

        #idx_max_all_current = np.argmax(J_all, axis=0)
        # check_max_correl=np.max(J_all,axis=0)

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Filling the lists with results for this loop")
            start = datetime.now()



        idx_max_all_unique[indices.flatten()]=idx_max_all_current_sig
        alpha_optim[indices.flatten()]=current_alpha_all_unique_optim

        if (niter > 0) or return_matched_signals:
            phase_optim[indices.flatten()]=phase_adj
            J_optim[indices.flatten()]=(J_all_optim)

        elif log_phase:
            phase_optim[indices.flatten()]=phase_adj

        #if not (return_matched_signals):
            #del phase_adj


        if verbose:
            end = datetime.now()
            print(end - start)

    # idx_max_all_unique = np.argmax(J_all, axis=0)
    #del J_all
    #del current_alpha_all_unique




    params_all_unique = np.array(
        [keys[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique.astype(int))])


    map_rebuilt = {
        "wT1": params_all_unique[:, 0],
        "fT1": params_all_unique[:, 1],
        "attB1": params_all_unique[:, 2],
        "df": params_all_unique[:, 3],
        "ff": params_all_unique[:, 4]

    }




    if not(log_phase):
        if return_matched_signals:
            print(phase_optim.shape)
            print(J_optim.shape)

            matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique, :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_unique, :].T * np.array(
                    alpha_optim).reshape(1, -1)
            print(matched_signals.shape)
            matched_signals/=np.linalg.norm(matched_signals,axis=0)
            matched_signals *= J_optim*np.exp(1j*phase_optim)
            return map_rebuilt,None,None,matched_signals.squeeze()
        else:
            return map_rebuilt, None, None
    else:
        return map_rebuilt, None, phase_optim


def match_signals(all_signals,keys,pca_water,pca_fat,array_water_unique,array_fat_unique,transformed_array_water_unique,transformed_array_fat_unique,var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates,verbose,niter,split,useGPU_dictsearch,mask,tv_denoising_weight,log_phase=False,return_matched_signals=False):

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

    idx_max_all_unique = []
    alpha_optim = []

    if niter > 0:
        phase_optim = []
        J_optim = []

    elif log_phase:
        phase_optim = []

    for j in tqdm(range(num_group)):
        j_signal = j * split
        j_signal_next = np.minimum((j + 1) * split, nb_signals)

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

        current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
        current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

        # current_sig_ws = current_sig_ws_for_phase.real
        # current_sig_fs = current_sig_fs_for_phase.real

        if verbose:
            end = datetime.now()
            print(end - start)

        if not (useGPU_dictsearch):
            # if adj_phase:
            if verbose:
                print("Adjusting Phase")
                print("Calculating alpha optim and flooring")

                ### Testing direct phase solving
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

            # current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
            #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

            #del alpha1
            #del alpha2

            if verbose:
                start = datetime.now()



            #current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

            apha_more_0=(current_alpha_all_unique>=0)
            alpha_less_1=(current_alpha_all_unique<=1)
            alpha_out_bounds=(1*(apha_more_0))*(1*(alpha_less_1))==0


            # phase_adj=np.angle((1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)



            d_oobounds_0 = current_sig_ws_for_phase[:]
            phase_adj_0 = -np.arctan(d_oobounds_0.imag / d_oobounds_0.real)
            cond = np.sin(phase_adj_0) * d_oobounds_0.imag - np.cos(phase_adj_0) * d_oobounds_0.real
            del d_oobounds_0

            phase_adj_0 = (phase_adj_0) * (
                    1 * (cond) <= 0) + (phase_adj_0 + np.pi) * (
                                1 * (cond) > 0)

            del cond

            current_sig_ws_0 = (current_sig_ws_for_phase[:] * np.exp(1j * phase_adj_0)).real
            J_0=current_sig_ws_0/np.sqrt(var_w)

            d_oobounds_1 = current_sig_fs_for_phase[:]
            phase_adj_1 = -np.arctan(d_oobounds_1.imag / d_oobounds_1.real)
            cond = np.sin(phase_adj_1) * d_oobounds_1.imag - np.cos(phase_adj_1) * d_oobounds_1.real
            del d_oobounds_1

            phase_adj_1 = (phase_adj_1) * (
                    1 * (cond) <= 0) + (phase_adj_1 + np.pi) * (
                                  1 * (cond) > 0)

            del cond

            current_sig_fs_1 = (current_sig_fs_for_phase[:] * np.exp(1j * phase_adj_1)).real
            J_1 = current_sig_fs_1 / np.sqrt(var_f)

            current_alpha_all_unique[alpha_out_bounds]=np.argmax(np.concatenate([J_0[alpha_out_bounds, None], J_1[alpha_out_bounds, None]], axis=-1), axis=-1).astype("float")


            d = (
                        1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase

            phase_adj = -np.arctan(d.imag / d.real)
            cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
            del d

            phase_adj = (phase_adj) * (
                    1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                1 * (cond) > 0)

            del cond

            ##############################################################################################################################
            # def J_alpha_pixel(alpha,phi, i, j):
            #
            #     current_sig_ws = (current_sig_ws_for_phase[i,j] * np.exp(1j * phi)).real
            #     current_sig_fs = (current_sig_fs_for_phase[i,j] * np.exp(1j * phi)).real
            #     return ((
            #              1 - alpha) * current_sig_ws + alpha * current_sig_fs) / np.sqrt(
            #         (
            #                 1 - alpha) ** 2 * var_w[i] + alpha ** 2 * var_f[i] + 2 * alpha * (
            #                 1 - alpha) * sig_wf[i])
            #
            # phi = np.arange(-np.pi,np.pi,np.pi/20)
            # alpha = np.arange(0.,1.01,0.01)
            # alphav_np, phiv_np = np.meshgrid(alpha, phi, sparse=False, indexing='ij')
            #
            # i_=0
            # j_=0
            #
            # s,t=current_sig_ws_for_phase.shape
            # n,m = alphav_np.shape
            # result_np=np.zeros(alphav_np.shape)
            #
            #
            # i_,j_=np.unravel_index(np.random.choice(np.arange(s*t)),(s,t))
            #
            #
            # for p in tqdm(range(n)):
            #     for q in range(m):
            #         result_np[p,q]=J_alpha_pixel(alphav_np[p,q],phiv_np[p,q],i_,j_)
            #
            #
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # ax.plot_surface(alphav_np, phiv_np, result_np,alpha=0.5)
            #
            #
            #
            # index_min_p,index_min_q = np.unravel_index(np.argmax(result_np), result_np.shape)
            # alpha_min = alphav_np[index_min_p,index_min_q]
            # phi_min = phiv_np[index_min_p, index_min_q]
            # result_min = result_np[index_min_p, index_min_q]
            #
            # # alpha_ = (1 * (alpha1[i, j] >= 0) & (alpha1[i, j] <= 1)) * alpha1[i, j] + (
            # #             1 - (1 * (alpha1[i, j] >= 0) & (alpha1[i, j] <= 1))) * alpha2[i, j]
            #
            # print("Max alpha on surface : {}".format(np.round(alpha_min,2)))
            # print("Alpha 1 : {}".format(np.round(alpha1[i_,j_],2)))
            # print("Alpha 2 : {}".format(np.round(alpha2[i_,j_],2)))
            # print("Alpha calc : {}".format(np.round(current_alpha_all_unique[i_,j_], 2)))
            #
            # # phi_calc = -np.angle((
            # #                               1 - alpha_) * current_sig_ws_for_phase[i, j] + alpha_ * current_sig_fs_for_phase[i, j])
            # #
            # # d = (1 - alpha_) * current_sig_ws_for_phase[i, j] + alpha_ * \
            # #      current_sig_fs_for_phase[i, j]
            # # phi_form = -np.arctan(d.imag / d.real)
            # # phi_form = (phi_form) * (
            # #             1 * (np.sin(phi_form) * d.imag - np.cos(phi_form) * d.real) <= 0) + (
            # #                  np.mod(phi_form + np.pi, 2 * np.pi)) * (
            # #                          1 * (np.sin(phi_form) * d.imag - np.cos(phi_form) * d.real) > 0)
            #
            # # phi_calc1 = -np.angle((
            # #                              1 - alpha1[i,j]) * current_sig_ws_for_phase[i,j] + alpha1[i,j] * current_sig_fs_for_phase[i,j])
            # # phi_calc2 = -np.angle((
            # #                              1 - alpha2[i, j]) * current_sig_ws_for_phase[i, j] +  alpha2[i, j] *
            # #                      current_sig_fs_for_phase[i, j])
            # #
            # # d1 = (1 - alpha1[i,j]) * current_sig_ws_for_phase[i,j] + alpha1[i,j] * current_sig_fs_for_phase[i,j]
            # # phi_form_1 = -np.arctan(d1.imag/d1.real)
            # # d2 = (1 - alpha2[i, j]) * current_sig_ws_for_phase[i, j] + alpha2[i, j] * current_sig_fs_for_phase[
            # #     i, j]
            # # phi_form_2 = -np.arctan(d2.imag/d2.real)
            # #
            # # phi_form_1 = (phi_form_1)*(1*(np.sin(phi_form_1)*d1.imag-np.cos(phi_form_1)*d1.real)<=0)+(np.mod(phi_form_1+np.pi,2*np.pi))*(1*(np.sin(phi_form_1)*d1.imag-np.cos(phi_form_1)*d1.real)>0)
            # # phi_form_2 = (phi_form_2) * (
            # #             1 * (np.sin(phi_form_2) * d2.imag - np.cos(phi_form_2) * d2.real) <= 0) + (
            # #                  np.mod(phi_form_2 + np.pi, 2 * np.pi)) * (
            # #                          1 * (np.sin(phi_form_2) * d2.imag - np.cos(phi_form_2) * d2.real) > 0)
            #
            # print("Max phi on surface : {}".format(np.round(phi_min, 2)))
            # # print("Phi Ideal 1 : {}".format(np.round(phi_calc1, 2)))
            # # print("Phi Ideal 2 : {}".format(np.round(phi_calc2, 2)))
            # # print("Phi Formula 1 : {}".format(np.round(phi_form_1, 2)))
            # # print("Phi Formula 2 : {}".format(np.round(phi_form_2, 2)))
            # print("Phi optim: {}".format(np.round(phase_adj[i_,j_], 2)))
            #
            # print("Max correl on surface {}".format(np.round(result_min,2)))
            # print("Retrieved correl on surface {}".format(np.round( J_alpha_pixel(current_alpha_all_unique[i_,j_], phase_adj[i_,j_], i_, j_)[0],2)))
            #
            # ax.plot(alpha_min,phi_min,result_min,marker="x")
            # ax.plot(current_alpha_all_unique[i_,j_], phase_adj[i_,j_], J_alpha_pixel(current_alpha_all_unique[i_,j_], phase_adj[i_,j_], i_, j_)[0], marker="o")
            # ax.set_title("Signal {},{}".format(i_,j_))
            # # ax.plot(alpha1[i,j], phi_form_1, J_alpha_pixel(alpha1[i,j],phi_form_1,i,j)[0], marker="o")
            # # ax.plot(alpha2[i,j], phi_form_2,
            # #         J_alpha_pixel(alpha2[i, j], phi_form_2, i, j)[0], marker="o")
            #################################################################################################################################""""

            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
            start = datetime.now()

            current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
            current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

            J_all = ((
                             1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
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

            idx_max_all_current = np.argmax(J_all, axis=0)

            current_alpha_all_unique = (ind_max_J == 0) * current_alpha_all_unique + (ind_max_J == 1) * 0 + (
                        ind_max_J == 2) * 1
            phase_adj = (ind_max_J == 0) * phase_adj + (ind_max_J == 1) * phase_adj_0 + (ind_max_J == 2) * phase_adj_1

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

            # phase_adj=np.angle((1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)

            d_oobounds_0 = current_sig_ws_for_phase[:]
            phase_adj_0 = -cp.arctan(d_oobounds_0.imag / d_oobounds_0.real)
            cond = cp.sin(phase_adj_0) * d_oobounds_0.imag - cp.cos(phase_adj_0) * d_oobounds_0.real
            del d_oobounds_0

            phase_adj_0 = (phase_adj_0) * (
                    1 * (cond) <= 0) + (phase_adj_0 + np.pi) * (
                                  1 * (cond) > 0)

            del cond

            current_sig_ws_0 = (current_sig_ws_for_phase[:] * cp.exp(1j * phase_adj_0)).real
            J_0 = current_sig_ws_0 / cp.sqrt(var_w)

            d_oobounds_1 = current_sig_fs_for_phase[:]
            phase_adj_1 = -cp.arctan(d_oobounds_1.imag / d_oobounds_1.real)
            cond = cp.sin(phase_adj_1) * d_oobounds_1.imag - cp.cos(phase_adj_1) * d_oobounds_1.real
            del d_oobounds_1

            phase_adj_1 = (phase_adj_1) * (
                    1 * (cond) <= 0) + (phase_adj_1 + np.pi) * (
                                  1 * (cond) > 0)

            del cond

            current_sig_fs_1 = (current_sig_fs_for_phase[:] * cp.exp(1j * phase_adj_1)).real
            J_1 = current_sig_fs_1 / cp.sqrt(var_f)

            # print(current_alpha_all_unique.shape)
            # print(J_1.shape)
            # print(J_0.shape)
            # print(alpha_out_bounds.shape)

            current_alpha_all_unique[alpha_out_bounds] = cp.argmax(
                cp.reshape(cp.concatenate([J_0[alpha_out_bounds], J_1[alpha_out_bounds]], axis=-1), (-1, 2)), axis=-1)
            # phase_adj = np.angle((
            #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)

            d = (
                        1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
            phase_adj = -cp.arctan(d.imag / d.real)
            cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real

            del d

            phase_adj = (phase_adj) * (
                    1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                1 * (cond) > 0)

            del cond

            if verbose:
                end = datetime.now()
                print(end - start)

            # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
            if verbose:
                print("Calculating cost for all signals")
                start = datetime.now()

            current_sig_ws = (current_sig_ws_for_phase * cp.exp(1j * phase_adj)).real
            current_sig_fs = (current_sig_fs_for_phase * cp.exp(1j * phase_adj)).real

            # del phase_adj
            del current_sig_ws_for_phase
            del current_sig_fs_for_phase

            J_all = ((
                             1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
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
            phase_adj = (ind_max_J == 0) * phase_adj + (ind_max_J == 1) * phase_adj_0 + (ind_max_J == 2) * phase_adj_1

            idx_max_all_current = cp.argmax(J_all, axis=0)
            idx_max_all_current = idx_max_all_current.get()
            current_alpha_all_unique = current_alpha_all_unique.get()
            phase_adj = phase_adj.get()

            if niter > 0 or log_phase:
                phase_adj=phase_adj.get()


            del current_sig_fs
            del current_sig_ws

            if verbose:
                end = datetime.now()
                print(end - start)

        if verbose:
            print("Extracting index of pattern with max correl")
            start = datetime.now()

        #idx_max_all_current = np.argmax(J_all, axis=0)
        # check_max_correl=np.max(J_all,axis=0)

        if verbose:
            end = datetime.now()
            print(end - start)

        if verbose:
            print("Filling the lists with results for this loop")
            start = datetime.now()

        idx_max_all_unique.extend(idx_max_all_current)
        alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(nb_signals_current)])

        if niter > 0:
            phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])
            J_optim.extend(J_all[idx_max_all_current, np.arange(J_all.shape[1])])

        elif log_phase:
            phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])

        if not (return_matched_signals):
            del phase_adj

        if verbose:
            end = datetime.now()
            print(end - start)

    # idx_max_all_unique = np.argmax(J_all, axis=0)
    if not(return_matched_signals):
        del J_all
    del current_alpha_all_unique

    if niter > 0:
        phase_optim = np.array(phase_optim)
        J_optim = np.array(J_optim)
    elif log_phase:
        phase_optim = np.array(phase_optim)



    # del sig_ws_all_unique
    # del sig_fs_all_unique

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

    if tv_denoising_weight is not None:
        for i,k in enumerate(map_rebuilt.keys()):
            map_rebuilt[k]=denoise_tv_chambolle(makevol(map_rebuilt[k],mask>0),weight=tv_denoising_weight)[mask>0]
            if k!="ff":#Projection back on dictionary parameters
                print("Projection back to dictionary values for {} after denoising".format(k))
                curr_values = np.unique(np.array(keys)[:,i])
                map_rebuilt[k]=curr_values[np.argmin(np.abs(map_rebuilt[k].reshape(-1, 1) - curr_values), axis=-1)]



    if niter==0:


        if not(log_phase):
            if return_matched_signals:
                J_optim=J_all[idx_max_all_current, np.arange(J_all.shape[1])]
                phase_optim=phase_adj[idx_max_all_current, np.arange(J_all.shape[1])]
                print(phase_optim.shape)
                print(J_optim.shape)

                matched_signals=array_water_unique[index_water_unique, :][idx_max_all_current, :].T * (
                        1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
                                                                    idx_max_all_current, :].T * np.array(
                    alpha_optim).reshape(1, -1)
                print(matched_signals.shape)
                matched_signals/=np.linalg.norm(matched_signals,axis=0)
                matched_signals *= J_optim*np.exp(1j*phase_optim)
                return map_rebuilt,None,None,matched_signals.squeeze()
            else:
                return map_rebuilt, None, None
        else:
            return map_rebuilt, None, phase_optim
    else:
        return map_rebuilt,J_optim,phase_optim


class Optimizer(object):

    def __init__(self,log=False,mask=None,verbose=False,useGPU=False,**kwargs):
        self.paramDict=kwargs
        self.paramDict["log"]=log
        self.paramDict["useGPU"]=useGPU
        self.mask=mask
        self.verbose=verbose


    def search_patterns(self,dictfile,volumes,retained_timesteps=None):
        #takes as input dictionary pattern and an array of images or volumes and outputs parametric maps
        raise ValueError("search_patterns should be implemented in child")

class SimpleDictSearch(Optimizer):

    def __init__(self,niter=0,seq=None,trajectory=None,split=500,pca=True,threshold_pca=15,useGPU_dictsearch=False,useGPU_simulation=True,movement_correction=False,cond=None,remove_duplicate_signals=False,threshold=None,tv_denoising_weight=None,log_phase=False,return_matched_signals=False,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)
        self.paramDict["niter"]=niter
        self.paramDict["split"] = split
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = threshold_pca
        self.paramDict["remove_duplicate_signals"] = remove_duplicate_signals
        #self.paramDict["useAdjPred"]=useAdjPred
        self.paramDict["return_matched_signals"] = return_matched_signals

        if niter>0:
#            if seq is None:
#                raise ValueError("When more than 0 iteration, one needs to supply a sequence in order to resimulate the image series")
#            else:
#                self.paramDict["sequence"]=seq
            if trajectory is None:
                raise ValueError("When more than 0 iteration, one needs to supply a kspace trajectory in order to resimulate the image series")
            else:
                self.paramDict["trajectory"]=trajectory

        self.paramDict["useGPU_dictsearch"]=useGPU_dictsearch
        self.paramDict["useGPU_simulation"] = useGPU_simulation
        self.paramDict["movement_correction"]=movement_correction
        self.paramDict["cond"]=cond
        self.paramDict["threshold"]=threshold
        self.paramDict["tv_denoising_weight"] = tv_denoising_weight
        self.paramDict["log_phase"] = log_phase

    def search_patterns(self,dictfile,volumes,retained_timesteps=None):


        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask


        verbose=self.verbose
        niter=self.paramDict["niter"]
        split=self.paramDict["split"]
        pca=self.paramDict["pca"]
        threshold_pca=self.paramDict["threshold_pca"]
        #useAdjPred=self.paramDict["useAdjPred"]
        if niter>0:
            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode=self.paramDict["gen_mode"]
        log=self.paramDict["log"]
        useGPU_dictsearch=self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction=self.paramDict["movement_correction"]
        cond_mvt=self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        #adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        signals = volumes[:, mask > 0]

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, volumes)


        del volumes

        if niter > 0:
            signals0 = signals

        #norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        norm_signals=np.linalg.norm(signals, 2, axis=0)
        all_signals = signals/norm_signals

        mrfdict = dictsearch.Dictionary()
        mrfdict.load(dictfile, force=True)

        keys = mrfdict.keys
        array_water = mrfdict.values[:, :, 0]
        array_fat = mrfdict.values[:, :, 1]

        del mrfdict

        if retained_timesteps is not None:
            array_water=array_water[:,retained_timesteps]
            array_fat=array_fat[:,retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):



            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))

            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            nb_signals = all_signals.shape[1]

            if remove_duplicates:
                all_signals, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)
                nb_signals = all_signals.shape[1]



            print("There are {} unique signals to match along {} water and {} fat components".format(nb_signals,array_water_unique.shape[0],array_fat_unique.shape[0]))



            num_group = int(nb_signals / split) + 1

            idx_max_all_unique = []
            alpha_optim = []

            if niter>0:
                phase_optim=[]
                J_optim=[]

            for j in tqdm(range(num_group)):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals)

                if self.verbose:
                    print("PCA transform")
                    start = datetime.now()

                if not(useGPU_dictsearch):

                    if pca:
                        transformed_all_signals_water = np.transpose(pca_water.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))
                        transformed_all_signals_fat = np.transpose(pca_fat.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))

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
                        transformed_all_signals_fat = cp.transpose(pca_fat.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next])))).get()

                        sig_ws_all_unique = (cp.matmul(cp.asarray(transformed_array_water_unique),
                                                      cp.asarray(transformed_all_signals_water).conj())).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(transformed_array_fat_unique),
                                                      cp.asarray(transformed_all_signals_fat).conj())).get()
                    else:

                        sig_ws_all_unique = (cp.matmul(cp.asarray(array_water_unique),
                                                      cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()
                        sig_fs_all_unique = (cp.matmul(cp.asarray(array_fat_unique),
                                                      cp.asarray(all_signals)[:, j_signal:j_signal_next].conj())).get()


                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if self.verbose:
                    print("Extracting all sig_ws and sig_fs")
                    start = datetime.now()



                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]

                #current_sig_ws = current_sig_ws_for_phase.real
                #current_sig_fs = current_sig_fs_for_phase.real

                if self.verbose:
                    end = datetime.now()
                    print(end-start)

                if not(useGPU_dictsearch):
                    #if adj_phase:
                    if self.verbose:
                        print("Adjusting Phase")
                        print("Calculating alpha optim and flooring")

                        ### Testing direct phase solving
                    A = sig_wf*current_sig_ws_for_phase-var_w*current_sig_fs_for_phase
                    B = (current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

                    a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    discr = b**2-4*a*c
                    alpha1 = (-b + np.sqrt(discr)) / (2 * a)
                    alpha2 = (-b - np.sqrt(discr)) / (2 * a)


                    del a
                    del b
                    del c
                    del discr

                    current_alpha_all_unique=(1 * (alpha1 >= 0) & (alpha1 <= 1)) * alpha1 + (1 - (1*(alpha1 >= 0) & (alpha1 <= 1))) * alpha2

                        #current_alpha_all_unique_2 = (1 * (alpha2 >= 0) & (alpha2 <= 1)) * alpha2 + (
                        #            1 - (1*(alpha2 >= 0) & (alpha2 <= 1))) * alpha1

                    del alpha1
                    del alpha2

                    if self.verbose:
                        start = datetime.now()

                    current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

                        #phase_adj=np.angle((1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)


                    d = (1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj = -np.arctan(d.imag /d.real)
                    cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
                    del d

                    phase_adj = (phase_adj) * (
                                    1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                                 1 * (cond) > 0)

                    del cond

                    if self.verbose:
                        end=datetime.now()
                        print(end-start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if self.verbose:
                        print("Calculating cost for all signals")
                    start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase*np.exp(1j*phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real


                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)
                    end = datetime.now()

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
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag +B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    del A
                    del B

                    #del beta
                    #del delta
                    #del gamma
                    #del nu

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

                    current_alpha_all_unique = cp.minimum(cp.maximum(current_alpha_all_unique, 0.0), 1.0)

                    #phase_adj = np.angle((
                    #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)


                    d = (1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj=-cp.arctan(d.imag / d.real)
                    cond = cp.sin(phase_adj)*d.imag-cp.cos(phase_adj)*d.real<=0

                    del d

                    phase_adj=phase_adj*(1*(cond))+(phase_adj+np.pi)*(1*(1-cond))

                    del cond


                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if verbose:
                        print("Calculating cost for all signals")
                        start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real



                    #del phase_adj
                    del current_sig_ws_for_phase
                    del current_sig_fs_for_phase

                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)

                    J_all = J_all.get()
                    current_alpha_all_unique=current_alpha_all_unique.get()

                    del current_sig_fs
                    del current_sig_ws


                    if verbose:
                        end = datetime.now()
                        print(end - start)


                if verbose:
                    print("Extracting index of pattern with max correl")
                    start = datetime.now()

                idx_max_all_current = np.argmax(J_all, axis=0)
                #check_max_correl=np.max(J_all,axis=0)

                if verbose:
                    end = datetime.now()
                    print(end-start)

                if verbose:
                    print("Filling the lists with results for this loop")
                    start = datetime.now()

                idx_max_all_unique.extend(idx_max_all_current)
                alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])])

                if niter>0:
                    phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])
                    J_optim.extend(J_all[idx_max_all_current, np.arange(J_all.shape[1])])

                del phase_adj

                if verbose:
                    end = datetime.now()
                    print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all
            del current_alpha_all_unique

            if niter>0:
                phase_optim=np.array(phase_optim)
                J_optim = np.array(J_optim)

            print("Building the maps for iteration {}".format(i))

            # del sig_ws_all_unique
            # del sig_fs_all_unique

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

            values_results.append((map_rebuilt, mask))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i):(map_rebuilt, mask)}, f)

            if useGPU_dictsearch:#Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()

                cp.cuda.set_allocator(None)
                # Disable memory pool for pinned memory (CPU).
                cp.cuda.set_pinned_memory_allocator(None)
                gc.collect()




            if i == niter:
                break

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            keys_simu = list(map_rebuilt.keys())
            values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            map_for_sim = dict(zip(keys_simu, values_simu))

            # predict spokes
            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True,gen_mode=gen_mode)
            images_pred.buildParamMap()

            del map_for_sim

            del keys_simu
            del values_simu

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            images_pred.build_ref_images(seq,norm=J_optim*norm_signals,phase=phase_optim)

            print("Normalizing image series")


            print("Filling images series with renormalized signals")

            kdatai = images_pred.generate_kdata(trajectory,useGPU=useGPU_simulation)

            if log:
                print("Saving Ideal Volumes for iteration {}".format(i))
                with open('./log/predvolumes_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, images_pred.images_series[::nspoke].astype(np.complex64))

            del images_pred

            if movement_correction:
                traj=trajectory.get_traj()
                kdatai, traj_retained_final, _ = correct_mvt_kdata(kdatai, trajectory, cond_mvt,self.paramDict["ntimesteps"],density_adj=True)

            kdatai = np.array(kdatai)
            #nans = np.nonzero(np.isnan(kdatai))
            nans = [np.nonzero(np.isnan(k))[0] for k in kdatai]
            nans_count = np.array([len(n) for n in nans]).sum()

            if nans_count>0:
                print("Warning : Nan Values replaced by zeros in rebuilt kdata")
                for i,k in enumerate(kdatai):
                    kdatai[i][nans[i]]=0.0

            if not(movement_correction):
                volumesi = simulate_radial_undersampled_images(kdatai,trajectory,mask.shape,useGPU=useGPU_simulation,density_adj=True)

            else:
                trajectory.traj_for_reconstruction=traj_retained_final
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape,
                                                               useGPU=useGPU_simulation, density_adj=True,is_theta_z_adjusted=True)

            #volumesi/=(2*np.pi)

            nans_volumes = np.argwhere(np.isnan(volumesi))
            if len(nans_volumes) > 0:
                np.save('./log/kdatai.npy', kdatai)
                np.save('./log/volumesi.npy',volumesi)
                raise ValueError("Error : Nan Values in volumes")

            del kdatai

            #volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)


            if log:
                print("Saving correction volumes for iteration {}".format(i))
                with open('./log/volumes1_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumesi))

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:,mask>0]
            #normi= np.linalg.norm(signalsi, axis=0)
            #signalsi *= norm_signals/normi

            del volumesi


            signals += signals0 - signalsi

            if threshold is not None:
                signals_for_matching = signals * (1-(np.abs(signals)>threshold))
            else:
                signals_for_matching = signals

            norm_signals = np.linalg.norm(signals_for_matching,axis=0)


            all_signals =signals_for_matching/norm_signals

        if log:
            print(date_time)

        return dict(zip(keys_results, values_results))

    def search_patterns_MRF_denoising(self, dictfile, volumes, retained_timesteps=None):

        gw = self.paramDict["Weighting"]
        sig_0=gw.sig

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        signals = volumes[:, mask > 0]

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, volumes)

        del volumes

        if niter > 0:
            signals0 = signals

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        norm_signals = np.linalg.norm(signals, 2, axis=0)
        all_signals = signals / norm_signals

        mrfdict = dictsearch.Dictionary()
        mrfdict.load(dictfile, force=True)

        keys = mrfdict.keys
        array_water = mrfdict.values[:, :, 0]
        array_fat = mrfdict.values[:, :, 1]

        del mrfdict

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))

            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
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

            idx_max_all_unique = []
            alpha_optim = []

            if niter > 0:
                phase_optim = []
                J_optim=[]

            for j in tqdm(range(num_group)):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals)

                if self.verbose:
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

                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if self.verbose:
                    print("Extracting all sig_ws and sig_fs")
                    start = datetime.now()

                current_sig_ws_for_phase = sig_ws_all_unique[index_water_unique, :]
                current_sig_fs_for_phase = sig_fs_all_unique[index_fat_unique, :]


                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if not (useGPU_dictsearch):
                    # if adj_phase:
                    if self.verbose:
                        print("Adjusting Phase")
                        print("Calculating alpha optim and flooring")

                    A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                    B = (
                                    current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase

                    a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    # a = beta + delta
                    # b = gamma-delta+nu
                    # c=-gamma

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

                    if self.verbose:
                        start = datetime.now()

                    current_alpha_all_unique = np.minimum(np.maximum(current_alpha_all_unique, 0.0), 1.0)

                    d = (
                                    1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj = -np.arctan(d.imag / d.real)
                    cond = np.sin(phase_adj) * d.imag - np.cos(phase_adj) * d.real
                    del d

                    phase_adj = (phase_adj) * (
                            1 * (cond) <= 0) + (phase_adj + np.pi) * (
                                        1 * (cond) > 0)

                    del cond

                    if self.verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if self.verbose:
                        print("Calculating cost for all signals")
                    start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real


                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)
                    end = datetime.now()

                else:
                    if verbose:
                        print("Calculating alpha optim and flooring")
                        start = datetime.now()

                    current_sig_ws_for_phase = cp.asarray(current_sig_ws_for_phase)
                    current_sig_fs_for_phase = cp.asarray(current_sig_fs_for_phase)


                    A = sig_wf * current_sig_ws_for_phase - var_w * current_sig_fs_for_phase
                    B = (
                                current_sig_ws_for_phase + current_sig_fs_for_phase) * sig_wf - var_w * current_sig_fs_for_phase - var_f * current_sig_ws_for_phase


                    a = B.real * current_sig_fs_for_phase.real + B.imag * current_sig_fs_for_phase.imag - B.imag * current_sig_ws_for_phase.imag - B.real * current_sig_ws_for_phase.real
                    b = A.real * current_sig_ws_for_phase.real + A.imag * current_sig_ws_for_phase.imag + B.imag * current_sig_ws_for_phase.imag + B.real * current_sig_ws_for_phase.real - A.imag * current_sig_fs_for_phase.imag - A.real * current_sig_fs_for_phase.real
                    c = -A.real * current_sig_ws_for_phase.real - A.imag * current_sig_ws_for_phase.imag

                    del A
                    del B


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

                    current_alpha_all_unique = cp.minimum(cp.maximum(current_alpha_all_unique, 0.0), 1.0)

                    # phase_adj = np.angle((
                    #                                 1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase)

                    d = (
                                    1 - current_alpha_all_unique) * current_sig_ws_for_phase + current_alpha_all_unique * current_sig_fs_for_phase
                    phase_adj = -cp.arctan(d.imag / d.real)
                    cond = cp.sin(phase_adj) * d.imag - cp.cos(phase_adj) * d.real <= 0

                    del d

                    phase_adj = phase_adj * (1 * (cond)) + (phase_adj + np.pi) * (1 * (1 - cond))

                    del cond

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                    # alpha_all_unique[:, j_signal:j_signal_next] = current_alpha_all_unique
                    if verbose:
                        print("Calculating cost for all signals")
                        start = datetime.now()

                    current_sig_ws = (current_sig_ws_for_phase * np.exp(1j * phase_adj)).real
                    current_sig_fs = (current_sig_fs_for_phase * np.exp(1j * phase_adj)).real

                    # del phase_adj
                    del current_sig_ws_for_phase
                    del current_sig_fs_for_phase

                    J_all = ((
                                     1 - current_alpha_all_unique) * current_sig_ws + current_alpha_all_unique * current_sig_fs) / np.sqrt(
                        (
                                1 - current_alpha_all_unique) ** 2 * var_w + current_alpha_all_unique ** 2 * var_f + 2 * current_alpha_all_unique * (
                                1 - current_alpha_all_unique) * sig_wf)

                    J_all = J_all.get()
                    current_alpha_all_unique = current_alpha_all_unique.get()

                    del current_sig_fs
                    del current_sig_ws

                    if verbose:
                        end = datetime.now()
                        print(end - start)

                if verbose:
                    print("Extracting index of pattern with max correl")
                    start = datetime.now()

                idx_max_all_current = np.argmax(J_all, axis=0)
                # check_max_correl=np.max(J_all,axis=0)

                if verbose:
                    end = datetime.now()
                    print(end - start)

                if verbose:
                    print("Filling the lists with results for this loop")
                    start = datetime.now()

                idx_max_all_unique.extend(idx_max_all_current)
                alpha_optim.extend(current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])])

                if niter > 0:
                    phase_optim.extend(phase_adj[idx_max_all_current, np.arange(J_all.shape[1])])
                    J_optim.extend(J_all[idx_max_all_current, np.arange(J_all.shape[1])])
                del phase_adj

                if verbose:
                    end = datetime.now()
                    print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all
            del current_alpha_all_unique

            if niter > 0:
                phase_optim = np.array(phase_optim)
                J_optim = np.array(J_optim)
            print("Building the maps for iteration {}".format(i))

            # del sig_ws_all_unique
            # del sig_fs_all_unique

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

            values_results.append((map_rebuilt, mask))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()

                cp.cuda.set_allocator(None)
                # Disable memory pool for pinned memory (CPU).
                cp.cuda.set_pinned_memory_allocator(None)
                gc.collect()

            if i == niter:
                break

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            keys_simu = list(map_rebuilt.keys())
            values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            map_for_sim = dict(zip(keys_simu, values_simu))

            # predict spokes
            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True, gen_mode=gen_mode)
            images_pred.buildParamMap()

            del map_for_sim

            del keys_simu
            del values_simu

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            images_pred.build_ref_images(seq, norm=norm_signals*J_optim,phase=np.array(phase_optim))

            kdatai = images_pred.generate_kdata(trajectory, useGPU=useGPU_simulation)

            gw.sig=gw.sig*(np.pi/sig_0)**(1/niter)
            traj=trajectory.get_traj()
            kdatai = [gw.apply(traj[j])*k for j,k in enumerate(kdatai)]

            if log:
                print("Saving Ideal Volumes for iteration {}".format(i))
                with open('./log/predvolumes_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, images_pred.images_series[::nspoke].astype(np.complex64))

            del images_pred

            if movement_correction:
                traj = trajectory.get_traj()
                kdatai, traj_retained_final, _ = correct_mvt_kdata(kdatai, trajectory, cond_mvt,
                                                                   self.paramDict["ntimesteps"], density_adj=True)

            kdatai = np.array(kdatai)
            # nans = np.nonzero(np.isnan(kdatai))
            nans = [np.nonzero(np.isnan(k))[0] for k in kdatai]
            nans_count = np.array([len(n) for n in nans]).sum()

            if nans_count > 0:
                print("Warning : Nan Values replaced by zeros in rebuilt kdata")
                for i, k in enumerate(kdatai):
                    kdatai[i][nans[i]] = 0.0

            if not (movement_correction):
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape, useGPU=useGPU_simulation,
                                                               density_adj=True)

            else:
                trajectory.traj_for_reconstruction = traj_retained_final
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape,
                                                               useGPU=useGPU_simulation, density_adj=True,
                                                               is_theta_z_adjusted=True)

            # volumesi/=(2*np.pi)

            nans_volumes = np.argwhere(np.isnan(volumesi))
            if len(nans_volumes) > 0:
                np.save('./log/kdatai.npy', kdatai)
                np.save('./log/volumesi.npy', volumesi)
                raise ValueError("Error : Nan Values in volumes")

            del kdatai

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            if log:
                print("Saving correction volumes for iteration {}".format(i))
                with open('./log/volumes1_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(volumesi))

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signals = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            del volumesi


            norm_signals = np.linalg.norm(signals, axis=0)

            all_signals = signals / norm_signals

        if log:
            print(date_time)

        return dict(zip(keys_results, values_results))

    def search_patterns_matrix(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            #seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase=self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim > 2:
            signals = volumes[:, mask > 0]
        else:  # already masked
            signals = volumes

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, volumes)

        del volumes

        if niter > 0:
            signals0 = signals

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        #norm_signals = np.linalg.norm(signals, 2, axis=0)
        all_signals = signals #/ norm_signals

        mrfdict = dictsearch.Dictionary()
        mrfdict.load(dictfile, force=True)

        keys = mrfdict.keys


        if pca:
            pca=PCAComplex(n_components_=threshold_pca)
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]
            array_for_pca = np.concatenate([np.unique(array_water, axis=0), np.unique(array_fat, axis=0)], axis=0)
            pca.fit(array_for_pca)
            del array_for_pca
            array_dict = np.stack([pca.transform(array_water), pca.transform(array_fat)], axis=-1)
            del array_water
            del array_fat

        else:
            array_dict = mrfdict.values

        del mrfdict.values

        M = np.real(np.einsum('ijk,ijl->ikl', array_dict.conj(), array_dict))
        M_inv = np.linalg.pinv(M)

        M=np.expand_dims(M,axis=1)
        M_inv = np.expand_dims(M_inv, axis=1)

        M_w = np.real(np.einsum('ijk,ijl->ikl', np.expand_dims(array_dict[:,:,0],axis=-1).conj(), np.expand_dims(array_dict[:,:,0],axis=-1)))
        M_inv_w = np.linalg.pinv(M_w)

        M_w = np.expand_dims(M_w, axis=1)
        M_inv_w = np.expand_dims(M_inv_w, axis=1)

        M_f = np.real(np.einsum('ijk,ijl->ikl', np.expand_dims(array_dict[:, :, 1], axis=-1).conj(),
                                np.expand_dims(array_dict[:, :, 1], axis=-1)))
        M_inv_f = np.linalg.pinv(M_f)

        M_f = np.expand_dims(M_f, axis=1)
        M_inv_f = np.expand_dims(M_inv_f, axis=1)

        if useGPU_dictsearch:
            #M = cp.asarray(M)
            M_inv=cp.asarray(M_inv)
            #M_w = cp.asarray(M_w)
            M_inv_w = cp.asarray(M_inv_w)
            #M_f = cp.asarray(M_f)
            M_inv_f = cp.asarray(M_inv_f)
            array_dict=cp.asarray(array_dict)


        values_results = []
        keys_results = list(range(1))


        nb_signals = all_signals.shape[1]

        print("There are {} unique signals to match along {} components".format(nb_signals,
                                                                                array_dict.shape[
                                                                                    0]))

        num_group = int(nb_signals / split) + 1

        idx_max_all_unique = []
        alpha_optim = []

        if niter > 0:
            phase_optim = []
            J_optim = []

        elif log_phase:
            phase_optim = []

        for j in tqdm(range(num_group)):
            j_signal = j * split
            j_signal_next = np.minimum((j + 1) * split, nb_signals)

            if verbose:
                print("PCA transform")
                start = datetime.now()

            if not (useGPU_dictsearch):

                if pca:
                    transformed_all_signals = np.transpose(
                        pca.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))


                else:
                    transformed_all_signals = all_signals[:, j_signal:j_signal_next]

                cov = np.einsum('ijk,jl->ilk', array_dict.conj(), transformed_all_signals)[...,None]
                phi=0.5*np.angle(cov.transpose((0,1,3,2))@M_inv@cov)
                cov_adjusted = np.real(np.einsum('ijk,jl->ilk', array_dict.conj(), transformed_all_signals)[...,None]*np.exp(-1j*phi))

                del cov
                lambd = np.squeeze(M_inv@cov_adjusted)
                #del cov_adjusted
                J_all = np.linalg.norm(
                    np.einsum('ijk,ilk->ijl', array_dict, lambd) * np.expand_dims(np.exp(1j * phi.squeeze()),
                                                                                  axis=1) - np.expand_dims(
                        transformed_all_signals, axis=0), axis=1)

                #
                select =np.argwhere(lambd[:,:,0]*lambd[:,:,1]<0)

                cov_w=np.einsum('ijk,jl->ilk', np.expand_dims(array_dict[:,:,0].conj(),axis=-1), transformed_all_signals)[..., None]
                phi_w = 0.5 * np.angle(cov_w.transpose((0, 1, 3, 2)) @ M_inv_w @ cov_w)
                cov_adjusted_w = np.real(
                    np.einsum('ijk,jl->ilk', np.expand_dims(array_dict[:,:,0].conj(),axis=-1), transformed_all_signals)[..., None] * np.exp(-1j * phi_w))
                del cov_w
                lambd_w = (M_inv_w @ cov_adjusted_w)[:,:,:,0]
                del cov_adjusted_w

                cov_f =np.einsum('ijk,jl->ilk', np.expand_dims(array_dict[:, :, 1].conj(), axis=-1), transformed_all_signals)[..., None]
                phi_f = 0.5 * np.angle(cov_f.transpose((0, 1, 3, 2)) @ M_inv_f @ cov_f)
                cov_adjusted_f = np.real(
                    np.einsum('ijk,jl->ilk', np.expand_dims(array_dict[:, :, 1].conj(), axis=-1),transformed_all_signals)[..., None] * np.exp(-1j * phi_f))
                del cov_f
                lambd_f = (M_inv_f @ cov_adjusted_f)[:,:,:,0]
                del cov_adjusted_f

                J_all_w = np.linalg.norm(
                    np.einsum('ijk,ilk->ijl', np.expand_dims(array_dict[:, :, 0],axis=-1).conj(), lambd_w) * np.expand_dims(np.exp(1j * phi_w.squeeze()),
                                                                                  axis=1) - np.expand_dims(
                        transformed_all_signals, axis=0), axis=1)

                J_all_f = np.linalg.norm(
                    np.einsum('ijk,ilk->ijl', np.expand_dims(array_dict[:, :, 1], axis=-1).conj(),
                              lambd_f) * np.expand_dims(np.exp(1j * phi_f.squeeze()),
                                                        axis=1) - np.expand_dims(
                        transformed_all_signals, axis=0), axis=1)

                all_J_select = np.stack([J_all_w[select[:,0],select[:,1]], J_all_f[select[:,0],select[:,1]]], axis=0)

                ind_min_J = np.argmin(all_J_select, axis=0)

                del all_J_select

                J_all[select[:,0],select[:,1]]=J_all_w[select[:,0],select[:,1]]*(ind_min_J==0)+J_all_f[select[:,0],select[:,1]]*(ind_min_J==1)
                del J_all_w
                del J_all_f

                lambd[select[:,0],select[:,1],:] = np.expand_dims((ind_min_J == 0),axis=-1) * np.stack([lambd_w[select[:,0],select[:,1],0],np.zeros(lambd_w[select[:,0],select[:,1],0].shape)],axis=-1) + \
                        np.expand_dims((ind_min_J == 1),axis=-1) * np.stack([np.zeros(lambd_f[select[:,0],select[:,1],0].shape),lambd_f[select[:,0],select[:,1],0]],axis=-1)
                epsilon = 1e-8
                lambd[np.all(lambd < -epsilon, axis=-1)] = 0

                while True:

                    ispos = lambd > -epsilon
                    numpos = np.count_nonzero(ispos, axis=-1)
                    minpos = np.min(numpos)
                    if minpos == lambd.shape[-1]:
                        break

                    select = np.nonzero(numpos == minpos)
                    active = np.nonzero(ispos[select])[1].reshape(-1, minpos)
                    lambd[select]=0
                    if np.array(select).shape[-1]>1:
                        lambd[select[0],select[1],np.squeeze(active.T)]=np.squeeze((np.squeeze(M_inv[select[0]]) @ cov_adjusted[select]))[np.arange(len(select[0])), np.squeeze(active.T)]
                    else:
                        lambd[select[0], select[1], np.squeeze(active.T)] = \
                        (np.squeeze(M_inv[select[0]]) @ cov_adjusted[select])[
                            np.arange(len(select[0])), np.squeeze(active.T)]



                if not self.paramDict["return_matched_signals"]:
                    del phi
                else:
                    phi[select[:, 0], select[:, 1]] = (ind_min_J == 0) * phi_w[select[:, 0], select[:, 1]] + (
                                ind_min_J == 1) * phi_f[select[:, 0], select[:, 1]]

                idx_max_all_current = np.argmin(J_all, axis=0)
                #print("Number of zero lambda {} : ".format(np.sum(np.sum(lambd,axis=-1)==0)))
                current_alpha_all_unique=lambd[:,:,-1]/np.sum(lambd,axis=-1)
                current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]

                if not self.paramDict["return_matched_signals"]:
                    del lambd

            else:

                if pca:

                    transformed_all_signals = cp.transpose(
                        pca.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next]))))#.get()

                else:
                    transformed_all_signals = cp.asarray(all_signals[:, j_signal:j_signal_next])  # .get()

                cov=cp.einsum('ijk,jl->ilk', array_dict.conj(), transformed_all_signals)[...,None]

                phi = 0.5*cp.angle(cp.matmul(cov.transpose((0, 1, 3, 2)),cp.matmul(M_inv,cov)))
                cov_adjusted = cp.real(
                    cp.einsum('ijk,jl->ilk', array_dict.conj(), transformed_all_signals)[..., None] * cp.exp(
                        -1j * phi))
                del cov
                lambd = cp.matmul(M_inv,cov_adjusted)[:,:,:,0]
                phi=phi[:,:,0, 0]

                #del cov_adjusted
                J_all = cp.linalg.norm(
                    cp.einsum('ijk,ilk->ijl', array_dict, lambd) * cp.expand_dims(cp.exp(1j * phi),
                                                        axis=1)- cp.expand_dims(transformed_all_signals, axis=0),
                    axis=1)


                select = cp.argwhere(lambd[:, :, 0] * lambd[:, :, 1] < 0)

                #print("Calculating lambda water for all signals")

                #print("Calculating cov water for all signals")

                cov_w = cp.einsum('ijk,jl->ilk',cp.expand_dims(array_dict[:, :, 0].conj(), axis=-1) , transformed_all_signals)[..., None]

                #print("Calculating phi water for all signals")
                phi_w = 0.5 * cp.angle(cp.matmul(cov_w.transpose((0, 1, 3, 2)),cp.matmul(M_inv_w,cov_w)))
                #print("Calculating cov adjusted water for all signals")
                cov_adjusted_w = cp.real(
                    cp.einsum('ijk,jl->ilk', cp.expand_dims(array_dict[:, :, 0].conj(), axis=-1),
                              transformed_all_signals)[..., None] * cp.exp(-1j * phi_w))
                del cov_w
                #print("Calculating final lambda water for all signals")
                lambd_w = (cp.matmul(M_inv_w,cov_adjusted_w))[:, :, :, 0]
                phi_w = phi_w[:, :, 0, 0]
                del cov_adjusted_w

                #print("Calculating lambda fat for all signals")

                cov_f = cp.einsum('ijk,jl->ilk', cp.expand_dims(array_dict[:, :, 1].conj(), axis=-1), transformed_all_signals)[..., None]
                phi_f = 0.5 * cp.angle(cp.matmul(cov_f.transpose((0, 1, 3, 2)), cp.matmul(M_inv_f, cov_f)))
                cov_adjusted_f = cp.real(
                    cp.einsum('ijk,jl->ilk', cp.expand_dims(array_dict[:, :, 1].conj(), axis=-1),
                              transformed_all_signals)[..., None] * cp.exp(-1j * phi_f))
                del cov_f

                lambd_f = (cp.matmul(M_inv_f, cov_adjusted_f))[:, :, :, 0]
                phi_f = phi_f[:, :, 0, 0]
                del cov_adjusted_f

                J_all_w = cp.linalg.norm(
                    cp.einsum('ijk,ilk->ijl', cp.expand_dims(array_dict[:, :, 0], axis=-1).conj(),
                              lambd_w) * cp.expand_dims(cp.exp(1j * phi_w),
                                                        axis=1) - cp.expand_dims(
                        transformed_all_signals, axis=0), axis=1)

                J_all_f = cp.linalg.norm(
                    cp.einsum('ijk,ilk->ijl', cp.expand_dims(array_dict[:, :, 1], axis=-1).conj(),
                              lambd_f) * cp.expand_dims(cp.exp(1j * phi_f),
                                                        axis=1) - cp.expand_dims(
                        transformed_all_signals, axis=0), axis=1)

                #print("Aggregating J")
                all_J_select = cp.stack([J_all_w[select[:, 0], select[:, 1]], J_all_f[select[:, 0], select[:, 1]]],
                                        axis=0)

                ind_min_J = cp.argmin(all_J_select, axis=0)

                del all_J_select

                J_all[select[:, 0], select[:, 1]] = J_all_w[select[:, 0], select[:, 1]] * (ind_min_J == 0) + J_all_f[
                    select[:, 0], select[:, 1]] * (ind_min_J == 1)
                del J_all_w
                del J_all_f

                lambd[select[:, 0], select[:, 1], :] = cp.expand_dims((ind_min_J == 0), axis=-1) * cp.stack(
                    [lambd_w[select[:, 0], select[:, 1], 0], cp.zeros(lambd_w[select[:, 0], select[:, 1], 0].shape)],
                    axis=-1) + \
                                                       cp.expand_dims((ind_min_J == 1), axis=-1) * cp.stack(
                    [cp.zeros(lambd_f[select[:, 0], select[:, 1], 0].shape), lambd_f[select[:, 0], select[:, 1], 0]],
                    axis=-1)


                epsilon = 1e-8
                lambd[cp.all(lambd < -epsilon, axis=-1)] = 0

                while True:

                    ispos = lambd > -epsilon
                    numpos = cp.count_nonzero(ispos, axis=-1)
                    minpos = cp.min(numpos)
                    if minpos == lambd.shape[-1]:
                        break

                    select = cp.nonzero(numpos == minpos)
                    active = cp.nonzero(ispos[select])[1].reshape(-1, minpos)
                    lambd[select] = 0
                    if select.shape[-1] > 1:
                        lambd[select[0], select[1], cp.squeeze(active.T)] = \
                        cp.squeeze((cp.squeeze(M_inv[select[0]]) @ cov_adjusted[select]))[
                            cp.arange(len(select[0])), cp.squeeze(active.T)]
                    else:
                        lambd[select[0], select[1], cp.squeeze(active.T)] = \
                            (cp.squeeze(M_inv[select[0]]) @ cov_adjusted[select])[
                                cp.arange(len(select[0])), cp.squeeze(active.T)]

                current_alpha_all_unique = lambd[:, :, -1] / cp.sum(lambd, axis=-1)

                if not((self.paramDict["return_matched_signals"]) or niter>0):
                    del lambd
                    del phi
                else:
                    lambd=lambd.get()
                    phi[select[:, 0], select[:, 1]] = (ind_min_J == 0)*phi_w[select[:, 0], select[:, 1]] + (ind_min_J == 1)*phi_f[select[:, 0], select[:, 1]]
                    phi=phi.get()

                idx_max_all_current=cp.argmin(J_all, axis=0)

                current_alpha_all_unique_optim = current_alpha_all_unique[idx_max_all_current, np.arange(J_all.shape[1])]
                current_alpha_all_unique_optim=current_alpha_all_unique_optim.get()
                idx_max_all_current = idx_max_all_current.get()


            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                print("Extracting index of pattern with max correl")
                start = datetime.now()

            #idx_max_all_current = np.argmin(J_all, axis=0)
            # check_max_correl=np.max(J_all,axis=0)

            if verbose:
                end = datetime.now()
                print(end - start)

            if verbose:
                print("Filling the lists with results for this loop")
                start = datetime.now()

            idx_max_all_unique.extend(idx_max_all_current)
            alpha_optim.extend(current_alpha_all_unique_optim)



            if verbose:
                end = datetime.now()
                print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all
            del current_alpha_all_unique


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

        values_results.append((map_rebuilt, mask))


        if self.paramDict["return_matched_signals"]:
            matched_signals=(np.einsum('ijk,ilk->ijl', array_dict, lambd) * np.expand_dims(np.exp(1j * phi.squeeze()), axis=1))
            matched_signals=matched_signals[idx_max_all_unique, :, np.arange(nb_signals)].T
            if pca:
                matched_signals=pca.components_@matched_signals

            return dict(zip(keys_results, values_results)),matched_signals

        else:
            return dict(zip(keys_results, values_results))


    def search_patterns_test(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
#            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase=self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim>2:
            signals = volumes[:, mask > 0]
        else:#already masked
            signals=volumes

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, signals)

        del volumes

        if niter > 0:
            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=None, rounding=True, gen_mode=gen_mode)
            # images_pred.buildParamMap()
            images_pred.mask = mask
            images_pred.image_size = mask.shape

            all_signals_extended = np.array([makevol(im, mask > 0) for im in signals])
            images_pred.images_series = np.repeat(all_signals_extended, nspoke, axis=0)
            kdata_all_signals = images_pred.generate_kdata(trajectory, useGPU=useGPU_simulation)

            del all_signals_extended
            del images_pred

            volumes_all_signals = simulate_radial_undersampled_images(kdata_all_signals, trajectory, mask.shape,
                                                                      useGPU=useGPU_simulation,
                                                                      density_adj=True,
                                                                      ntimesteps=self.paramDict["ntimesteps"])

            mu_numerator = 0
            mu_denom = 0

            for ts in tqdm(range(volumes_all_signals.shape[0])):
                curr_grad = signals[ts]
                curr_volumes_grad = volumes_all_signals[ts][mask > 0]
                mu_denom += np.real(np.dot(curr_grad.conj(), curr_volumes_grad))
                mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

            mu0 = mu_numerator / mu_denom
            signals*=mu0
            print("Mu0 : {}".format(mu0))
            signals0 = signals


        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        #norm_signals = np.linalg.norm(signals, 2, axis=0)
        #Normalize
        #all_signals = signals / norm_signals
        all_signals=signals
        if type(dictfile)==str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        else:#otherwise dictfile contains (s_w,s_f,keys)
            array_water=dictfile[0]
            array_fat=dictfile[1]
            keys=dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        else:
            pca_water=None
            pca_fat=None
            transformed_array_water_unique=None
            transformed_array_fat_unique=None

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):

            if log:
                print("Saving signals for iteration {}".format(i))
                with open('./log/signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(all_signals))

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not(self.paramDict["return_matched_signals"])and(niter==0):
                map_rebuilt,J_optim,phase_optim=match_signals_v2(all_signals,keys, pca_water, pca_fat, array_water_unique, array_fat_unique,
                          transformed_array_water_unique, transformed_array_fat_unique, var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates, verbose,
                          niter, split, useGPU_dictsearch,mask,tv_denoising_weight,log_phase)
            else:
                map_rebuilt, J_optim, phase_optim,matched_signals = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                  array_water_unique, array_fat_unique,
                                                                  transformed_array_water_unique,
                                                                  transformed_array_fat_unique, var_w, var_f, sig_wf,
                                                                  pca, index_water_unique, index_fat_unique,
                                                                  remove_duplicates, verbose,
                                                                  niter, split, useGPU_dictsearch, mask,
                                                                  tv_denoising_weight, log_phase,return_matched_signals=True)

            print("Maps build for iteration {}".format(i))

            #import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);

            if not(log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask,phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if i == niter:
                break

            if i>0 and threshold is not None :
                all_signals=all_signals_unthresholded
                if not(self.paramDict["return_matched_signals"]):
                    map_rebuilt,J_optim,phase_optim = match_signals_v2(all_signals,keys, pca_water, pca_fat, array_water_unique, array_fat_unique,
                              transformed_array_water_unique, transformed_array_fat_unique, var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates, verbose,
                              niter, split, useGPU_dictsearch,mask,tv_denoising_weight)
                else:
                    map_rebuilt, J_optim, phase_optim,matched_signals = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                      array_water_unique, array_fat_unique,
                                                                      transformed_array_water_unique,
                                                                      transformed_array_fat_unique, var_w, var_f,
                                                                      sig_wf, pca, index_water_unique, index_fat_unique,
                                                                      remove_duplicates, verbose,
                                                                      niter, split, useGPU_dictsearch, mask,
                                                                      tv_denoising_weight,return_matched_signals=True)

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            #keys_simu = list(map_rebuilt.keys())
            #values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            #map_for_sim = dict(zip(keys_simu, values_simu))

            # predict spokes
            images_pred = MapFromDict("RebuiltMapFromParams", paramMap=None, rounding=True, gen_mode=gen_mode)
            #images_pred.buildParamMap()
            images_pred.mask=mask
            images_pred.image_size=mask.shape

            #del map_for_sim

            #del keys_simu
            #del values_simu



            #images_pred.build_ref_images(seq, norm=J_optim * norm_signals, phase=phase_optim)
            #matched_signals_extended=np.repeat(matched_signals,nspoke,axis=0)
            matched_signals=matched_signals.astype("complex64")
            matched_signals_extended=np.array([makevol(im, mask > 0) for im in matched_signals])
            images_pred.images_series=np.repeat(matched_signals_extended,nspoke,axis=0)
            kdatai = images_pred.generate_kdata(trajectory, useGPU=useGPU_simulation)

            del images_pred.images_series
            del matched_signals_extended
            #del images_pred

            if log:
                print("Saving matched signals for iteration {}".format(i))
                with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, matched_signals.astype(np.complex64))



            if movement_correction:
                traj = trajectory.get_traj()
                kdatai, traj_retained_final, _ = correct_mvt_kdata(kdatai, trajectory, cond_mvt,
                                                                   self.paramDict["ntimesteps"], density_adj=True)

            kdatai = np.array(kdatai)
            # nans = np.nonzero(np.isnan(kdatai))
            nans = [np.nonzero(np.isnan(k))[0] for k in kdatai]
            nans_count = np.array([len(n) for n in nans]).sum()

            if nans_count > 0:
                print("Warning : Nan Values replaced by zeros in rebuilt kdata")
                for i, k in enumerate(kdatai):
                    kdatai[i][nans[i]] = 0.0

            if not (movement_correction):
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape, useGPU=useGPU_simulation,
                                                               density_adj=True,ntimesteps=self.paramDict["ntimesteps"])

            else:
                trajectory.traj_for_reconstruction = traj_retained_final
                volumesi = simulate_radial_undersampled_images(kdatai, trajectory, mask.shape,
                                                               useGPU=useGPU_simulation, density_adj=True,
                                                               is_theta_z_adjusted=True,ntimesteps=self.paramDict["ntimesteps"])

            # volumesi/=(2*np.pi)

            nans_volumes = np.argwhere(np.isnan(volumesi))
            if len(nans_volumes) > 0:
                np.save('./log/kdatai.npy', kdatai)
                np.save('./log/volumesi.npy', volumesi)
                raise ValueError("Error : Nan Values in volumes")

            del kdatai

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)



            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            if log:
                print("Saving signalsi for iteration {}".format(i))
                with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(signalsi))

            del volumesi




            #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]);plt.plot(signalsi[:,j]);
            #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(matched_signals[:,j]);plt.plot(signals0[:,j] - signalsi[:,j]);
            #signals = matched_signals +  signals0 - signalsi
            #mu=0.01;signals = mu*matched_signals +  mu*signals0 - mu**2*signalsi
            #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]/np.linalg.norm(signals0[:,j]));plt.plot(matched_signals[:,j]/np.linalg.norm(matched_signals[:,j]));plt.plot(signals[:,j]/np.linalg.norm(signals[:,j]));

            grad = signalsi - signals0

            grad_extended = np.array([makevol(im, mask > 0) for im in grad])
            images_pred.images_series = np.repeat(grad_extended, nspoke, axis=0)
            del grad_extended
            kdata_grad = images_pred.generate_kdata(trajectory, useGPU=useGPU_simulation)

            del images_pred
            volumes_grad = simulate_radial_undersampled_images(kdata_grad, trajectory, mask.shape, useGPU=useGPU_simulation,
                                                           density_adj=True, ntimesteps=self.paramDict["ntimesteps"])

            mu_numerator=0
            mu_denom=0

            for ts in tqdm(range(volumes_grad.shape[0])):
                curr_grad=grad[ts]
                curr_volumes_grad=volumes_grad[ts][mask>0]
                mu_denom+=np.real(np.dot(curr_grad.conj(),curr_volumes_grad))
                mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

            mu = -mu_numerator/mu_denom
            print("Mu for iter {} : {}".format(i,mu))

            signals = matched_signals +mu* grad

            #norm_signals = np.linalg.norm(signals, axis=0)
            #all_signals_unthresholded = signals / norm_signals

            if threshold is not None:

                signals_for_map = signals * (1 - (np.abs(signals) > threshold))
                all_signals = signals_for_map/np.linalg.norm(signals_for_map,axis=0)

            else:

                all_signals=signals

        if log:
            print(date_time)

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)),matched_signals
        else:
            return dict(zip(keys_results, values_results))

    def search_patterns_test_multi_CSA(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        ntimesteps=self.paramDict["ntimesteps"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            trajectory_all_groups = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"]="Adaptative"

            if "dens_adj" in self.paramDict:
                dens_adj=self.paramDict["dens_adj"]
            else:
                dens_adj=True

            num_samples=trajectory_all_groups[0].get_traj().reshape(ntimesteps,-1,3).shape[1]

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase=self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim>2:
            signals_all_groups = [v[:, mask > 0] for v in volumes]




        if "kdata_init" in self.paramDict:
            kdata_init=self.paramDict["kdata_init"]
            nb_channels=kdata_init.shape[0]
            kdata_init=kdata_init.reshape(nb_channels,ntimesteps,-1)
            def J(m,kdata_init,dens_adj,trajectory):
                kdata=generate_kdata_multi(m,trajectory,self.paramDict["b1"],ntimesteps=self.paramDict["ntimesteps"])
                kdata_error=kdata-kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error=kdata_error.reshape(-1,trajectory.paramDict["npoint"])
                    density=np.expand_dims(density,axis=0)
                    kdata_error*=np.sqrt(density)

                return np.linalg.norm(kdata_error)**2

        ngroups=len(signals_all_groups)
        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

            mu0_all_groups=[]
            for i in range(ngroups):

                if self.paramDict["mu"] == "Adaptative":
                    trajectory=trajectory_all_groups[i]
                    signals=signals_all_groups[i]
                    kdata_all_signals = generate_kdata_multi(volumes[i], trajectory, self.paramDict["b1"],ntimesteps=ntimesteps)
                    mu_numerator = 0
                    mu_denom = 0

                    for ts in tqdm(range(kdata_all_signals.shape[1])):
                        curr_grad = signals[ts]
                        curr_volumes_grad = kdata_all_signals[:, ts].flatten()
                        if dens_adj:
                            density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                        else:
                            density=1
                        mu_denom += np.real(
                                np.dot(
                                    (density * curr_volumes_grad.reshape(-1, trajectory.paramDict["npoint"])).flatten().conj(),
                                    curr_volumes_grad))
                        mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                    del kdata_all_signals

                    mu0 = num_samples * mu_numerator / mu_denom
                    #mu0/=2




                else:
                    mu0 = self.paramDict["mu"]

                mu0_all_groups.append(mu0)

                signals_all_groups[i]=signals_all_groups[i]*mu0

            print("Mu0 : {}".format(mu0))
            if "mu_TV" in self.paramDict:
                mu_TV=self.paramDict["mu_TV"]
                grad_TV = grad_J_TV(volumes, 1, mask=mask) + grad_J_TV(volumes, 2, mask=mask)
                grad_TV=grad_TV[:,mask>0]
                grad_norm=np.linalg.norm(signals/mu0,axis=0)
                grad_TV_norm=np.linalg.norm(grad_TV,axis=0)
                signals -= mu0*mu_TV*grad_norm/grad_TV_norm*grad_TV


                #signals_corrected=signals-0.5*grad_norm/grad_TV_norm*grad_TV

                #ts=0
                #vol_no_TV=makevol(signals[ts],mask>0)
                #vol_TV = makevol(signals_corrected[ts],mask>0)
                #from utils_mrf import animate_multiple_images
                #animate_multiple_images(vol_no_TV,vol_TV)


                #del grad_TV
            signals0_all_groups = signals_all_groups

        del volumes

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        #norm_signals = np.linalg.norm(signals, 2, axis=0)
        #Normalize
        #all_signals = signals / norm_signals
        #all_signals=signals
        if type(dictfile)==str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        else:#otherwise dictfile contains (s_w,s_f,keys)
            array_water=dictfile[0]
            array_fat=dictfile[1]
            keys=dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        else:
            pca_water=None
            pca_fat=None
            transformed_array_water_unique=None
            transformed_array_fat_unique=None

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):


            all_signals=np.mean(signals_all_groups,axis=0)

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not(self.paramDict["return_matched_signals"])and(niter==0):
                map_rebuilt,J_optim,phase_optim=match_signals_v2(all_signals,keys, pca_water, pca_fat, array_water_unique, array_fat_unique,
                          transformed_array_water_unique, transformed_array_fat_unique, var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates, verbose,
                          niter, split, useGPU_dictsearch,mask,tv_denoising_weight,log_phase)
            else:
                map_rebuilt, J_optim, phase_optim,matched_signals = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                  array_water_unique, array_fat_unique,
                                                                  transformed_array_water_unique,
                                                                  transformed_array_fat_unique, var_w, var_f, sig_wf,
                                                                  pca, index_water_unique, index_fat_unique,
                                                                  remove_duplicates, verbose,
                                                                  niter, split, useGPU_dictsearch, mask,
                                                                  tv_denoising_weight, log_phase,return_matched_signals=True)



                if log:
                    print("Saving matched signals for iteration {}".format(i))
                    with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                        np.save(f, matched_signals.astype(np.complex64))

            #import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))

            if not(log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask,phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))

            if i == niter:
                break





            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes




            for j in range(ngroups):
                trajectory=trajectory_all_groups[j]

                map_rebuilt, J_optim, phase_optim, matched_signals = match_signals_v2(signals_all_groups[j], keys, pca_water,
                                                                                      pca_fat,
                                                                                      array_water_unique,
                                                                                      array_fat_unique,
                                                                                      transformed_array_water_unique,
                                                                                      transformed_array_fat_unique,
                                                                                      var_w, var_f, sig_wf,
                                                                                      pca, index_water_unique,
                                                                                      index_fat_unique,
                                                                                      remove_duplicates, verbose,
                                                                                      niter, split, useGPU_dictsearch,
                                                                                      mask,
                                                                                      tv_denoising_weight, log_phase,
                                                                                      return_matched_signals=True)

                matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])
                volumesi = undersampling_operator(matched_volumes, trajectory, self.paramDict["b1"],
                                                           density_adj=dens_adj)


                print("Correcting volumes for iteration {}".format(i))

                signalsi = volumesi[:, mask > 0]

                del volumesi




                #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]);plt.plot(signalsi[:,j]);
                #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(matched_signals[:,j]);plt.plot(signals0[:,j] - signalsi[:,j]);
                #signals = matched_signals +  signals0 - signalsi
                #mu=0.01;signals = mu*matched_signals +  mu*signals0 - mu**2*signalsi
                #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]/np.linalg.norm(signals0[:,j]));plt.plot(matched_signals[:,j]/np.linalg.norm(matched_signals[:,j]));plt.plot(signals[:,j]/np.linalg.norm(signals[:,j]));
                grad = signalsi - signals0_all_groups[j]/mu0_all_groups[j]


                #if "kdata_init" in self.paramDict:
                #    print("Debugging cost function")
                #    #volumes = np.array([makevol(im, mask > 0) for im in signals0])
                #    grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                #    mu_list=np.linspace(-1.6,-0,8)
                #    J_list=[J(matched_volumes +mu_* grad_volumes,kdata_init,True,trajectory) for mu_ in mu_list]
                #    import matplotlib.pyplot as plt
                #    plt.figure();plt.plot(mu_list,J_list)

                if self.paramDict["mu"]=="Adaptative":

                    grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                    kdata_grad = generate_kdata_multi(grad_volumes, trajectory, self.paramDict["b1"],ntimesteps=ntimesteps)

                    mu_numerator = 0
                    mu_denom = 0

                    for ts in tqdm(range(kdata_grad.shape[1])):
                        curr_grad = grad[ts]
                        curr_kdata_grad = kdata_grad[:, ts].flatten()
                        if dens_adj:
                            density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                        else:
                            density=1

                        mu_denom += np.real(
                            np.dot((density * curr_kdata_grad.reshape(-1, trajectory.paramDict["npoint"])).flatten().conj(),
                                   curr_kdata_grad))
                        mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                    mu = -num_samples * mu_numerator / mu_denom
                    #mu/=2
                elif self.paramDict["mu"]=="Brute":
                    grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                    mu_list = np.linspace(-1.6, -0, 8)
                    J_list = [J(matched_volumes +mu_* grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                    J_list = np.array(J_list)
                    mu = mu_list[np.argmin(J_list)]
                else:
                    mu=-self.paramDict["mu"]
                print("Mu for iter {} : {}".format(i,mu))

                # if "mu_TV" in self.paramDict:
                #    grad_TV = grad_J_TV(matched_volumes, 1, mask=mask) + grad_J_TV(matched_volumes, 2, mask=mask)
                #    grad_TV = grad_TV[:,mask>0]
                #    grad_norm = np.linalg.norm(grad, axis=0)
                #    grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                #    signals = matched_signals + mu * grad
                #
                #    signals += mu*mu_TV * grad_norm / grad_TV_norm * grad_TV
                #
                #     # mu_TV_list = np.linspace(0.22,0.45,4)
                #     # J_list=[]
                #     # for mu_TV in tqdm(mu_TV_list):
                #     #     signals_corrected = signals + mu_TV * mu * grad_norm / grad_TV_norm * grad_TV
                #     #     volumes_corrected=np.array([makevol(im, mask > 0) for im in signals_corrected])
                #     #     J_list.append(J(volumes_corrected, kdata_init, True, trajectory))
                #     # J_list = np.array(J_list)
                #     # import matplotlib.pyplot as plt
                #     # plt.figure();plt.plot(mu_TV_list,J_list)
                #
                #    # signals_corrected=signals+0.2*mu*grad_norm/grad_TV_norm*grad_TV
                #    #  #
                #    # ts=0
                #    # vol_no_TV=makevol(signals[ts],mask>0)
                #    # vol_TV = makevol(signals_corrected[ts],mask>0)
                #    # from utils_mrf import animate_multiple_images
                #    # animate_multiple_images(vol_no_TV,vol_TV)
                #    # import matplotlib.pyplot as plt
                #    # plt.figure()
                #    # plt.plot(vol_no_TV[6,128, :])
                #    # plt.plot(vol_TV[6,128,:])
                #
                # else:
                #     signals = matched_signals +mu* grad
                signals_all_groups[j] = matched_signals + mu * grad
                del grad



        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)),matched_signals
        else:
            return dict(zip(keys_results, values_results))

    def search_patterns_test_multi(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        ntimesteps = self.paramDict["ntimesteps"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            #            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"] = "Adaptative"

            if "dens_adj" in self.paramDict:
                dens_adj = self.paramDict["dens_adj"]
            else:
                dens_adj = True

            num_samples = trajectory.get_traj().reshape(ntimesteps, -1, 3).shape[1]

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase = self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]
        if pca and (type(dictfile)==str):
            pca_file = str.split(dictfile, ".dict")[0] + "_{}pca_simple.pkl".format(threshold_pca)
            pca_file_name = str.split(pca_file, "/")[-1]

        if type(dictfile)==str:
            vars_file = str.split(dictfile, ".dict")[0] + "_vars_simple.pkl".format(threshold_pca)
            vars_file_name = str.split(vars_file, "/")[-1]
            path = str.split(os.path.realpath(__file__), "/dictoptimizers.py")[0]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim > 2:
            all_signals = volumes[:, mask > 0]
        else:  # already masked
            all_signals = volumes

        all_signals=all_signals.astype("complex64")

        # if log:
        #     now = datetime.now()
        #     date_time = now.strftime("%Y%m%d_%H%M%S")
        #     with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
        #         np.save(f, all_signals)

        if "kdata_init" in self.paramDict:
            kdata_init = self.paramDict["kdata_init"]
            nb_channels = kdata_init.shape[0]
            kdata_init = kdata_init.reshape(nb_channels, ntimesteps, -1)

            def J(m, kdata_init, dens_adj, trajectory):
                kdata = generate_kdata_multi(m, trajectory, self.paramDict["b1"],
                                             ntimesteps=self.paramDict["ntimesteps"])
                kdata_error = kdata - kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
                    density = np.expand_dims(density, axis=0)
                    kdata_error *= np.sqrt(density)

                return np.linalg.norm(kdata_error) ** 2

        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            if self.paramDict["mu"] == "Adaptative":

                kdata_all_signals = generate_kdata_multi(volumes, trajectory, self.paramDict["b1"],
                                                         ntimesteps=ntimesteps)
                # L0=volumes.shape[0];test_volumes_all_signals = volumes_all_signals.reshape(L0,-1);test_volumes = volumes.reshape(L0,-1);
                # import matplotlib.pyplot as plt; j = np.random.choice(test_volumes.shape[-1]);plt.plot(test_volumes[:,j]);plt.plot(test_volumes_all_signals[:,j]);
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_all_signals.shape[1])):
                    curr_grad = all_signals[ts]
                    curr_volumes_grad = kdata_all_signals[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1
                    mu_denom += np.real(
                        np.dot(
                            (density * curr_volumes_grad.reshape(-1,
                                                                 trajectory.paramDict["npoint"])).flatten().conj(),
                            curr_volumes_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                del kdata_all_signals

                mu0 = num_samples * mu_numerator / mu_denom
                # mu0/=2

            elif self.paramDict["mu"] == "Brute":
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(- mu_ * volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu0 = -mu_list[np.argmin(J_list)]





            else:
                mu0 = self.paramDict["mu"]




            all_signals *= mu0
            print("Mu0 : {}".format(mu0))
            signals0 = copy(all_signals)/mu0

            if "mu_TV" in self.paramDict:
                mu_TV = self.paramDict["mu_TV"]

                if "weights_TV" not in self.paramDict:
                    self.paramDict["weights_TV"]=[1.,0.,0.]
                weights_TV=np.array(self.paramDict["weights_TV"])
                weights_TV/=np.sum(weights_TV)

        del volumes

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        # norm_signals = np.linalg.norm(signals, 2, axis=0)
        # Normalize
        # all_signals = signals / norm_signals
        #all_signals = signals
        if type(dictfile) == str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        else:  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dictfile[0]
            array_fat = dictfile[1]
            keys = dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        if not(type(dictfile)==str)or(vars_file_name not in os.listdir(path)) or ((pca) and (pca_file_name not in os.listdir(path))):

            print("Calculating unique dico signals")
            array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
            array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)



        #nb_water_timesteps = array_water_unique.shape[1]
        #nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            if not(type(dictfile)==str) or(pca_file_name not in os.listdir(path)):
                pca_water = PCAComplex(n_components_=threshold_pca)
                pca_fat = PCAComplex(n_components_=threshold_pca)

                pca_water.fit(array_water_unique)
                pca_fat.fit(array_fat_unique)



                transformed_array_water_unique = pca_water.transform(array_water_unique)
                transformed_array_fat_unique = pca_fat.transform(array_fat_unique)
                if type(dictfile) == str:
                    with open(pca_file,"wb") as file:
                        pickle.dump((pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique),file)
            else:
                print("Loading pca")
                with open(pca_file, "rb") as file:
                    (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)=pickle.load(file)

        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        if not(type(dictfile)==str) or (vars_file_name not in os.listdir(path)):
            var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
            var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
            sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                            axis=1).real

            var_w = var_w[index_water_unique]
            var_f = var_f[index_fat_unique]

            var_w = np.reshape(var_w, (-1, 1))
            var_f = np.reshape(var_f, (-1, 1))
            sig_wf = np.reshape(sig_wf, (-1, 1))
            if type(dictfile) == str:
                with open(vars_file,"wb") as file:
                    pickle.dump((var_w,var_f,sig_wf,index_water_unique,index_fat_unique),file)
        else:
            print("Loading var w / var f / sig wf")
            with open(vars_file, "rb") as file:
                (var_w, var_f, sig_wf,index_water_unique,index_fat_unique)=pickle.load(file)

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):

            # if log:
            #     print("Saving signals for iteration {}".format(i))
            #     with open('./log/signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
            #         np.save(f, np.array(all_signals))

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not (self.paramDict["return_matched_signals"]) and (niter == 0):
                map_rebuilt, J_optim, phase_optim = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                     array_water_unique, array_fat_unique,
                                                                     transformed_array_water_unique,
                                                                     transformed_array_fat_unique, var_w, var_f,
                                                                     sig_wf, pca, index_water_unique,
                                                                     index_fat_unique, remove_duplicates, verbose,
                                                                     niter, split, useGPU_dictsearch, mask,
                                                                     tv_denoising_weight, log_phase)
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
                                                                                      remove_duplicates, verbose,
                                                                                      niter, split,
                                                                                      useGPU_dictsearch, mask,
                                                                                      tv_denoising_weight,
                                                                                      log_phase,
                                                                                      return_matched_signals=True)

                # if log:
                #     print("Saving matched signals for iteration {}".format(i))
                #     with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                #         np.save(f, matched_signals.astype(np.complex64))

            # import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))

            if not (log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask, phase_optim))

            # if log:
            #     with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
            #         pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if i == niter:
                break

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))



            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            # keys_simu = list(map_rebuilt.keys())
            # values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            # map_for_sim = dict(zip(keys_simu, values_simu))

            # images_pred.build_ref_images(seq, norm=J_optim * norm_signals, phase=phase_optim)
            # matched_signals_extended=np.repeat(matched_signals,nspoke,axis=0)
            matched_signals=matched_signals.astype("complex64")
            matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])

            volumesi = undersampling_operator(matched_volumes, trajectory, self.paramDict["b1"],
                                              density_adj=dens_adj,light_memory_usage=True)

            del matched_volumes

            # del images_pred

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            # if log:
            #     print("Saving signalsi for iteration {}".format(i))
            #     with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
            #         np.save(f, np.array(signalsi))

            del volumesi

            # j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]);plt.plot(signalsi[:,j]);
            # j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(matched_signals[:,j]);plt.plot(signals0[:,j] - signalsi[:,j]);
            # signals = matched_signals +  signals0 - signalsi
            # mu=0.01;signals = mu*matched_signals +  mu*signals0 - mu**2*signalsi
            # j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]/np.linalg.norm(signals0[:,j]));plt.plot(matched_signals[:,j]/np.linalg.norm(matched_signals[:,j]));plt.plot(signals[:,j]/np.linalg.norm(signals[:,j]));
            grad = signalsi - signals0
            del signalsi


            # if "kdata_init" in self.paramDict:
            #    print("Debugging cost function")
            #    #volumes = np.array([makevol(im, mask > 0) for im in signals0])
            #    grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
            #    mu_list=np.linspace(-1.6,-0,16)
            #    J_list=[J(matched_volumes +mu_* grad_volumes,kdata_init,True,trajectory) for mu_ in mu_list]
            #    import matplotlib.pyplot as plt
            #    plt.figure();plt.plot(mu_list,J_list)
            #     plt.axvline(x=mu,color="red")

            if self.paramDict["mu"] == "Adaptative":

                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                kdata_grad = generate_kdata_multi(grad_volumes, trajectory, self.paramDict["b1"],
                                                  ntimesteps=ntimesteps)

                del grad_volumes
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_grad.shape[1])):
                    curr_grad = grad[ts]
                    curr_kdata_grad = kdata_grad[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1

                    mu_denom += np.real(
                        np.dot((density * curr_kdata_grad.reshape(-1,
                                                                  trajectory.paramDict["npoint"])).flatten().conj(),
                               curr_kdata_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))
                del kdata_grad
                mu = -num_samples * mu_numerator / mu_denom
                # mu/=2
            elif self.paramDict["mu"] == "Brute":
                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(matched_volumes + mu_ * grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu = mu_list[np.argmin(J_list)]
            else:
                mu = -self.paramDict["mu"]
            print("Mu for iter {} : {}".format(i, mu))

            all_signals = matched_signals + mu * grad

            if "mu_TV" not in self.paramDict:
                del grad


            if "mu_TV" in self.paramDict:
                print("Applying TV regularization")
                # matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])
                #
                # grad_TV=0
                # for ind_w,w in tqdm(enumerate(weights_TV)):
                #     if w>0:
                #         grad_TV+=w*grad_J_TV(matched_volumes, ind_w, mask=mask)
                #
                #
                # del matched_volumes
                # grad_TV = grad_TV[:, mask > 0]
                # grad_norm = np.linalg.norm(grad, axis=0)
                # grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                # #signals = matched_signals + mu * grad
                #
                # signals += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                # del grad_TV
                # del grad_norm
                # del grad_TV_norm

                grad_norm = np.linalg.norm(grad, axis=0)
                del grad

                grad_TV=np.zeros(matched_signals.shape,dtype=matched_signals.dtype)
                for ts in tqdm(range(ntimesteps)):
                    matched_volumes_ts =makevol(matched_signals[ts], mask > 0)
                    for ind_w, w in (enumerate(weights_TV)):
                        if w > 0:
                            grad_TV[ts] += (w * grad_J_TV(matched_volumes_ts, ind_w, mask=mask,shift=0))[mask>0]


                grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                # signals = matched_signals + mu * grad

                all_signals += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm



                # mu_TV_list = np.linspace(0.22,0.45,4)
                # J_list=[]
                # for mu_TV in tqdm(mu_TV_list):
                #     signals_corrected = signals + mu_TV * mu * grad_norm / grad_TV_norm * grad_TV
                #     volumes_corrected=np.array([makevol(im, mask > 0) for im in signals_corrected])
                #     J_list.append(J(volumes_corrected, kdata_init, True, trajectory))
                # J_list = np.array(J_list)
                # import matplotlib.pyplot as plt
                # plt.figure();plt.plot(mu_TV_list,J_list)

               # signals_corrected=signals+0.2*mu*grad_norm/grad_TV_norm*grad_TV
               #  #
               # ts=0
               # vol_no_TV=makevol(signals[ts],mask>0)
               # vol_TV = makevol(signals_corrected[ts],mask>0)
               # from utils_mrf import animate_multiple_images
               # animate_multiple_images(vol_no_TV,vol_TV)
               # import matplotlib.pyplot as plt
               # plt.figure()
               # plt.plot(vol_no_TV[6,128, :])
               # plt.plot(vol_TV[6,128,:])
            #
            # else:
            #     signals = matched_signals +mu* grad
            del matched_signals


            # norm_signals = np.linalg.norm(signals, axis=0)
            # all_signals_unthresholded = signals / norm_signals



        

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))



    def search_patterns_test_multi_grouping(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        ntimesteps = self.paramDict["ntimesteps"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            #            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"] = "Adaptative"

            if "dens_adj" in self.paramDict:
                dens_adj = self.paramDict["dens_adj"]
            else:
                dens_adj = True

            num_samples = trajectory.get_traj().reshape(ntimesteps, -1, 3).shape[1]

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase = self.paramDict["log_phase"]

        n_clusters_dico=self.paramDict["n_clusters_dico"]
        n_clusters_signals=self.paramDict["n_clusters_signals"]
        pruning = self.paramDict["pruning"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim > 2:
            all_signals = volumes[:, mask > 0]
        else:  # already masked
            all_signals = volumes

        all_signals=all_signals.astype("complex64")

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, all_signals)

        if "kdata_init" in self.paramDict:
            kdata_init = self.paramDict["kdata_init"]
            nb_channels = kdata_init.shape[0]
            kdata_init = kdata_init.reshape(nb_channels, ntimesteps, -1)

            def J(m, kdata_init, dens_adj, trajectory):
                kdata = generate_kdata_multi(m, trajectory, self.paramDict["b1"],
                                             ntimesteps=self.paramDict["ntimesteps"])
                kdata_error = kdata - kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
                    density = np.expand_dims(density, axis=0)
                    kdata_error *= np.sqrt(density)

                return np.linalg.norm(kdata_error) ** 2

        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

            if self.paramDict["mu"] == "Adaptative":

                kdata_all_signals = generate_kdata_multi(volumes, trajectory, self.paramDict["b1"],
                                                         ntimesteps=ntimesteps)
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_all_signals.shape[1])):
                    curr_grad = all_signals[ts]
                    curr_volumes_grad = kdata_all_signals[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1
                    mu_denom += np.real(
                        np.dot(
                            (density * curr_volumes_grad.reshape(-1,
                                                                 trajectory.paramDict["npoint"])).flatten().conj(),
                            curr_volumes_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                del kdata_all_signals

                mu0 = num_samples * mu_numerator / mu_denom

            elif self.paramDict["mu"] == "Brute":
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(- mu_ * volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu0 = -mu_list[np.argmin(J_list)]

            else:
                mu0 = self.paramDict["mu"]

            all_signals *= mu0
            print("Mu0 : {}".format(mu0))
            signals0 = copy(all_signals)/mu0

            if "mu_TV" in self.paramDict:
                mu_TV = self.paramDict["mu_TV"]

                if "weights_TV" not in self.paramDict:
                    self.paramDict["weights_TV"]=[1.,0.,0.]
                weights_TV=np.array(self.paramDict["weights_TV"])
                weights_TV/=np.sum(weights_TV)

        del volumes

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        # norm_signals = np.linalg.norm(signals, 2, axis=0)
        # Normalize
        # all_signals = signals / norm_signals
        #all_signals = signals
        if type(dictfile) == str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        else:  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dictfile[0]
            array_fat = dictfile[1]
            keys = dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_all = np.concatenate([np.real(array_water), np.imag(array_water), np.real(array_fat), np.imag(array_fat)],
                                   axis=-1)

        model_dico = KMeans(n_clusters=n_clusters_dico)
        model_dico.fit(array_all)

        array_water_unique_clustering = model_dico.cluster_centers_[:, :ntimesteps] + 1j * model_dico.cluster_centers_[:,
                                                                           (ntimesteps):(2 * ntimesteps)]
        array_fat_unique_clustering = model_dico.cluster_centers_[:, (2 * ntimesteps):(3 * ntimesteps)] + 1j * model_dico.cluster_centers_[:,
                                                                                               (3 * ntimesteps):(
                                                                                                           4 * ntimesteps)]

        del array_all

        nb_water_timesteps = array_water_unique_clustering.shape[1]
        nb_fat_timesteps = array_fat_unique_clustering.shape[1]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        var_w_total = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f_total = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf_total = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                              axis=1).real
        var_w_total = var_w_total[index_water_unique]
        var_f_total = var_f_total[index_fat_unique]
        var_w_total = np.reshape(var_w_total, (-1, 1))
        var_f_total = np.reshape(var_f_total, (-1, 1))
        sig_wf_total = np.reshape(sig_wf_total, (-1, 1))

        #del array_water
        #del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique_clustering)
            pca_fat.fit(array_fat_unique_clustering)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_,
                                                                          nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique_clustering)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique_clustering)

        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        var_w = np.sum(array_water_unique_clustering * array_water_unique_clustering.conj(), axis=1).real
        var_f = np.sum(array_fat_unique_clustering * array_fat_unique_clustering.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique_clustering* array_fat_unique_clustering.conj(),
                        axis=1).real

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        model_signals = KMeans(n_clusters=n_clusters_signals)
        model_signals.fit(np.concatenate([np.real(all_signals), np.imag(all_signals)]).T)

        for i in range(niter + 1):

            idx_clusters=match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                     array_water_unique_clustering, array_fat_unique_clustering,
                                                                     transformed_array_water_unique,
                                                                     transformed_array_fat_unique, var_w, var_f,
                                                                     sig_wf, pca, None,
                                                                     None, remove_duplicates, verbose,
                                                                     0, split, useGPU_dictsearch, mask,
                                                                     tv_denoising_weight, False,n_clusters_dico=n_clusters_dico,pruning=pruning)



            if log:
                print("Saving signals for iteration {}".format(i))
                with open('./log/signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(all_signals))

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not (self.paramDict["return_matched_signals"]) and (niter == 0):
                map_rebuilt, J_optim, phase_optim = match_signals_v2_on_clusters(all_signals, keys, threshold_pca,
                                                                     array_water, array_fat,
                                                                      var_w_total, var_f_total,
                                                                     sig_wf_total, verbose,niter,
                                                                     useGPU_dictsearch,idx_clusters,model_dico,model_signals,
                                                                     log_phase)
            else:
                map_rebuilt, J_optim, phase_optim, matched_signals = match_signals_v2_on_clusters(all_signals, keys, threshold_pca,
                                                                     array_water, array_fat,
                                                                      var_w_total, var_f_total,
                                                                     sig_wf_total, verbose,niter,
                                                                     useGPU_dictsearch,idx_clusters,model_dico,model_signals,
                                                                     log_phase,return_matched_signals=True)


                if log:
                    print("Saving matched signals for iteration {}".format(i))
                    with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                        np.save(f, matched_signals.astype(np.complex64))

            # import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))

            if not (log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask, phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if i == niter:
                break

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))



            print("Generating prediction volumes and undersampled images for iteration {}".format(i))

            matched_signals=matched_signals.astype("complex64")
            matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])

            volumesi = undersampling_operator(matched_volumes, trajectory, self.paramDict["b1"],
                                              density_adj=dens_adj,light_memory_usage=True)

            del matched_volumes

            # del images_pred

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            if log:
                print("Saving signalsi for iteration {}".format(i))
                with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(signalsi))

            del volumesi

            grad = signalsi - signals0
            del signalsi

            if self.paramDict["mu"] == "Adaptative":

                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                kdata_grad = generate_kdata_multi(grad_volumes, trajectory, self.paramDict["b1"],
                                                  ntimesteps=ntimesteps)

                del grad_volumes
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_grad.shape[1])):
                    curr_grad = grad[ts]
                    curr_kdata_grad = kdata_grad[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1

                    mu_denom += np.real(
                        np.dot((density * curr_kdata_grad.reshape(-1,
                                                                  trajectory.paramDict["npoint"])).flatten().conj(),
                               curr_kdata_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))
                del kdata_grad
                mu = -num_samples * mu_numerator / mu_denom
                # mu/=2
            elif self.paramDict["mu"] == "Brute":
                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(matched_volumes + mu_ * grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu = mu_list[np.argmin(J_list)]
            else:
                mu = -self.paramDict["mu"]
            print("Mu for iter {} : {}".format(i, mu))

            all_signals = matched_signals + mu * grad

            if "mu_TV" not in self.paramDict:
                del grad


            if "mu_TV" in self.paramDict:
                print("Applying TV regularization")
                grad_norm = np.linalg.norm(grad, axis=0)
                del grad

                grad_TV=np.zeros(matched_signals.shape,dtype=matched_signals.dtype)
                for ts in tqdm(range(ntimesteps)):
                    matched_volumes_ts =makevol(matched_signals[ts], mask > 0)
                    for ind_w, w in (enumerate(weights_TV)):
                        if w > 0:
                            grad_TV[ts] += (w * grad_J_TV(matched_volumes_ts, ind_w, mask=mask,shift=0))[mask>0]


                grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                # signals = matched_signals + mu * grad

                all_signals += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

            del matched_signals


            # norm_signals = np.linalg.norm(signals, axis=0)
            # all_signals_unthresholded = signals / norm_signals



        if log:
            print(date_time)

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))

    def search_patterns_test_multi_mvt(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        ntimesteps = self.paramDict["ntimesteps"]

        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            #            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"] = "Adaptative"

            if "dens_adj" in self.paramDict:
                dens_adj = self.paramDict["dens_adj"]
            else:
                dens_adj = True
            weights = self.paramDict["weights"]

            num_samples = trajectory.get_traj().reshape(ntimesteps, -1, 3).shape[1]

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase = self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim > 2:
            signals = volumes[:, mask > 0]
        else:  # already masked
            signals = volumes

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, signals)

        if "kdata_init" in self.paramDict:
            kdata_init = self.paramDict["kdata_init"]
            nb_channels = kdata_init.shape[0]
            kdata_init = kdata_init.reshape(nb_channels, ntimesteps, -1)

            def J(m, kdata_init, dens_adj, trajectory):
                kdata = generate_kdata_multi(m, trajectory, self.paramDict["b1"],
                                             ntimesteps=self.paramDict["ntimesteps"])
                kdata_error = kdata - kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
                    density = np.expand_dims(density, axis=0)
                    kdata_error *= np.sqrt(density)

                return np.linalg.norm(kdata_error) ** 2

        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])
            nb_channels=self.paramDict["b1"].shape[0]

            if self.paramDict["mu"] == "Adaptative":

                kdata_all_signals = generate_kdata_multi_new(volumes, trajectory, self.paramDict["b1"],
                                                         ntimesteps=ntimesteps,retained_timesteps=retained_timesteps)
                # L0=volumes.shape[0];test_volumes_all_signals = volumes_all_signals.reshape(L0,-1);test_volumes = volumes.reshape(L0,-1);
                # import matplotlib.pyplot as plt; j = np.random.choice(test_volumes.shape[-1]);plt.plot(test_volumes[:,j]);plt.plot(test_volumes_all_signals[:,j]);
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_all_signals.shape[1])):
                    curr_grad = signals[ts]
                    curr_volumes_grad = kdata_all_signals[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1

                    curr_volumes_grad_adjusted = (density * curr_volumes_grad.reshape(-1,
                                                                 trajectory.paramDict["npoint"])).flatten()
                    curr_volumes_grad_adjusted = curr_volumes_grad_adjusted.reshape(
                        (nb_channels,) + weights.shape[1:] + (-1,))
                    curr_weights = np.expand_dims(weights[ts], axis=(0, -1))
                    curr_volumes_grad_adjusted *= curr_weights
                    curr_volumes_grad_adjusted = curr_volumes_grad_adjusted.flatten()

                    mu_denom += np.real(
                        np.dot(
                            curr_volumes_grad_adjusted.conj(),
                            curr_volumes_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                del kdata_all_signals

                mu0 = num_samples * mu_numerator / mu_denom
                # mu0/=2

            elif self.paramDict["mu"] == "Brute":
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(- mu_ * volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu0 = -mu_list[np.argmin(J_list)]





            else:
                mu0 = self.paramDict["mu"]

            signals *= mu0

            signals0 = copy(signals)
            print("Mu0 : {}".format(mu0))
            if "mu_TV" in self.paramDict:
                mu_TV = self.paramDict["mu_TV"]

                if "weights_TV" not in self.paramDict:
                    self.paramDict["weights_TV"]=[1.,0.,0.]
                weights_TV=np.array(self.paramDict["weights_TV"])
                weights_TV/=np.sum(weights_TV)
            #    print("Applying TV regularization")
            #    mu_TV = self.paramDict["mu_TV"]
            #    grad_TV = grad_J_TV(volumes, 0, mask=mask)#grad_J_TV(volumes, 1, mask=mask) + grad_J_TV(volumes, 2, mask=mask)
            #    grad_TV = grad_TV[:, mask > 0]
            #    grad_norm = np.linalg.norm(signals / mu0, axis=0)
            #    grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
            #    signals -= mu0 * mu_TV * grad_norm / grad_TV_norm * grad_TV

                # signals_corrected=signals-0.5*grad_norm/grad_TV_norm*grad_TV

                # ts=0
                # vol_no_TV=makevol(signals[ts],mask>0)
                # vol_TV = makevol(signals_corrected[ts],mask>0)
                # from utils_mrf import animate_multiple_images
                # animate_multiple_images(vol_no_TV,vol_TV)

                # del grad_TV


        del volumes

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        # norm_signals = np.linalg.norm(signals, 2, axis=0)
        # Normalize
        # all_signals = signals / norm_signals
        all_signals = signals
        if type(dictfile) == str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        else:  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dictfile[0]
            array_fat = dictfile[1]
            keys = dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_,
                                                                          nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                        axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):

            if log:
                print("Saving signals for iteration {}".format(i))
                with open('./log/signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(all_signals))

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not (self.paramDict["return_matched_signals"]) and (niter == 0):
                map_rebuilt, J_optim, phase_optim = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                     array_water_unique, array_fat_unique,
                                                                     transformed_array_water_unique,
                                                                     transformed_array_fat_unique, var_w, var_f,
                                                                     sig_wf, pca, index_water_unique,
                                                                     index_fat_unique, remove_duplicates, verbose,
                                                                     niter, split, useGPU_dictsearch, mask,
                                                                     tv_denoising_weight, log_phase)
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
                                                                                      remove_duplicates, verbose,
                                                                                      niter, split,
                                                                                      useGPU_dictsearch, mask,
                                                                                      tv_denoising_weight,
                                                                                      log_phase,
                                                                                      return_matched_signals=True)

                if log:
                    print("Saving matched signals for iteration {}".format(i))
                    with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                        np.save(f, matched_signals.astype(np.complex64))

            # import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))

            if not (log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask, phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if i == niter:
                break

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))



            print("Generating prediction volumes and undersampled images for iteration {}".format(i))

            matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])

            volumesi = undersampling_operator_new(matched_volumes, trajectory, self.paramDict["b1"],
                                              density_adj=dens_adj,ntimesteps=ntimesteps,retained_timesteps=retained_timesteps,weights=weights)

            # del images_pred

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            if log:
                print("Saving signalsi for iteration {}".format(i))
                with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(signalsi))

            del volumesi

            # j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]);plt.plot(signalsi[:,j]);
            # j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(matched_signals[:,j]);plt.plot(signals0[:,j] - signalsi[:,j]);
            # signals = matched_signals +  signals0 - signalsi
            # mu=0.01;signals = mu*matched_signals +  mu*signals0 - mu**2*signalsi
            # j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]/np.linalg.norm(signals0[:,j]));plt.plot(matched_signals[:,j]/np.linalg.norm(matched_signals[:,j]));plt.plot(signals[:,j]/np.linalg.norm(signals[:,j]));
            grad = signalsi - signals0 / mu0

            # if "kdata_init" in self.paramDict:
            #    print("Debugging cost function")
            #    #volumes = np.array([makevol(im, mask > 0) for im in signals0])
            #    grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
            #    mu_list=np.linspace(-1.6,-0,8)
            #    J_list=[J(matched_volumes +mu_* grad_volumes,kdata_init,True,trajectory) for mu_ in mu_list]
            #    import matplotlib.pyplot as plt
            #    plt.figure();plt.plot(mu_list,J_list)

            if self.paramDict["mu"] == "Adaptative":

                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                kdata_grad = generate_kdata_multi_new(grad_volumes, trajectory, self.paramDict["b1"],
                                                  ntimesteps=ntimesteps,retained_timesteps=retained_timesteps)

                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_grad.shape[1])):
                    curr_grad = grad[ts]
                    curr_kdata_grad = kdata_grad[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1

                    curr_kdata_grad_adjusted=(density * curr_kdata_grad.reshape(-1,trajectory.paramDict["npoint"])).flatten()
                    curr_kdata_grad_adjusted = curr_kdata_grad_adjusted.reshape((nb_channels,)+weights.shape[1:] + (-1,))
                    curr_weights = np.expand_dims(weights[ts], axis=(0,-1))
                    curr_kdata_grad_adjusted *= curr_weights
                    curr_kdata_grad_adjusted=curr_kdata_grad_adjusted.flatten()

                    mu_denom += np.real(np.dot(curr_kdata_grad_adjusted.conj(),curr_kdata_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                mu = -num_samples * mu_numerator / mu_denom
                # mu/=2
            elif self.paramDict["mu"] == "Brute":
                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(matched_volumes + mu_ * grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu = mu_list[np.argmin(J_list)]
            else:
                mu = -self.paramDict["mu"]
            print("Mu for iter {} : {}".format(i, mu))


            #
            #     # mu_TV_list = np.linspace(0.22,0.45,4)
            #     # J_list=[]
            #     # for mu_TV in tqdm(mu_TV_list):
            #     #     signals_corrected = signals + mu_TV * mu * grad_norm / grad_TV_norm * grad_TV
            #     #     volumes_corrected=np.array([makevol(im, mask > 0) for im in signals_corrected])
            #     #     J_list.append(J(volumes_corrected, kdata_init, True, trajectory))
            #     # J_list = np.array(J_list)
            #     # import matplotlib.pyplot as plt
            #     # plt.figure();plt.plot(mu_TV_list,J_list)
            #
            #    # signals_corrected=signals+0.2*mu*grad_norm/grad_TV_norm*grad_TV
            #    #  #
            #    # ts=0
            #    # vol_no_TV=makevol(signals[ts],mask>0)
            #    # vol_TV = makevol(signals_corrected[ts],mask>0)
            #    # from utils_mrf import animate_multiple_images
            #    # animate_multiple_images(vol_no_TV,vol_TV)
            #    # import matplotlib.pyplot as plt
            #    # plt.figure()
            #    # plt.plot(vol_no_TV[6,128, :])
            #    # plt.plot(vol_TV[6,128,:])
            #
            # else:
            #     signals = matched_signals +mu* grad
            signals = matched_signals + mu * grad

            if "mu_TV" in self.paramDict:
                print("Applying TV regularization")
                grad_TV = weights_TV[0]*grad_J_TV(matched_volumes, 0, mask=mask)+weights_TV[1]*grad_J_TV(matched_volumes, 1, mask=mask) + weights_TV[2]*grad_J_TV(matched_volumes, 2, mask=mask)
                grad_TV = grad_TV[:,mask>0]
                grad_norm = np.linalg.norm(grad, axis=0)
                grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                #signals = matched_signals + mu * grad

                signals += mu*mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV

            del grad
            # norm_signals = np.linalg.norm(signals, axis=0)
            # all_signals_unthresholded = signals / norm_signals

            if threshold is not None:

                signals_for_map = signals * (1 - (np.abs(signals) > threshold))
                all_signals = signals_for_map / np.linalg.norm(signals_for_map, axis=0)

            else:

                all_signals = signals

        if log:
            print(date_time)

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))

    def search_patterns_test_multi_2_steps_dico_mvt(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        threshold_ff=self.paramDict["threshold_ff"]
        dictfile_light=self.paramDict["dictfile_light"]
        ntimesteps = self.paramDict["ntimesteps"]


        if "return_cost" not in self.paramDict:
            self.paramDict["return_cost"]=False
        return_cost = self.paramDict["return_cost"]

        if "calculate_matched_signals" not in self.paramDict:
            self.paramDict["calculate_matched_signals"]=False
        calculate_matched_signals = self.paramDict["calculate_matched_signals"]

        if "return_matched_signals" not in self.paramDict:
            self.paramDict["return_matched_signals"]=False
        return_matched_signals = self.paramDict["return_matched_signals"]

        if niter>0 or return_matched_signals:
            calculate_matched_signals=True

        if calculate_matched_signals:
            return_cost=True

        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            #            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"] = "Adaptative"

            if "dens_adj" in self.paramDict:
                dens_adj = self.paramDict["dens_adj"]
            else:
                dens_adj = True
            weights = self.paramDict["weights"]

            num_samples = trajectory.get_traj().reshape(ntimesteps, -1, 3).shape[1]

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        if pca:
            pca_file = str.split(dictfile, ".dict")[0] + "_{}pca.pkl".format(threshold_pca)
            pca_file_name = str.split(pca_file, "/")[-1]

        vars_file = str.split(dictfile, ".dict")[0] + "_vars.pkl".format(threshold_pca)
        vars_file_name=str.split(vars_file,"/")[-1]
        path=str.split(os.path.realpath(__file__),"/dictoptimizers.py")[0]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim > 2:
            all_signals = volumes[:, mask > 0]
        else:  # already masked
            all_signals = volumes

        all_signals=all_signals.astype("complex64")
        nb_signals=all_signals.shape[1]

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, all_signals)

        if "kdata_init" in self.paramDict:
            kdata_init = self.paramDict["kdata_init"]
            nb_channels = kdata_init.shape[0]
            kdata_init = kdata_init.reshape(nb_channels, ntimesteps, -1)

            def J(m, kdata_init, dens_adj, trajectory):
                kdata = generate_kdata_multi(m, trajectory, self.paramDict["b1"],
                                             ntimesteps=self.paramDict["ntimesteps"])
                kdata_error = kdata - kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
                    density = np.expand_dims(density, axis=0)
                    kdata_error *= np.sqrt(density)

                return np.linalg.norm(kdata_error) ** 2

        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

            if self.paramDict["mu"] == "Adaptative":

                kdata_all_signals = generate_kdata_multi(volumes, trajectory, self.paramDict["b1"],
                                                         ntimesteps=ntimesteps)
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_all_signals.shape[1])):
                    curr_grad = all_signals[ts]
                    curr_volumes_grad = kdata_all_signals[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1
                    curr_volumes_grad_adjusted = (density * curr_volumes_grad.reshape(-1,
                                                                 trajectory.paramDict["npoint"])).flatten()
                    curr_volumes_grad_adjusted = curr_volumes_grad_adjusted.reshape(
                        (nb_channels,) + weights.shape[1:] + (-1,))
                    curr_weights = np.expand_dims(weights[ts], axis=(0, -1))
                    curr_volumes_grad_adjusted *= curr_weights
                    curr_volumes_grad_adjusted = curr_volumes_grad_adjusted.flatten()
                    mu_denom += np.real(
                        np.dot(
                            curr_volumes_grad_adjusted.conj(),
                            curr_volumes_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                del kdata_all_signals

                mu0 = num_samples * mu_numerator / mu_denom

            elif self.paramDict["mu"] == "Brute":
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(- mu_ * volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu0 = -mu_list[np.argmin(J_list)]

            else:
                mu0 = self.paramDict["mu"]

            all_signals *= mu0
            print("Mu0 : {}".format(mu0))
            signals0 = copy(all_signals)/mu0

            if "mu_TV" in self.paramDict:
                mu_TV = self.paramDict["mu_TV"]

                if "weights_TV" not in self.paramDict:
                    self.paramDict["weights_TV"]=[1.,0.,0.]
                weights_TV=np.array(self.paramDict["weights_TV"])
                weights_TV/=np.sum(weights_TV)

        del volumes

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        # norm_signals = np.linalg.norm(signals, 2, axis=0)
        # Normalize
        # all_signals = signals / norm_signals
        #all_signals = signals
        if type(dictfile) == str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]
            keys=np.array(keys)

            del mrfdict
        else:  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dictfile[0]
            array_fat = dictfile[1]
            keys = dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        print(path)
        if (vars_file_name not in os.listdir(path)) or ((pca) and (pca_file_name not in os.listdir(path))) or (calculate_matched_signals):

            print("Calculating unique dico signals")
            array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
            array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water.shape[1]
        nb_fat_timesteps = array_fat.shape[1]

        if vars_file_name not in os.listdir(path):

            var_w_total = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
            var_f_total = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
            sig_wf_total = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                                  axis=1).real
            var_w_total = var_w_total[index_water_unique]
            var_f_total = var_f_total[index_fat_unique]
            var_w_total = np.reshape(var_w_total, (-1, 1))
            var_f_total = np.reshape(var_f_total, (-1, 1))
            sig_wf_total = np.reshape(sig_wf_total, (-1, 1))
            with open(vars_file,"wb") as file:
                pickle.dump((var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique),file)

        else:
            print("Loading var w / var f / sig wf")
            with open(vars_file, "rb") as file:
                (var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique)=pickle.load(file)

        #del array_water
        #del array_fat

        if pca:
            if pca_file_name not in os.listdir(path):
                pca_water = PCAComplex(n_components_=threshold_pca)
                pca_fat = PCAComplex(n_components_=threshold_pca)

                pca_water.fit(array_water_unique)
                pca_fat.fit(array_fat_unique)

                print(
                    "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_,
                                                                              nb_water_timesteps))
                print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

                transformed_array_water_unique = pca_water.transform(array_water_unique)
                transformed_array_fat_unique = pca_fat.transform(array_fat_unique)
                with open(pca_file,"wb") as file:
                    pickle.dump((pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique),file)
                if not calculate_matched_signals:
                    del array_water_unique
                    del array_fat_unique
            else:
                print("Loading pca")
                with open(pca_file, "rb") as file:
                    (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)=pickle.load(file)

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
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))

            #Trick to avoid returning matched signals in the coarse dictionary matching step
            return_matched_signals_backup=self.paramDict["return_matched_signals"]
            self.paramDict["return_matched_signals"]=False

            niter_backup=self.paramDict["niter"]
            self.paramDict["niter"]=0

            all_maps_bc_cf_light = self.search_patterns_test_multi(dictfile_light,all_signals)

            self.paramDict["return_matched_signals"] = return_matched_signals_backup
            self.paramDict["niter"]=niter_backup

            ind_high_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] >= threshold_ff)
            ind_low_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] < threshold_ff)
            all_maps_low_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_low_ff] for k in list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
            all_maps_high_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_high_ff] for k in
                                        list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
            unique_keys, labels = np.unique(all_maps_low_ff, axis=-1, return_inverse=True)
            #nb_clusters = unique_keys.shape[-1]
            unique_keys_high_ff, labels_high_ff = np.unique(all_maps_high_ff, axis=-1, return_inverse=True)

            nb_signals_low_ff = len(ind_low_ff)
            nb_signals_high_ff = len(ind_high_ff)

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
            d_fT1 = 100
            d_B1 = 0.2
            d_DF = 0.030#0.015

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
            d_fT1 = 100
            d_B1 = 0.2
            d_DF = 0.030#0.015


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

            if calculate_matched_signals:
                matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique.astype(int), :].T * (1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][idx_max_all_unique.astype(int), :].T * np.array(alpha_optim).reshape(1, -1)
                matched_signals *=np.linalg.norm(all_signals,axis=0)/np.linalg.norm(matched_signals, axis=0)
                matched_signals *= J_optim * np.exp(1j * phase_optim)


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
                if return_matched_signals:
                    values_results.append((map_rebuilt, mask,J_optim,phase_optim,matched_signals))
                else:
                    values_results.append((map_rebuilt, mask,J_optim,phase_optim))
            else:
                values_results.append((map_rebuilt, mask))

            # import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))

            if i == niter:
                break

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                keys = cp.asarray(keys)



            print("Generating prediction volumes and undersampled images for iteration {}".format(i))

            matched_signals=matched_signals.astype("complex64")
            matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])

            volumesi = undersampling_operator_new(matched_volumes, trajectory, self.paramDict["b1"],
                                              density_adj=dens_adj,ntimesteps=ntimesteps,retained_timesteps=retained_timesteps,weights=weights)

            # del images_pred

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            if log:
                print("Saving signalsi for iteration {}".format(i))
                with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(signalsi))

            del volumesi

            grad = signalsi - signals0
            del signalsi

            if self.paramDict["mu"] == "Adaptative":

                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                kdata_grad = generate_kdata_multi_new(grad_volumes, trajectory, self.paramDict["b1"],
                                                  ntimesteps=ntimesteps,retained_timesteps=retained_timesteps)

                del grad_volumes
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_grad.shape[1])):
                    curr_grad = grad[ts]
                    curr_kdata_grad = kdata_grad[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1

                    curr_kdata_grad_adjusted=(density * curr_kdata_grad.reshape(-1,trajectory.paramDict["npoint"])).flatten()
                    curr_kdata_grad_adjusted = curr_kdata_grad_adjusted.reshape((nb_channels,)+weights.shape[1:] + (-1,))
                    curr_weights = np.expand_dims(weights[ts], axis=(0,-1))
                    curr_kdata_grad_adjusted *= curr_weights
                    curr_kdata_grad_adjusted=curr_kdata_grad_adjusted.flatten()

                    mu_denom += np.real(np.dot(curr_kdata_grad_adjusted.conj(),curr_kdata_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))
                del kdata_grad
                mu = -num_samples * mu_numerator / mu_denom
                # mu/=2
            elif self.paramDict["mu"] == "Brute":
                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(matched_volumes + mu_ * grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu = mu_list[np.argmin(J_list)]
            else:
                mu = -self.paramDict["mu"]
            print("Mu for iter {} : {}".format(i, mu))

            all_signals = matched_signals + mu * grad

            if "mu_TV" not in self.paramDict:
                del grad


            if "mu_TV" in self.paramDict:
                print("Applying TV regularization")
                grad_norm = np.linalg.norm(grad, axis=0)
                del grad

                grad_TV=np.zeros(matched_signals.shape,dtype=matched_signals.dtype)
                for ts in tqdm(range(ntimesteps)):
                    matched_volumes_ts =makevol(matched_signals[ts], mask > 0)
                    for ind_w, w in (enumerate(weights_TV)):
                        if w > 0:
                            grad_TV[ts] += (w * grad_J_TV(matched_volumes_ts, ind_w, mask=mask,shift=0))[mask>0]


                grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                # signals = matched_signals + mu * grad

                all_signals += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

            del matched_signals


            # norm_signals = np.linalg.norm(signals, axis=0)
            # all_signals_unthresholded = signals / norm_signals



        if log:
            print(date_time)

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))


    def search_patterns_test_multi_grouping(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        ntimesteps = self.paramDict["ntimesteps"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            #            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"] = "Adaptative"

            if "dens_adj" in self.paramDict:
                dens_adj = self.paramDict["dens_adj"]
            else:
                dens_adj = True

            num_samples = trajectory.get_traj().reshape(ntimesteps, -1, 3).shape[1]

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase = self.paramDict["log_phase"]

        n_clusters_dico=self.paramDict["n_clusters_dico"]
        n_clusters_signals=self.paramDict["n_clusters_signals"]
        pruning = self.paramDict["pruning"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim > 2:
            all_signals = volumes[:, mask > 0]
        else:  # already masked
            all_signals = volumes

        all_signals=all_signals.astype("complex64")

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, all_signals)

        if "kdata_init" in self.paramDict:
            kdata_init = self.paramDict["kdata_init"]
            nb_channels = kdata_init.shape[0]
            kdata_init = kdata_init.reshape(nb_channels, ntimesteps, -1)

            def J(m, kdata_init, dens_adj, trajectory):
                kdata = generate_kdata_multi(m, trajectory, self.paramDict["b1"],
                                             ntimesteps=self.paramDict["ntimesteps"])
                kdata_error = kdata - kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
                    density = np.expand_dims(density, axis=0)
                    kdata_error *= np.sqrt(density)

                return np.linalg.norm(kdata_error) ** 2

        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

            if self.paramDict["mu"] == "Adaptative":

                kdata_all_signals = generate_kdata_multi(volumes, trajectory, self.paramDict["b1"],
                                                         ntimesteps=ntimesteps)
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_all_signals.shape[1])):
                    curr_grad = all_signals[ts]
                    curr_volumes_grad = kdata_all_signals[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1
                    mu_denom += np.real(
                        np.dot(
                            (density * curr_volumes_grad.reshape(-1,
                                                                 trajectory.paramDict["npoint"])).flatten().conj(),
                            curr_volumes_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                del kdata_all_signals

                mu0 = num_samples * mu_numerator / mu_denom

            elif self.paramDict["mu"] == "Brute":
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(- mu_ * volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu0 = -mu_list[np.argmin(J_list)]

            else:
                mu0 = self.paramDict["mu"]

            all_signals *= mu0
            print("Mu0 : {}".format(mu0))
            signals0 = copy(all_signals)/mu0

            if "mu_TV" in self.paramDict:
                mu_TV = self.paramDict["mu_TV"]

                if "weights_TV" not in self.paramDict:
                    self.paramDict["weights_TV"]=[1.,0.,0.]
                weights_TV=np.array(self.paramDict["weights_TV"])
                weights_TV/=np.sum(weights_TV)

        del volumes

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        # norm_signals = np.linalg.norm(signals, 2, axis=0)
        # Normalize
        # all_signals = signals / norm_signals
        #all_signals = signals
        if type(dictfile) == str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        else:  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dictfile[0]
            array_fat = dictfile[1]
            keys = dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_all = np.concatenate([np.real(array_water), np.imag(array_water), np.real(array_fat), np.imag(array_fat)],
                                   axis=-1)

        model_dico = KMeans(n_clusters=n_clusters_dico)
        model_dico.fit(array_all)

        array_water_unique_clustering = model_dico.cluster_centers_[:, :ntimesteps] + 1j * model_dico.cluster_centers_[:,
                                                                           (ntimesteps):(2 * ntimesteps)]
        array_fat_unique_clustering = model_dico.cluster_centers_[:, (2 * ntimesteps):(3 * ntimesteps)] + 1j * model_dico.cluster_centers_[:,
                                                                                               (3 * ntimesteps):(
                                                                                                           4 * ntimesteps)]

        del array_all

        nb_water_timesteps = array_water_unique_clustering.shape[1]
        nb_fat_timesteps = array_fat_unique_clustering.shape[1]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        var_w_total = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f_total = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf_total = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
                              axis=1).real
        var_w_total = var_w_total[index_water_unique]
        var_f_total = var_f_total[index_fat_unique]
        var_w_total = np.reshape(var_w_total, (-1, 1))
        var_f_total = np.reshape(var_f_total, (-1, 1))
        sig_wf_total = np.reshape(sig_wf_total, (-1, 1))

        #del array_water
        #del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique_clustering)
            pca_fat.fit(array_fat_unique_clustering)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_,
                                                                          nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique_clustering)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique_clustering)

        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        var_w = np.sum(array_water_unique_clustering * array_water_unique_clustering.conj(), axis=1).real
        var_f = np.sum(array_fat_unique_clustering * array_fat_unique_clustering.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique_clustering* array_fat_unique_clustering.conj(),
                        axis=1).real

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        model_signals = KMeans(n_clusters=n_clusters_signals)
        model_signals.fit(np.concatenate([np.real(all_signals), np.imag(all_signals)]).T)

        for i in range(niter + 1):

            idx_clusters=match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                     array_water_unique_clustering, array_fat_unique_clustering,
                                                                     transformed_array_water_unique,
                                                                     transformed_array_fat_unique, var_w, var_f,
                                                                     sig_wf, pca, None,
                                                                     None, remove_duplicates, verbose,
                                                                     0, split, useGPU_dictsearch, mask,
                                                                     tv_denoising_weight, False,n_clusters_dico=n_clusters_dico,pruning=pruning)



            if log:
                print("Saving signals for iteration {}".format(i))
                with open('./log/signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(all_signals))

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not (self.paramDict["return_matched_signals"]) and (niter == 0):
                map_rebuilt, J_optim, phase_optim = match_signals_v2_on_clusters(all_signals, keys, threshold_pca,
                                                                     array_water, array_fat,
                                                                      var_w_total, var_f_total,
                                                                     sig_wf_total, verbose,niter,
                                                                     useGPU_dictsearch,idx_clusters,model_dico,model_signals,
                                                                     log_phase)
            else:
                map_rebuilt, J_optim, phase_optim, matched_signals = match_signals_v2_on_clusters(all_signals, keys, threshold_pca,
                                                                     array_water, array_fat,
                                                                      var_w_total, var_f_total,
                                                                     sig_wf_total, verbose,niter,
                                                                     useGPU_dictsearch,idx_clusters,model_dico,model_signals,
                                                                     log_phase,return_matched_signals=True)


                if log:
                    print("Saving matched signals for iteration {}".format(i))
                    with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                        np.save(f, matched_signals.astype(np.complex64))

            # import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))

            if not (log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask, phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if i == niter:
                break

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))



            print("Generating prediction volumes and undersampled images for iteration {}".format(i))

            matched_signals=matched_signals.astype("complex64")
            matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])

            volumesi = undersampling_operator(matched_volumes, trajectory, self.paramDict["b1"],
                                              density_adj=dens_adj,light_memory_usage=True)

            del matched_volumes

            # del images_pred

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            if log:
                print("Saving signalsi for iteration {}".format(i))
                with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(signalsi))

            del volumesi

            grad = signalsi - signals0
            del signalsi

            if self.paramDict["mu"] == "Adaptative":

                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                kdata_grad = generate_kdata_multi(grad_volumes, trajectory, self.paramDict["b1"],
                                                  ntimesteps=ntimesteps)

                del grad_volumes
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_grad.shape[1])):
                    curr_grad = grad[ts]
                    curr_kdata_grad = kdata_grad[:, ts].flatten()
                    if dens_adj:
                        density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    else:
                        density = 1

                    mu_denom += np.real(
                        np.dot((density * curr_kdata_grad.reshape(-1,
                                                                  trajectory.paramDict["npoint"])).flatten().conj(),
                               curr_kdata_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))
                del kdata_grad
                mu = -num_samples * mu_numerator / mu_denom
                # mu/=2
            elif self.paramDict["mu"] == "Brute":
                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                mu_list = np.linspace(-1.6, -0, 8)
                J_list = [J(matched_volumes + mu_ * grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
                J_list = np.array(J_list)
                mu = mu_list[np.argmin(J_list)]
            else:
                mu = -self.paramDict["mu"]
            print("Mu for iter {} : {}".format(i, mu))

            all_signals = matched_signals + mu * grad

            if "mu_TV" not in self.paramDict:
                del grad


            if "mu_TV" in self.paramDict:
                print("Applying TV regularization")
                grad_norm = np.linalg.norm(grad, axis=0)
                del grad

                grad_TV=np.zeros(matched_signals.shape,dtype=matched_signals.dtype)
                for ts in tqdm(range(ntimesteps)):
                    matched_volumes_ts =makevol(matched_signals[ts], mask > 0)
                    for ind_w, w in (enumerate(weights_TV)):
                        if w > 0:
                            grad_TV[ts] += (w * grad_J_TV(matched_volumes_ts, ind_w, mask=mask,shift=0))[mask>0]


                grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                # signals = matched_signals + mu * grad

                all_signals += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

            del matched_signals


            # norm_signals = np.linalg.norm(signals, axis=0)
            # all_signals_unthresholded = signals / norm_signals



        if log:
            print(date_time)

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))



    # def search_patterns_test_multi_2_steps_dico(self, dictfile, volumes, retained_timesteps=None):

    #     if self.mask is None:
    #         mask = build_mask_from_volume(volumes)
    #     else:
    #         mask = self.mask

    #     verbose = self.verbose

    #     if "index_ref_def" not in self.paramDict:
    #         self.paramDict["index_ref_def"]=None



    #     if "nb_rep_center_part" not in self.paramDict:
    #         self.paramDict["nb_rep_center_part"]=1

    #     nb_rep_center_part=self.paramDict["nb_rep_center_part"]

    #     if "volumes_type" not in self.paramDict:
    #         self.paramDict["volumes_type"]="Normal"

    #     if ("deformation_map" in self.paramDict)and(self.paramDict["deformation_map"] is not None):
    #         deformation_map = self.paramDict["deformation_map"]
    #         if self.paramDict["index_ref_def"] is not None:
    #             change_deformation_map_ref(deformation_map,self.paramDict["index_ref_def"])
    #         nbins = deformation_map.shape[1]
    #         inv_deformation_map = np.zeros_like(deformation_map)
    #         for gr in range(nbins):
    #             inv_deformation_map[:, gr] = calculate_inverse_deformation_map(deformation_map[:, gr])
    #         #self.paramDict["multibins"] = True

    #     else:
    #         self.paramDict["deformation_map"] = None



    #     if "weights" not in self.paramDict:
    #         self.paramDict["weights"]=1
        
    #     if "clustering" not in self.paramDict:
    #         self.paramDict["clustering"]=True

    #     volumes_type=self.paramDict["volumes_type"]
    #     weights=self.paramDict["weights"]
        
    #     if (self.paramDict["weights"] is not None)and not(type(self.paramDict["weights"])==int):
    #         nbins=self.paramDict["weights"].shape[0]
    #         print("Weights shape : {}".format(self.paramDict["weights"].shape))

    #     niter = self.paramDict["niter"]
    #     split = self.paramDict["split"]
    #     pca = self.paramDict["pca"]

    #     if volumes.ndim==5:
    #         ntimesteps=volumes.shape[1]
    #     else:
    #         ntimesteps=volumes.shape[0]

    #     threshold_pca = self.paramDict["threshold_pca"]
        
    #     threshold_pca=np.minimum(ntimesteps,threshold_pca)

        
    #     threshold_ff=self.paramDict["threshold_ff"]
    #     dictfile_light=self.paramDict["dictfile_light"]


    #     # ntimesteps = self.paramDict["ntimesteps"]


    #     if "return_cost" not in self.paramDict:
    #         self.paramDict["return_cost"]=False
    #     return_cost = self.paramDict["return_cost"]

    #     if "calculate_matched_signals" not in self.paramDict:
    #         self.paramDict["calculate_matched_signals"]=False
    #     calculate_matched_signals = self.paramDict["calculate_matched_signals"]

    #     if "return_matched_signals" not in self.paramDict:
    #         self.paramDict["return_matched_signals"]=False
    #     return_matched_signals = self.paramDict["return_matched_signals"]

    #     if niter>0 or return_matched_signals:
    #         calculate_matched_signals=True

    #     if calculate_matched_signals:
    #         return_cost=True

    #     # useAdjPred=self.paramDict["useAdjPred"]
    #     if niter > 0:
    #         #            seq = self.paramDict["sequence"]
    #         trajectory = self.paramDict["trajectory"]
    #         gen_mode = self.paramDict["gen_mode"]
    #         if "mu" not in self.paramDict:
    #             self.paramDict["mu"] = "Adaptative"

    #         if "dens_adj" in self.paramDict:
    #             dens_adj = self.paramDict["dens_adj"]
    #         else:
    #             dens_adj = True

    #         dim=trajectory.get_traj().shape[-1]
    #         num_samples = trajectory.get_traj().reshape(ntimesteps, -1, dim).shape[1]

    #     log = self.paramDict["log"]
    #     if log:
    #         now = datetime.now()
    #         date_time = now.strftime("%Y%m%d_%H%M%S")
    #         filemap_log="map_MRF_{}.pkl".format(date_time)

    #     useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
    #     useGPU_simulation = self.paramDict["useGPU_simulation"]

        
    #     if "multibins" not in self.paramDict:
    #         self.paramDict["multibins"]=False



    #     movement_correction = self.paramDict["movement_correction"]
    #     cond_mvt = self.paramDict["cond"]
    #     if pca and (type(dictfile)==str):
    #         pca_file = str.split(dictfile, ".dict")[0] + "_{}pca.pkl".format(threshold_pca)
    #         pca_file_name = str.split(pca_file, "/")[-1]

    #     if type(dictfile)==str:
    #         vars_file = str.split(dictfile, ".dict")[0] + "_vars.pkl".format(threshold_pca)
    #         vars_file_name=str.split(vars_file,"/")[-1]
    #         path=str.split(os.path.realpath(__file__),"/dictoptimizers.py")[0]

    #     if movement_correction:
    #         if cond_mvt is None:
    #             raise ValueError("indices of retained kdata should be given in cond for movement correction")

    #     print(volumes.shape)
    #     if volumes.ndim > 2:
    #         if volumes.ndim==5:#3D with 1st dimension as the respiratory bin
    #             print("Multibins reco")
    #             ntimesteps=volumes.shape[1]
    #             nbins=volumes.shape[0]
    #             all_signals = volumes[:,:, mask > 0]
    #             all_signals=np.moveaxis(all_signals,1,0)
    #             all_signals=all_signals.reshape(ntimesteps,-1)
    #             self.paramDict["multibins"]=True
    #         else:
    #             all_signals = volumes[:, mask > 0]
            
    #     else:  # already masked
    #         all_signals = volumes

    #     if self.paramDict["multibins"]:
    #         if ("mu_bins" not in self.paramDict) or (self.paramDict["mu_bins"] is None):
    #             print("mu_bins not in parameters, setting it to 0.0")
    #             self.paramDict["mu_bins"]=0.0

    #     print(all_signals.shape)
    #     all_signals=all_signals.astype("complex64")
    #     nb_signals=all_signals.shape[1]


    #     if "kdata_init" in self.paramDict:
    #         kdata_init = self.paramDict["kdata_init"]
    #         nb_channels = kdata_init.shape[0]
    #         kdata_init = kdata_init.reshape(nb_channels, ntimesteps, -1)

    #         def J(m, kdata_init, dens_adj, trajectory):
    #             kdata = generate_kdata_multi(m, trajectory, self.paramDict["b1"],
    #                                          ntimesteps=self.paramDict["ntimesteps"])
    #             kdata_error = kdata - kdata_init
    #             if dens_adj:
    #                 density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
    #                 kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
    #                 density = np.expand_dims(density, axis=0)
    #                 kdata_error *= np.sqrt(density)

    #             return np.linalg.norm(kdata_error) ** 2

    #     if niter > 0:
    #         if "b1" not in self.paramDict:
    #             raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

    #         if self.paramDict["mu"] == "Adaptative":

    #             kdata_all_signals = generate_kdata_multi(volumes, trajectory, self.paramDict["b1"],
    #                                                      ntimesteps=ntimesteps)
    #             mu_numerator = 0
    #             mu_denom = 0

    #             for ts in tqdm(range(kdata_all_signals.shape[1])):
    #                 curr_grad = all_signals[ts]
    #                 curr_volumes_grad = kdata_all_signals[:, ts].flatten()
    #                 if dens_adj:
    #                     density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
    #                 else:
    #                     density = 1
    #                 mu_denom += np.real(
    #                     np.dot(
    #                         (density * curr_volumes_grad.reshape(-1,
    #                                                              trajectory.paramDict["npoint"])).flatten().conj(),
    #                         curr_volumes_grad))
    #                 mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

    #             del kdata_all_signals

    #             mu0 = num_samples * mu_numerator / mu_denom

    #         elif self.paramDict["mu"] == "Brute":
    #             mu_list = np.linspace(-1.6, -0, 8)
    #             J_list = [J(- mu_ * volumes, kdata_init, True, trajectory) for mu_ in mu_list]
    #             J_list = np.array(J_list)
    #             mu0 = -mu_list[np.argmin(J_list)]

    #         else:
    #             mu0 = self.paramDict["mu"]

    #         all_signals *= mu0
    #         print("Mu0 : {}".format(mu0))
    #         signals0 = copy(all_signals)/mu0

    #         print("multibins : {}".format(self.paramDict["multibins"]))
    #         if self.paramDict["multibins"]:
    #             signals0=signals0.reshape(ntimesteps,nbins,-1)
    #             signals0=np.moveaxis(signals0,1,0)

    #         if "mu_TV" in self.paramDict:
    #             mu_TV = self.paramDict["mu_TV"]

    #             if "weights_TV" not in self.paramDict:
    #                 self.paramDict["weights_TV"]=[1.,0.,0.]
    #             weights_TV=np.array(self.paramDict["weights_TV"])
    #             weights_TV/=np.sum(weights_TV)

    #     del volumes

    #     # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

    #     # norm_signals = np.linalg.norm(signals, 2, axis=0)
    #     # Normalize
    #     # all_signals = signals / norm_signals
    #     #all_signals = signals
    #     if type(dictfile) == str:
    #         mrfdict = dictsearch.Dictionary()
    #         mrfdict.load(dictfile, force=True)

    #         keys = mrfdict.keys
    #         array_water = mrfdict.values[:, :, 0]
    #         array_fat = mrfdict.values[:, :, 1]
    #         keys=np.array(keys)

    #         del mrfdict
    #     else:  # otherwise dictfile contains (s_w,s_f,keys)
    #         array_water = dictfile[0]
    #         array_fat = dictfile[1]
    #         keys = dictfile[2]
    #         keys=np.array(keys)

    #     if retained_timesteps is not None:
    #         array_water = array_water[:, retained_timesteps]
    #         array_fat = array_fat[:, retained_timesteps]

    #     #print(path)
    #     if not(type(dictfile)==str)or(vars_file_name not in os.listdir(path)) or ((pca) and (pca_file_name not in os.listdir(path))) or (calculate_matched_signals):

    #         print("Calculating unique dico signals")
    #         array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
    #         array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

    #     nb_water_timesteps = array_water.shape[1]
    #     nb_fat_timesteps = array_fat.shape[1]

    #     if not(type(dictfile)==str) or (vars_file_name not in os.listdir(path)):

    #         var_w_total = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
    #         var_f_total = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
    #         sig_wf_total = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(),
    #                               axis=1).real
    #         var_w_total = var_w_total[index_water_unique]
    #         var_f_total = var_f_total[index_fat_unique]
    #         var_w_total = np.reshape(var_w_total, (-1, 1))
    #         var_f_total = np.reshape(var_f_total, (-1, 1))
    #         sig_wf_total = np.reshape(sig_wf_total, (-1, 1))
    #         if type(dictfile)==str:
    #             with open(vars_file,"wb") as file:
    #                 pickle.dump((var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique),file)

    #     else:
    #         print("Loading var w / var f / sig wf")
    #         with open(vars_file, "rb") as file:
    #             (var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique)=pickle.load(file)

    #     #del array_water
    #     #del array_fat

    #     if pca:
    #         if not(type(dictfile)==str) or (pca_file_name not in os.listdir(path)):
    #             pca_water = PCAComplex(n_components_=threshold_pca)
    #             pca_fat = PCAComplex(n_components_=threshold_pca)

    #             pca_water.fit(array_water_unique)
    #             pca_fat.fit(array_fat_unique)

    #             print(
    #                 "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_,
    #                                                                           nb_water_timesteps))
    #             print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

    #             transformed_array_water_unique = pca_water.transform(array_water_unique)
    #             transformed_array_fat_unique = pca_fat.transform(array_fat_unique)
    #             if type(dictfile) == str:
    #                 with open(pca_file,"wb") as file:
    #                     pickle.dump((pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique),file)
    #             #if not calculate_matched_signals:
    #             #    del array_water_unique
    #             #    del array_fat_unique
    #         else:
    #             print("Loading pca")
    #             with open(pca_file, "rb") as file:
    #                 (pca_water, pca_fat, transformed_array_water_unique, transformed_array_fat_unique)=pickle.load(file)

    #     else:
    #         pca_water = None
    #         pca_fat = None
    #         transformed_array_water_unique = None
    #         transformed_array_fat_unique = None



    #     if useGPU_dictsearch:
    #         var_w_total = cp.asarray(var_w_total)
    #         var_f_total = cp.asarray(var_f_total)
    #         sig_wf_total = cp.asarray(sig_wf_total)
    #         keys=cp.asarray(keys)

    #     values_results = []
    #     keys_results = list(range(niter + 1))

    #     for i in range(niter + 1):

    #         print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
    #         print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))

    #         if self.paramDict["clustering"]:
    #             #Trick to avoid returning matched signals in the coarse dictionary matching step
    #             return_matched_signals_backup=self.paramDict["return_matched_signals"]
    #             self.paramDict["return_matched_signals"]=False

    #             niter_backup=self.paramDict["niter"]
    #             self.paramDict["niter"]=0

    #             all_maps_bc_cf_light = self.search_patterns_test_multi(dictfile_light,all_signals)

    #             self.paramDict["return_matched_signals"] = return_matched_signals_backup
    #             self.paramDict["niter"]=niter_backup

    #             ind_high_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] >= threshold_ff)
    #             ind_low_ff = np.argwhere(all_maps_bc_cf_light[0][0]["ff"] < threshold_ff)
    #             all_maps_low_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_low_ff] for k in list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
    #             all_maps_high_ff = np.array([all_maps_bc_cf_light[0][0][k][ind_high_ff] for k in
    #                                         list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
    #             unique_keys, labels = np.unique(all_maps_low_ff, axis=-1, return_inverse=True)
    #             #nb_clusters = unique_keys.shape[-1]
    #             unique_keys_high_ff, labels_high_ff = np.unique(all_maps_high_ff, axis=-1, return_inverse=True)

    #             nb_signals_low_ff = len(ind_low_ff)
    #             nb_signals_high_ff = len(ind_high_ff)

    #             idx_max_all_unique = np.zeros(nb_signals)
    #             alpha_optim = np.zeros(nb_signals)
    #             if return_cost:
    #                 J_optim = np.zeros(nb_signals)
    #                 phase_optim = np.zeros(nb_signals)

    #             if useGPU_dictsearch:
    #                 unique_keys=cp.asarray(unique_keys)
    #                 labels = cp.asarray(labels)
    #                 unique_keys_high_ff = cp.asarray(unique_keys_high_ff)
    #                 labels_high_ff = cp.asarray(labels_high_ff)

    #             all_signals_low_ff = all_signals[:, ind_low_ff.flatten()]
    #             all_signals_high_ff = all_signals[:, ind_high_ff.flatten()]

    #             d_T1 = 400
    #             d_fT1 = 100
    #             d_B1 = 0.2
    #             d_DF = 0.030  # 0.015

    #             if return_cost:
    #                 idx_max_all_unique_low_ff, alpha_optim_low_ff,J_optim_low_ff,phase_optim_low_ff = match_signals_v2_clustered_on_dico(all_signals_low_ff,
    #                                                                                                 keys, pca_water,
    #                                                                                                 pca_fat,
    #                                                                                                 transformed_array_water_unique,
    #                                                                                                 transformed_array_fat_unique,
    #                                                                                                 var_w_total,
    #                                                                                                 var_f_total,
    #                                                                                                 sig_wf_total,
    #                                                                                                 index_water_unique,
    #                                                                                                 index_fat_unique,
    #                                                                                                 useGPU_dictsearch,
    #                                                                                                 unique_keys, d_T1,
    #                                                                                                 d_fT1,
    #                                                                                                 d_B1, d_DF, labels,
    #                                                                                                 split, False,return_cost=True)

    #             else:
    #                 idx_max_all_unique_low_ff,alpha_optim_low_ff=match_signals_v2_clustered_on_dico(all_signals_low_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
    #                                         transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
    #                                         index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys, d_T1, d_fT1,
    #                                         d_B1, d_DF, labels,split,False)

    #             d_T1 = 400
    #             d_fT1 = 100
    #             d_B1 = 0.2
    #             d_DF = 0.030  # 0.015


    #             if return_cost:
    #                 idx_max_all_unique_high_ff, alpha_optim_high_ff,J_optim_high_ff,phase_optim_high_ff = match_signals_v2_clustered_on_dico(
    #                     all_signals_high_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
    #                     transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
    #                     index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys_high_ff, d_T1, d_fT1,
    #                     d_B1, d_DF, labels_high_ff, split, True,return_cost=True)
    #             else:
    #                 idx_max_all_unique_high_ff,alpha_optim_high_ff=match_signals_v2_clustered_on_dico(all_signals_high_ff, keys, pca_water, pca_fat, transformed_array_water_unique,
    #                                     transformed_array_fat_unique, var_w_total, var_f_total, sig_wf_total,
    #                                     index_water_unique, index_fat_unique, useGPU_dictsearch, unique_keys_high_ff, d_T1, d_fT1,
    #                                     d_B1, d_DF, labels_high_ff,split,True)



    #             idx_max_all_unique[ind_low_ff.flatten()] = idx_max_all_unique_low_ff
    #             idx_max_all_unique[ind_high_ff.flatten()] = idx_max_all_unique_high_ff

    #             alpha_optim[ind_low_ff.flatten()] = alpha_optim_low_ff
    #             alpha_optim[ind_high_ff.flatten()] = alpha_optim_high_ff

    #             if return_cost:
    #                 J_optim[ind_low_ff.flatten()] = J_optim_low_ff
    #                 J_optim[ind_high_ff.flatten()] = J_optim_high_ff

    #                 phase_optim[ind_low_ff.flatten()] = phase_optim_low_ff
    #                 phase_optim[ind_high_ff.flatten()] = phase_optim_high_ff
    #                 matched_signals = array_water_unique[index_water_unique, :][idx_max_all_unique.astype(int), :].T * (
    #                             1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][
    #                                                                         idx_max_all_unique.astype(int),
    #                                                                         :].T * np.array(alpha_optim).reshape(1, -1)
    #                 rho_optim= J_optim*np.linalg.norm(all_signals,axis=0)/np.linalg.norm(matched_signals, axis=0)

    #             if calculate_matched_signals:
    #                 matched_signals=array_water_unique[index_water_unique, :][idx_max_all_unique.astype(int), :].T * (1 - np.array(alpha_optim)).reshape(1, -1) + array_fat_unique[index_fat_unique, :][idx_max_all_unique.astype(int), :].T * np.array(alpha_optim).reshape(1, -1)
    #                 matched_signals *=np.linalg.norm(all_signals,axis=0)/np.linalg.norm(matched_signals, axis=0)
    #                 matched_signals *= J_optim * np.exp(1j * phase_optim)



    #             if useGPU_dictsearch:
    #                 keys=keys.get()

    #             keys_for_map = [tuple(k) for k in keys]

    #             params_all_unique = np.array(
    #                 [keys_for_map[idx] + (alpha_optim[l],) for l, idx in enumerate(idx_max_all_unique.astype(int))])
    #             map_rebuilt = {
    #                 "wT1": params_all_unique[:, 0],
    #                 "fT1": params_all_unique[:, 1],
    #                 "attB1": params_all_unique[:, 2],
    #                 "df": params_all_unique[:, 3],
    #                 "ff": params_all_unique[:, 4]

    #             }
    #             if return_cost:
    #                 if not(return_matched_signals):
    #                     values_results.append((map_rebuilt, mask,J_optim,phase_optim,rho_optim))
    #                 else:
    #                     values_results.append((map_rebuilt, mask,J_optim,phase_optim,rho_optim,matched_signals))
    #             else:
    #                 values_results.append((map_rebuilt, mask))

    #         else:
    #             #Trick to avoid returning matched signals in the coarse dictionary matching step
    #             return_matched_signals_backup=self.paramDict["return_matched_signals"]
                

    #             niter_backup=self.paramDict["niter"]
    #             self.paramDict["niter"]=0

    #             if calculate_matched_signals:
    #                 all_maps,matched_signals = self.search_patterns_test_multi(dictfile_light,all_signals)

    #             else:
    #                 all_maps = self.search_patterns_test_multi(dictfile_light,all_signals)
    #             self.paramDict["niter"]=niter_backup

    #             map_rebuilt=all_maps[0][0]
    #             mask=all_maps[0][1]

    #             if return_cost:
    #                 if not(return_matched_signals):
    #                     values_results.append((map_rebuilt, mask,None,None))
    #                 else:
    #                     values_results.append((map_rebuilt, mask,None,None,matched_signals))
    #             else:
    #                 values_results.append((map_rebuilt, mask))

    #         # import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
    #         print("Maps build for iteration {}".format(i))
            
    #         if log:
    #             print("Logging map {}".format(filemap_log))
    #             keys_temp=list(range(len(values_results)))
    #             with open(filemap_log,"wb") as file:
    #                 pickle.dump(dict(zip(keys_temp,values_results)),file)
    #         if i == niter:
    #             break

    #         if useGPU_dictsearch:  # Forcing to free the memory
    #             mempool = cp.get_default_memory_pool()
    #             print("Cupy memory usage {}:".format(mempool.used_bytes()))
    #             mempool.free_all_blocks()
    #             print("Cupy memory usage {}:".format(mempool.used_bytes()))
    #             keys=cp.asarray(keys)



    #         print("Generating prediction volumes and undersampled images for iteration {}".format(i))

    #         matched_signals=matched_signals.astype("complex64")
            

            

    #         if (not self.paramDict["multibins"])and(self.paramDict["deformation_map"] is None):
    #             matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])
    #             if volumes_type=="Normal":
    #                 volumesi = undersampling_operator(matched_volumes, trajectory, self.paramDict["b1"],
    #                                             density_adj=dens_adj,light_memory_usage=True)
                    
    #             elif volumes_type=="Singular":
    #                 volumesi=undersampling_operator_singular_new(matched_volumes,trajectory, self.paramDict["b1"],ntimesteps,density_adj=dens_adj,weights=weights,retained_timesteps=retained_timesteps,nb_rep_center_part=nb_rep_center_part)

    #             del matched_volumes

    #             # del images_pred

    #             # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

    #             # correct volumes
    #             print("Correcting volumes for iteration {}".format(i))

    #             signalsi = volumesi[:, mask > 0]
    #             # normi= np.linalg.norm(signalsi, axis=0)
    #             # signalsi *= norm_signals/normi



    #             del volumesi

    #             grad = signalsi - signals0
    #             del signalsi

    #             if self.paramDict["mu"] == "Adaptative":

    #                 grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
    #                 kdata_grad = generate_kdata_multi(grad_volumes, trajectory, self.paramDict["b1"],
    #                                                 ntimesteps=ntimesteps)

    #                 del grad_volumes
    #                 mu_numerator = 0
    #                 mu_denom = 0

    #                 for ts in tqdm(range(kdata_grad.shape[1])):
    #                     curr_grad = grad[ts]
    #                     curr_kdata_grad = kdata_grad[:, ts].flatten()
    #                     if dens_adj:
    #                         density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
    #                     else:
    #                         density = 1

    #                     mu_denom += np.real(
    #                         np.dot((density * curr_kdata_grad.reshape(-1,
    #                                                                 trajectory.paramDict["npoint"])).flatten().conj(),
    #                             curr_kdata_grad))
    #                     mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))
    #                 del kdata_grad
    #                 mu = -num_samples * mu_numerator / mu_denom
    #                 # mu/=2
    #             elif self.paramDict["mu"] == "Brute":
    #                 grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
    #                 mu_list = np.linspace(-1.6, -0, 8)
    #                 J_list = [J(matched_volumes + mu_ * grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
    #                 J_list = np.array(J_list)
    #                 mu = mu_list[np.argmin(J_list)]
    #             else:
    #                 mu = -self.paramDict["mu"]
    #             print("Mu for iter {} : {}".format(i, mu))

    #             all_signals = matched_signals + mu * grad

    #             if "mu_TV" not in self.paramDict:
    #                 del grad


    #             if "mu_TV" in self.paramDict:
    #                 print("Applying TV regularization")
    #                 grad_norm = np.linalg.norm(grad, axis=0)
    #                 del grad

    #                 grad_TV=np.zeros(matched_signals.shape,dtype=matched_signals.dtype)
    #                 for ts in tqdm(range(ntimesteps)):
    #                     matched_volumes_ts =makevol(matched_signals[ts], mask > 0)
    #                     for ind_w, w in (enumerate(weights_TV)):
    #                         if w > 0:
    #                             grad_TV[ts] += (w * grad_J_TV(matched_volumes_ts, ind_w, mask=mask,shift=0))[mask>0]


    #                 grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
    #                 # signals = matched_signals + mu * grad

    #                 all_signals += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
    #                 del grad_TV
    #                 del grad_TV_norm

    #             del matched_signals


    #             # norm_signals = np.linalg.norm(signals, axis=0)
    #             # all_signals_unthresholded = signals / norm_signals


    #         elif (not self.paramDict["multibins"])and(self.paramDict["deformation_map"] is not None):

    #             matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])

    #             for gr in range(nbins):
    #                 deformed_volumes=np.zeros_like(matched_volumes)
    #                 for ts in range(ntimesteps):
    #                     deformed_volumes[ts]=apply_deformation_to_complex_volume(matched_volumes[ts],inv_deformation_map[:,gr])

                    
    #                 if volumes_type == "Normal":
                        
    #                     volumesi = undersampling_operator(deformed_volumes, trajectory, self.paramDict["b1"],
    #                                                       density_adj=dens_adj, light_memory_usage=True)

    #                 elif volumes_type == "Singular":
    #                     print(deformed_volumes.shape)
    #                     print(weights.shape)
    #                     print(ntimesteps)
    #                     print(retained_timesteps)
    #                     volumesi = undersampling_operator_singular_new(deformed_volumes, trajectory, self.paramDict["b1"],
    #                                                                    ntimesteps, density_adj=dens_adj, weights=weights[gr],
    #                                                                    retained_timesteps=retained_timesteps,nb_rep_center_part=nb_rep_center_part)
                    
    #                 for ts in range(ntimesteps):
    #                     volumesi[ts]=apply_deformation_to_complex_volume(volumesi[ts],deformation_map[:,gr])

    #                 if gr==0:
    #                     final_volumesi=volumesi
    #                 else:
    #                     final_volumesi += volumesi

    #             del matched_volumes
    #             del volumesi

    #             # del images_pred

    #             # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

    #             # correct volumes
    #             print("Correcting volumes for iteration {}".format(i))

    #             signalsi = final_volumesi[:, mask > 0]
    #             # normi= np.linalg.norm(signalsi, axis=0)
    #             # signalsi *= norm_signals/normi


    #             del final_volumesi

                

    #             grad = signalsi - signals0
    #             del signalsi

    #             if self.paramDict["mu"] == "Adaptative":

    #                 grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
    #                 kdata_grad = generate_kdata_multi(grad_volumes, trajectory, self.paramDict["b1"],
    #                                                   ntimesteps=ntimesteps)

    #                 del grad_volumes
    #                 mu_numerator = 0
    #                 mu_denom = 0

    #                 for ts in tqdm(range(kdata_grad.shape[1])):
    #                     curr_grad = grad[ts]
    #                     curr_kdata_grad = kdata_grad[:, ts].flatten()
    #                     if dens_adj:
    #                         density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
    #                     else:
    #                         density = 1

    #                     mu_denom += np.real(
    #                         np.dot((density * curr_kdata_grad.reshape(-1,
    #                                                                   trajectory.paramDict["npoint"])).flatten().conj(),
    #                                curr_kdata_grad))
    #                     mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))
    #                 del kdata_grad
    #                 mu = -num_samples * mu_numerator / mu_denom
    #                 # mu/=2
    #             elif self.paramDict["mu"] == "Brute":
    #                 grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
    #                 mu_list = np.linspace(-1.6, -0, 8)
    #                 J_list = [J(matched_volumes + mu_ * grad_volumes, kdata_init, True, trajectory) for mu_ in mu_list]
    #                 J_list = np.array(J_list)
    #                 mu = mu_list[np.argmin(J_list)]
    #             else:
    #                 mu = -self.paramDict["mu"]
    #             print("Mu for iter {} : {}".format(i, mu))

    #             all_signals = matched_signals + mu * grad

    #             if "mu_TV" not in self.paramDict:
    #                 del grad

    #             if "mu_TV" in self.paramDict:
    #                 print("Applying TV regularization")
    #                 grad_norm = np.linalg.norm(grad, axis=0)
    #                 del grad

    #                 grad_TV = np.zeros(matched_signals.shape, dtype=matched_signals.dtype)
    #                 for ts in tqdm(range(ntimesteps)):
    #                     matched_volumes_ts = makevol(matched_signals[ts], mask > 0)
    #                     for ind_w, w in (enumerate(weights_TV)):
    #                         if w > 0:
    #                             grad_TV[ts] += (w * grad_J_TV(matched_volumes_ts, ind_w, mask=mask, shift=0))[mask > 0]

    #                 grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
    #                 # signals = matched_signals + mu * grad

    #                 all_signals += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
    #                 del grad_TV
    #                 del grad_TV_norm

    #             del matched_signals



    #         else:#multibins
    #             mu = -self.paramDict["mu"]
    #             print("Multi-bins reconstruction")
    #             matched_signals=matched_signals.reshape(ntimesteps,nbins,-1)
    #             matched_signals=np.moveaxis(matched_signals,1,0)
    #             all_signals=all_signals.reshape(ntimesteps,nbins,-1)
    #             all_signals=np.moveaxis(all_signals,1,0)


                

    #             all_grad_norm=0
    #             for gr in range(nbins):
    #                 matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals[gr]])
    #                 if volumes_type=="Normal":
    #                     volumesi = undersampling_operator(matched_volumes, trajectory, self.paramDict["b1"],
    #                                                 density_adj=dens_adj,light_memory_usage=True)
                        
    #                 elif volumes_type=="Singular":
    #                     print("Building undersampled singular volumes for multibin reco")
    #                     volumesi=undersampling_operator_singular_new(matched_volumes,trajectory, self.paramDict["b1"],ntimesteps,density_adj=dens_adj,weights=weights[gr],retained_timesteps=retained_timesteps,nb_rep_center_part=nb_rep_center_part)

    #                 del matched_volumes

    #                 # del images_pred

    #                 # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)

    #                 # correct volumes
    #                 print("Correcting volumes for iteration {}".format(i))

    #                 signalsi = volumesi[:, mask > 0]
    #                 # normi= np.linalg.norm(signalsi, axis=0)
    #                 # signalsi *= norm_signals/normi


    #                 del volumesi

    #                 grad = signalsi - signals0[gr]
    #                 del signalsi

    #                 all_signals[gr] = matched_signals[gr] - self.paramDict["mu"] * grad

    #                 if "mu_TV" not in self.paramDict:
    #                     del grad


    #                 if "mu_TV" in self.paramDict:
    #                     print("Applying TV regularization")
    #                     #grad_norm = np.linalg.norm(grad, axis=0)
    #                     grad_norm=np.linalg.norm(grad)
    #                     all_grad_norm+=grad_norm**2
    #                     print("grad norm {}".format(grad_norm))
    #                     del grad

    #                     grad_TV=np.zeros(matched_signals[gr].shape,dtype=matched_signals.dtype)
    #                     for ts in tqdm(range(ntimesteps)):
    #                         matched_volumes_ts =makevol(matched_signals[gr,ts], mask > 0)
    #                         for ind_w, w in (enumerate(weights_TV)):
    #                             if w > 0:
    #                                 grad_TV[ts] += (w * grad_J_TV(matched_volumes_ts, ind_w, mask=mask,shift=0))[mask>0]

    #                     #grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
    #                     grad_TV_norm = np.linalg.norm(grad_TV)
    #                     # signals = matched_signals + mu * grad

    #                     print("grad_TV_norm {}".format(grad_TV_norm))

    #                     all_signals[gr] += mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
    #                     del grad_TV
    #                     del grad_TV_norm

    #             all_grad_norm=np.sqrt(all_grad_norm)
    #             if ("mu_bins" in self.paramDict)and(self.paramDict["mu_bins"] is not None):
    #                 grad_TV_bins=grad_J_TV(matched_signals,0,is_weighted=False,shift=0)
    #                 #grad_TV_bins_norm=np.linalg.norm(grad_TV_bins,axis=(0,1))
    #                 grad_TV_bins_norm = np.linalg.norm(grad_TV_bins)

    #                 all_signals += mu * self.paramDict["mu_bins"] * grad_TV_bins/grad_TV_bins_norm*all_grad_norm
    #                 print("grad_TV_bins norm {}".format(grad_TV_bins_norm))

    #                 del grad_TV_bins

                
    #             all_signals=np.moveaxis(all_signals,1,0)
    #             #all_signals=all_signals[:,:,mask>0]
    #             all_signals=all_signals.reshape(ntimesteps,-1)










    #     #if self.paramDict["return_matched_signals"]:

    #     #    return dict(zip(keys_results, values_results)), matched_signals
    #     #else:
    #     return dict(zip(keys_results, values_results))






    def search_patterns_test_multi_singular(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
#            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            num_samples=trajectory.get_traj().reshape(-1, 3).shape[0]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"]="Adaptative"

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase=self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim>2:
            signals = volumes[:, mask > 0]
        else:#already masked
            signals=volumes

        # if log:
        #     now = datetime.now()
        #     date_time = now.strftime("%Y%m%d_%H%M%S")
        #     with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
        #         np.save(f, signals)

        #del volumes

        if "kdata_init" in self.paramDict:
            kdata_init=self.paramDict["kdata_init"]
            def J(m,kdata_init,dens_adj,trajectory):
                kdata=generate_kdata_singular_multi(m,trajectory,self.paramDict["b1"])
                kdata_error=kdata-kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error=kdata_error.reshape(-1,trajectory.paramDict["npoint"])
                    density=np.expand_dims(density,axis=0)
                    kdata_error*=np.sqrt(density)

                return np.linalg.norm(kdata_error)**2

        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")


            if self.paramDict["mu"]=="Adaptative":


                kdata_all_signals = generate_kdata_singular_multi(volumes,trajectory,self.paramDict["b1"])
                print('kdatashape', kdata_all_signals.shape)#shape 1 = 175?
                #L0=volumes.shape[0];test_volumes_all_signals = volumes_all_signals.reshape(L0,-1);test_volumes = volumes.reshape(L0,-1);
                #import matplotlib.pyplot as plt; j = np.random.choice(test_volumes.shape[-1]);plt.plot(test_volumes[:,j]);plt.plot(test_volumes_all_signals[:,j]);
                mu_numerator = 0
                mu_denom = 0

                for ts in tqdm(range(kdata_all_signals.shape[1])):
                    curr_grad = signals[ts]
                    curr_volumes_grad = kdata_all_signals[:, ts].flatten()
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    mu_denom += np.real(
                        np.dot((density * curr_volumes_grad.reshape(-1, trajectory.paramDict["npoint"])).flatten().conj(),
                               curr_volumes_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))

                del kdata_all_signals

                mu0 = num_samples * mu_numerator / mu_denom

            else:
                mu0 = self.paramDict["mu"]

            signals*=mu0
            print("Mu0 : {}".format(mu0))
            signals0 = signals

        del volumes

        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        #norm_signals = np.linalg.norm(signals, 2, axis=0)
        #Normalize
        #all_signals = signals / norm_signals
        all_signals=signals
        if type(dictfile)==str:
            raise ValueError("String not supported for singular volume matching - should project water and fat elements first and provide (w,f,keys)")
        else:#otherwise dictfile contains (s_w,s_f,keys)
            array_water=dictfile[0]
            array_fat=dictfile[1]
            keys=dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        else:
            pca_water=None
            pca_fat=None
            transformed_array_water_unique=None
            transformed_array_fat_unique=None

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        for i in range(niter + 1):

            # if log:
            #     print("Saving signals for iteration {}".format(i))
            #     with open('./log/signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
            #         np.save(f, np.array(all_signals))

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not(self.paramDict["return_matched_signals"])and(niter==0):
                map_rebuilt,J_optim,phase_optim=match_signals_v2(all_signals,keys, pca_water, pca_fat, array_water_unique, array_fat_unique,
                          transformed_array_water_unique, transformed_array_fat_unique, var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates, verbose,
                          niter, split, useGPU_dictsearch,mask,tv_denoising_weight,log_phase)
            else:
                map_rebuilt, J_optim, phase_optim,matched_signals = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                  array_water_unique, array_fat_unique,
                                                                  transformed_array_water_unique,
                                                                  transformed_array_fat_unique, var_w, var_f, sig_wf,
                                                                  pca, index_water_unique, index_fat_unique,
                                                                  remove_duplicates, verbose,
                                                                  niter, split, useGPU_dictsearch, mask,
                                                                  tv_denoising_weight, log_phase,return_matched_signals=True)



                # if log:
                #     print("Saving matched signals for iteration {}".format(i))
                #     with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                #         np.save(f, matched_signals.astype(np.complex64))

            #import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))

            if not(log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask,phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.pkl'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            if i == niter:
                break

            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))

            if i>0 and threshold is not None :
                all_signals=all_signals_unthresholded
                if not(self.paramDict["return_matched_signals"]):
                    map_rebuilt,J_optim,phase_optim = match_signals(all_signals,keys, pca_water, pca_fat, array_water_unique, array_fat_unique,
                              transformed_array_water_unique, transformed_array_fat_unique, var_w,var_f,sig_wf,pca,index_water_unique,index_fat_unique,remove_duplicates, verbose,
                              niter, split, useGPU_dictsearch,mask,tv_denoising_weight)
                else:
                    map_rebuilt, J_optim, phase_optim,matched_signals = match_signals(all_signals, keys, pca_water, pca_fat,
                                                                      array_water_unique, array_fat_unique,
                                                                      transformed_array_water_unique,
                                                                      transformed_array_fat_unique, var_w, var_f,
                                                                      sig_wf, pca, index_water_unique, index_fat_unique,
                                                                      remove_duplicates, verbose,
                                                                      niter, split, useGPU_dictsearch, mask,
                                                                      tv_denoising_weight,return_matched_signals=True)

            print("Generating prediction volumes and undersampled images for iteration {}".format(i))
            # generate prediction volumes
            #keys_simu = list(map_rebuilt.keys())
            #values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
            #map_for_sim = dict(zip(keys_simu, values_simu))




            #images_pred.build_ref_images(seq, norm=J_optim * norm_signals, phase=phase_optim)
            #matched_signals_extended=np.repeat(matched_signals,nspoke,axis=0)
            #matched_signals=matched_signals.astype("complex64")
            matched_volumes=np.array([makevol(im, mask > 0) for im in matched_signals])

            volumesi = undersampling_operator_singular(matched_volumes, trajectory, self.paramDict["b1"],density_adj=True)

            # volumesi = volumesi / np.linalg.norm(volumesi, 2, axis=0)



            # correct volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            if log:
                print("Saving signalsi for iteration {}".format(i))
                with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(signalsi))

            del volumesi




            #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signals0[:,j]);plt.plot(signalsi[:,j]);
            #j=np.random.choice(signals0.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(matched_signals[:,j]);plt.plot(signals0[:,j] - signalsi[:,j]);
            #signals = matched_signals +  signals0 - signalsi
            #mu=0.01;signals = mu*matched_signals +  mu*signals0 - mu**2*signalsi
            #import matplotlib.pyplot as plt;j=np.random.choice(matched_signals.shape[1]);import matplotlib.pyplot as plt;plt.figure();plt.plot(signalsi[:,j]);plt.plot(matched_signals[:,j])
            grad = signalsi - signals0/mu0

            if "kdata_init" in self.paramDict:
                print("Debugging cost function")
                volumes = np.array([makevol(im, mask > 0) for im in signals0])
                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                mu_list=np.linspace(-2,-0,10)
                J_list=[J(matched_volumes +mu_* grad_volumes,kdata_init,True,trajectory) for mu_ in mu_list]
                import matplotlib.pyplot as plt
                plt.figure();plt.plot(J_list)


            if self.paramDict["mu"]=="Adaptative":


                grad_volumes = np.array([makevol(im, mask > 0) for im in grad])
                kdata_grad = generate_kdata_singular_multi(grad_volumes,trajectory,self.paramDict["b1"])

                mu_numerator=0
                mu_denom=0


                for ts in tqdm(range(kdata_grad.shape[1])):
                    curr_grad=grad[ts]
                    curr_kdata_grad=kdata_grad[:,ts].flatten()
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))

                    mu_denom+=np.real(np.dot((density*curr_kdata_grad.reshape(-1,trajectory.paramDict["npoint"])).flatten().conj(),curr_kdata_grad))
                    mu_numerator += np.real(np.dot(curr_grad.conj(), curr_grad))


                mu = -num_samples*mu_numerator/mu_denom

            else:
                mu=-self.paramDict["mu"]
            print("Mu for iter {} : {}".format(i,mu))

            signals = matched_signals +mu* grad

            del grad
            #norm_signals = np.linalg.norm(signals, axis=0)
            #all_signals_unthresholded = signals / norm_signals

            if threshold is not None:

                signals_for_map = signals * (1 - (np.abs(signals) > threshold))
                all_signals = signals_for_map/np.linalg.norm(signals_for_map,axis=0)

            else:

                all_signals=signals

        if log:
            print(date_time)

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)),matched_signals
        else:
            return dict(zip(keys_results, values_results))

    def search_patterns_test_multi_CS(self, dictfile, volumes, retained_timesteps=None):

        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask

        verbose = self.verbose
        niter = self.paramDict["niter"]
        split = self.paramDict["split"]
        pca = self.paramDict["pca"]
        threshold_pca = self.paramDict["threshold_pca"]
        ntimesteps = self.paramDict["ntimesteps"]
        # useAdjPred=self.paramDict["useAdjPred"]
        if niter > 0:
            #            seq = self.paramDict["sequence"]
            trajectory = self.paramDict["trajectory"]
            gen_mode = self.paramDict["gen_mode"]
            if "mu" not in self.paramDict:
                self.paramDict["mu"] = 1
            if "kappa" not in self.paramDict:
                self.paramDict["kappa"]=0.9

            num_samples = trajectory.get_traj().reshape(ntimesteps, -1, 3).shape[1]

        log = self.paramDict["log"]
        useGPU_dictsearch = self.paramDict["useGPU_dictsearch"]
        useGPU_simulation = self.paramDict["useGPU_simulation"]

        movement_correction = self.paramDict["movement_correction"]
        cond_mvt = self.paramDict["cond"]
        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        threshold = self.paramDict["threshold"]
        tv_denoising_weight = self.paramDict["tv_denoising_weight"]
        log_phase = self.paramDict["log_phase"]
        # adj_phase=self.paramDict["adj_phase"]

        if movement_correction:
            if cond_mvt is None:
                raise ValueError("indices of retained kdata should be given in cond for movement correction")

        if volumes.ndim > 2:
            signals = volumes[:, mask > 0]
        else:  # already masked
            signals = volumes

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/signals0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, signals)

        if "kdata_init" in self.paramDict:
            kdata_init = self.paramDict["kdata_init"]
            nb_channels = kdata_init.shape[0]
            kdata_init = kdata_init.reshape(nb_channels, ntimesteps, -1)

            def J(m, kdata_init, dens_adj, trajectory):
                kdata = generate_kdata_multi(m, trajectory, self.paramDict["b1"],
                                             ntimesteps=self.paramDict["ntimesteps"])
                kdata_error = kdata - kdata_init
                if dens_adj:
                    density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
                    kdata_error = kdata_error.reshape(-1, trajectory.paramDict["npoint"])
                    density = np.expand_dims(density, axis=0)
                    kdata_error *= np.sqrt(density)

                return np.linalg.norm(kdata_error) ** 2

        if niter > 0:
            if "b1" not in self.paramDict:
                raise ValueError("b1 should be furnished for multi iteration multi channel MRF reconstruction")

            nspoke = int(trajectory.paramDict["total_nspokes"] / self.paramDict["ntimesteps"])

            mu=-self.paramDict["mu"]
            if "mu_TV" in self.paramDict:
                mu_TV = self.paramDict["mu_TV"]
                grad_TV = grad_J_TV(volumes, 1, mask=mask) + grad_J_TV(volumes, 2, mask=mask)
                grad_TV = grad_TV[:, mask > 0]
                grad_norm = np.linalg.norm(signals / mu0, axis=0)
                grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                signals -= mu0 * mu_TV * grad_norm / grad_TV_norm * grad_TV

                # signals_corrected=signals-0.5*grad_norm/grad_TV_norm*grad_TV

                # ts=0
                # vol_no_TV=makevol(signals[ts],mask>0)
                # vol_TV = makevol(signals_corrected[ts],mask>0)
                # from utils_mrf import animate_multiple_images
                # animate_multiple_images(vol_no_TV,vol_TV)

                # del grad_TV
            signals0 = copy(signals)

        del volumes

        matched_signals_prev=np.zeros(signals0.shape,dtype=signals0.dtype)
        signalsi=copy(matched_signals_prev)
        # norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        # norm_signals = np.linalg.norm(signals, 2, axis=0)
        # Normalize
        # all_signals = signals / norm_signals
        all_signals = signals
        if type(dictfile) == str:
            mrfdict = dictsearch.Dictionary()
            mrfdict.load(dictfile, force=True)

            keys = mrfdict.keys
            array_water = mrfdict.values[:, :, 0]
            array_fat = mrfdict.values[:, :, 1]

            del mrfdict
        else:  # otherwise dictfile contains (s_w,s_f,keys)
            array_water = dictfile[0]
            array_fat = dictfile[1]
            keys = dictfile[2]

        if retained_timesteps is not None:
            array_water = array_water[:, retained_timesteps]
            array_fat = array_fat[:, retained_timesteps]

        array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
        array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)

        nb_water_timesteps = array_water_unique.shape[1]
        nb_fat_timesteps = array_fat_unique.shape[1]

        del array_water
        del array_fat

        if pca:
            pca_water = PCAComplex(n_components_=threshold_pca)
            pca_fat = PCAComplex(n_components_=threshold_pca)

            pca_water.fit(array_water_unique)
            pca_fat.fit(array_fat_unique)

            print(
                "Water Components Retained {} out of {} timesteps".format(pca_water.n_components_, nb_water_timesteps))
            print("Fat Components Retained {} out of {} timesteps".format(pca_fat.n_components_, nb_fat_timesteps))

            transformed_array_water_unique = pca_water.transform(array_water_unique)
            transformed_array_fat_unique = pca_fat.transform(array_fat_unique)

        else:
            pca_water = None
            pca_fat = None
            transformed_array_water_unique = None
            transformed_array_fat_unique = None

        var_w = np.sum(array_water_unique * array_water_unique.conj(), axis=1).real
        var_f = np.sum(array_fat_unique * array_fat_unique.conj(), axis=1).real
        sig_wf = np.sum(array_water_unique[index_water_unique] * array_fat_unique[index_fat_unique].conj(), axis=1).real

        var_w = var_w[index_water_unique]
        var_f = var_f[index_fat_unique]

        var_w = np.reshape(var_w, (-1, 1))
        var_f = np.reshape(var_f, (-1, 1))
        sig_wf = np.reshape(sig_wf, (-1, 1))

        if useGPU_dictsearch:
            var_w = cp.asarray(var_w)
            var_f = cp.asarray(var_f)
            sig_wf = cp.asarray(sig_wf)

        values_results = []
        keys_results = list(range(niter + 1))

        i=0
        while(i<=niter):

            grad=(signalsi-signals0)
            all_signals=matched_signals_prev+mu*grad

            if log:
                print("Saving signals for iteration {}".format(i))
                with open('./log/signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(all_signals))

            print("################# ITERATION : Number {} out of {} ####################".format(i, niter))
            print("Calculating optimal fat fraction and best pattern per signal for iteration {}".format(i))
            if not (self.paramDict["return_matched_signals"]) and (niter == 0):
                map_rebuilt, J_optim, phase_optim = match_signals_v2(all_signals, keys, pca_water, pca_fat,
                                                                     array_water_unique, array_fat_unique,
                                                                     transformed_array_water_unique,
                                                                     transformed_array_fat_unique, var_w, var_f, sig_wf,
                                                                     pca, index_water_unique, index_fat_unique,
                                                                     remove_duplicates, verbose,
                                                                     niter, split, useGPU_dictsearch, mask,
                                                                     tv_denoising_weight, log_phase)
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
                                                                                      remove_duplicates, verbose,
                                                                                      niter, split, useGPU_dictsearch,
                                                                                      mask,
                                                                                      tv_denoising_weight, log_phase,
                                                                                      return_matched_signals=True)

                if log:
                    print("Saving matched signals for iteration {}".format(i))
                    with open('./log/matched_signals_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                        np.save(f, matched_signals.astype(np.complex64))

            # import matplotlib.pyplot as plt;j=np.random.choice(all_signals.shape[1]);plt.plot(all_signals[:,j]);plt.plot(matched_signals[:,j]);
            print("Maps build for iteration {}".format(i))


            if useGPU_dictsearch:  # Forcing to free the memory
                mempool = cp.get_default_memory_pool()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))
                mempool.free_all_blocks()
                print("Cupy memory usage {}:".format(mempool.used_bytes()))

            if i == niter:
                break





            print("Generating prediction volumes and undersampled images for iteration {}".format(i))

            matched_volumes_prev=np.array([makevol(im, mask > 0) for im in matched_signals_prev])
            matched_volumes = np.array([makevol(im, mask > 0) for im in matched_signals])
            diff_matched = matched_volumes-matched_volumes_prev
            del matched_volumes_prev
            density = np.abs(np.linspace(-1, 1, trajectory.paramDict["npoint"]))
            density = np.expand_dims(density, axis=(0))

            kappa=self.paramDict["kappa"]
            omega=kappa*np.sqrt(num_samples)*np.linalg.norm(diff_matched)/np.linalg.norm(np.sqrt(density)*generate_kdata_multi(diff_matched,trajectory,self.paramDict["b1"],ntimesteps=self.paramDict["ntimesteps"]).reshape(-1,trajectory.paramDict["npoint"]))

            del diff_matched
            print("Mu {} vs Omega {}".format(np.abs(mu),omega))

            if np.abs(mu)>omega:
                mu=mu/2
                print("Reducing mu : {}".format(mu))
                continue
            i = i + 1

            if not (log_phase):
                values_results.append((map_rebuilt, mask))

            else:
                values_results.append((map_rebuilt, mask, phase_optim))

            if log:
                with open('./log/maps_it_{}_{}.pkl'.format(int(i), date_time), 'wb') as f:
                    pickle.dump({int(i): (map_rebuilt, mask)}, f)

            volumesi = undersampling_operator(matched_volumes, trajectory, self.paramDict["b1"],
                                              density_adj=True)
            del matched_volumes
            print("Correcting volumes for iteration {}".format(i))

            signalsi = volumesi[:, mask > 0]
            matched_signals_prev=matched_signals
            # normi= np.linalg.norm(signalsi, axis=0)
            # signalsi *= norm_signals/normi

            if log:
                print("Saving signalsi for iteration {}".format(i))
                with open('./log/signalsi_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                    np.save(f, np.array(signalsi))

            del volumesi


            # norm_signals = np.linalg.norm(signals, axis=0)
            # all_signals_unthresholded = signals / norm_signals


        if log:
            print(date_time)

        if self.paramDict["return_matched_signals"]:

            return dict(zip(keys_results, values_results)), matched_signals
        else:
            return dict(zip(keys_results, values_results))


class ToyNN(Optimizer):

    def __init__(self,model,fitting_opt,model_opt={},input_scaler=StandardScaler(),output_scaler=MinMaxScaler(),niter=0,log=True,fitted=False,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)

        self.paramDict["model"]=model

        self.paramDict["input_scaler"]=input_scaler
        self.paramDict["output_scaler"] = output_scaler

        self.paramDict["fitting_opt"]=fitting_opt
        self.paramDict["is_fitted"] = fitted
        self.paramDict["model_opt"]=model_opt




    def search_patterns(self,dictfile,volumes,force_refit=False):
        model = self.paramDict["model"]
        fitting_opt=self.paramDict["fitting_opt"]

        if not(self.paramDict["is_fitted"]) or (force_refit):
            self.fit_and_set(dictfile)
            model=self.paramDict["model"]

        mask=self.mask
        all_signals = volumes[:, mask > 0]

        #all_signals=all_signals/np.expand_dims(np.linalg.norm(all_signals,axis=0),axis=0)

        real_signals=all_signals.real.T
        imag_signals=all_signals.imag.T

        #real_signals = real_signals / np.expand_dims(np.linalg.norm(real_signals, axis=-1), axis=-1)
        #imag_signals = imag_signals / np.expand_dims(np.linalg.norm(imag_signals, axis=-1), axis=-1)


        signals_for_model = np.concatenate((real_signals, imag_signals), axis=-1)
        #signals_for_model=signals_for_model/np.expand_dims(np.linalg.norm(signals_for_model,axis=-1),axis=-1)
        if self.paramDict["input_scaler"] is not None:
            signals_for_model=self.paramDict["input_scaler"].transform(signals_for_model)

        print(signals_for_model.shape)

        params_all = self.paramDict["output_scaler"].inverse_transform(model.predict(signals_for_model))

        map_rebuilt = {
            "wT1": params_all[:, 0],
            "fT1": params_all[:, 1],
            "attB1": params_all[:, 2],
            "df": params_all[:, 3],
            "ff": params_all[:, 4]

        }

        return {0:(map_rebuilt,mask)}




    def fit_and_set(self,dictfile):
        #For keras - shape is n_observations * n_features

        model = self.paramDict["model"]
        fitting_opt = self.paramDict["fitting_opt"]
        model_opt = self.paramDict["model_opt"]

        FF_list = list(np.arange(0., 1.05, 0.05))
        keys, signal = read_mrf_dict(dictfile, FF_list)

        Y_TF = np.array(keys)

        #signal=signal/np.expand_dims(np.linalg.norm(signal,axis=-1),axis=-1)

        real_signal = signal.real
        imag_signal = signal.imag

        #real_signal=real_signal/np.expand_dims(np.linalg.norm(real_signal,axis=-1),axis=-1)
        #imag_signal=imag_signal/np.expand_dims(np.linalg.norm(imag_signal,axis=-1),axis=-1)

        X_TF = np.concatenate((real_signal, imag_signal), axis=1)
        #X_TF = X_TF/np.expand_dims(np.linalg.norm(X_TF,axis=-1),axis=-1)

        input_scaler = self.paramDict["input_scaler"]
        output_scaler = self.paramDict["output_scaler"]

        if input_scaler is not None:
            X_TF = input_scaler.fit_transform(X_TF)
        Y_TF = output_scaler.fit_transform(Y_TF)

        self.paramDict["input_scaler"]=input_scaler
        self.paramDict["output_scaler"] = output_scaler

        print(X_TF.shape)
        print(Y_TF.shape)

        n_outputs = Y_TF.shape[1]

        print(n_outputs)

        final_model = model(n_outputs,**model_opt)

        history = final_model.fit(
            X_TF, Y_TF, **fitting_opt)

        self.paramDict["model"]=final_model
        self.paramDict["is_fitted"]=True




#
# class SPIJNSearch(Optimizer):
#
#     def __init__(self,num_comp=2,max_iter=10,ff_list=np.arange(0.,1.05,0.05),pca=True,threshold_pca=15,**kwargs):
#         #transf is a function that takes as input timesteps arrays and outputs shifts as output
#         super().__init__(**kwargs)
#
#         self.paramDict["num_comp"]=num_comp
#         self.paramDict["max_iter"] = max_iter
#         self.paramDict["ff_list"] = ff_list
#         self.paramDict["pca"] = pca
#         self.paramDict["threshold_pca"] = threshold_pca
#
#
#
#     def search_patterns(self,dictfile,volumes):
#
#         num_comp=self.paramDict["num_comp"]
#         max_iter=self.paramDict["max_iter"]
#         ff_list=self.paramDict["ff_list"]
#         pca=self.paramDict["pca"]
#         threshold_pca=self.paramDict["threshold_pca"]
#
#         keys,values=read_mrf_dict(dictfile,ff_list)
#
#         mask=self.mask
#         all_signals = volumes[:, mask > 0]
#         #values=values.T
#
#
#         if pca:
#             pca_values = PCAComplex(n_components_=threshold_pca)
#
#             pca_values.fit(values)
#
#             print(
#                 "Components Retained {} out of {} timesteps".format(pca_signal.n_components_, values.shape[1]))
#
#             transformed_values= pca_values.transform(values)
#
#             transformed_all_signals = np.transpose(
#                 pca_values.transform(np.transpose(all_signals)))
#
#
#         else:
#             pca_values=None
#             transformed_values=values
#             transformed_all_signals=all_signals
#
#
#         transformed_values=values.T
#
#         print(transformed_values.T)
#
#         map_rebuilt = {
#             "wT1": params_all[:, 0],
#             "fT1": params_all[:, 1],
#             "attB1": params_all[:, 2],
#             "df": params_all[:, 3],
#             "ff": params_all[:, 4]
#
#         }
#
#         return {0:(map_rebuilt,mask)}
#



class BruteDictSearch(Optimizer):

    def __init__(self,FF_list=np.arange(0.,1.05,0.05),split=500,pca=True,threshold_pca=15,useGPU_dictsearch=False,remove_duplicate_signals=False,log_phase=False,return_matched_signals=False,n_clusters_dico=None,pruning=None,**kwargs):
        #transf is a function that takes as input timesteps arrays and outputs shifts as output
        super().__init__(**kwargs)
        self.paramDict["split"] = split
        self.paramDict["pca"] = pca
        self.paramDict["threshold_pca"] = threshold_pca
        self.paramDict["remove_duplicate_signals"] = remove_duplicate_signals
        self.paramDict["FF"]=FF_list
        self.paramDict["return_matched_signals"]=return_matched_signals
        #self.paramDict["useAdjPred"]=useAdjPred



        self.paramDict["useGPU_dictsearch"]=useGPU_dictsearch
        self.paramDict["log_phase"] = log_phase

        self.paramDict["n_clusters_dico"]=n_clusters_dico
        self.paramDict["pruning"] = pruning

        if "ignore_rho" not in self.paramDict:
            self.paramDict["ignore_rho"]=False

        if "dictfile_light" not in self.paramDict:
            self.paramDict["dictfile_light"]=None

    def search_patterns(self,dictfile,volumes,retained_timesteps=None):


        if self.mask is None:
            mask = build_mask_from_volume(volumes)
        else:
            mask = self.mask


        verbose=self.verbose
        split=self.paramDict["split"]
        pca=self.paramDict["pca"]
        threshold_pca=self.paramDict["threshold_pca"]
        #useAdjPred=self.paramDict["useAdjPred"]
        log=self.paramDict["log"]
        useGPU_dictsearch=self.paramDict["useGPU_dictsearch"]


        remove_duplicates = self.paramDict["remove_duplicate_signals"]
        #adj_phase=self.paramDict["adj_phase"]
        n_clusters_dico = self.paramDict["n_clusters_dico"]
        pruning = self.paramDict["pruning"]
        if n_clusters_dico is not None:
            clustering_file=str.split(dictfile,".dict")[0]+"_{}groups.pkl".format(n_clusters_dico)
            clustering_file_name=str.split(clustering_file, "/")[-1]

        if pca:
            pca_file = str.split(dictfile, ".dict")[0] + "_{}pca_brute.pkl".format(threshold_pca)
            pca_file_name = str.split(pca_file, "/")[-1]

        path = str.split(os.path.realpath(__file__), "/dictoptimizers.py")[0]

        if volumes.ndim > 2:
            signals = volumes[:, mask > 0]
        else:  # already masked
            signals = volumes

        if log:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d_%H%M%S")
            with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                np.save(f, volumes)


        del volumes


        #norm_volumes = np.linalg.norm(volumes, 2, axis=0)

        norm_signals=np.linalg.norm(signals, 2, axis=0)
        all_signals = signals/norm_signals

        keys,values=read_mrf_dict(dictfile,self.paramDict["FF"])


        if retained_timesteps is not None:
            values=values[:,retained_timesteps]

        if pca:
            if pca_file_name not in os.listdir(path):
                pca = PCAComplex(n_components_=threshold_pca)
                pca.fit(values)
                transformed_values = pca.transform(values)
                with open(pca_file, "wb") as file:
                    pickle.dump((pca,transformed_values), file)
            else:
                with open(pca_file,"rb") as file:
                    (pca,transformed_values)=pickle.load(file)

        else:
            transformed_values=values

        var = np.sum(transformed_values * transformed_values.conj(), axis=1).real


        values_results = []
        keys_results = [0]


        print("Calculating optimal fat fraction and best pattern per signal")
        nb_signals = all_signals.shape[1]
        nb_signals_dico=values.shape[0]

        if n_clusters_dico is not None:

            if clustering_file_name not in os.listdir(path):
                #print(os.listdir())
                print("Forming dictionary groups for group matching")
                n_el_per_group = int(nb_signals_dico / n_clusters_dico)

                curr_values = copy(values)
                s_cur = values[0]

                cluster_signals = []
                labels = np.zeros(nb_signals_dico)
                for j in tqdm(range(n_clusters_dico+1)):
                    if j == (n_clusters_dico):
                        n_el_per_group_curr = nb_signals_dico - n_clusters_dico * n_el_per_group
                        if n_el_per_group_curr==0:
                            break
                    else:
                        n_el_per_group_curr = n_el_per_group
                    indices_group = np.argsort(np.linalg.norm(curr_values - s_cur, axis=1))[:n_el_per_group_curr]
                    labels[indices_group] = j
                    #dico_group_indices_aggreg.append(list(indices_group))

                    s_cur = np.mean(curr_values[indices_group], axis=0)
                    cluster_signals.append(s_cur)
                    # curr_values=np.delete(curr_values,indices_group,axis=0)
                    curr_values[indices_group] = np.inf

                with open(clustering_file,"wb") as file:
                    pickle.dump((cluster_signals,labels),file)
            else:
                with open(clustering_file,"rb") as file:
                    (cluster_signals, labels)=pickle.load(file)


            labels = labels.astype(int)
            cluster_signals = np.array(cluster_signals)

            if pca:
                cluster_signals = pca.transform(cluster_signals)

            var_clusters = np.sum(cluster_signals * cluster_signals.conj(), axis=1).real

            if useGPU_dictsearch:
                var_clusters=cp.asarray(var_clusters)
                labels=cp.asarray(labels)
                cluster_signals=cp.asarray(cluster_signals)

        if remove_duplicates:
            all_signals, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)
            nb_signals = all_signals.shape[1]



        print("There are {} unique signals to match along {} dic components".format(nb_signals,nb_signals_dico))


        if n_clusters_dico is None:
            print("Standard Dictionary matching without group matching")
            num_group = int(nb_signals / split) + 1

            idx_max_all_unique = []

            lambd=[]



            for j in tqdm(range(num_group)):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals)
                if j_signal>=nb_signals:
                    break

                if self.verbose:
                    print("PCA transform")
                    start = datetime.now()


                if not(useGPU_dictsearch):

                    if pca:
                        transformed_all_signals = np.transpose(pca.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))

                        sig = np.matmul(transformed_values.conj(),
                                                      transformed_all_signals)

                        phi = 0.5 * np.angle(sig ** 2 / np.expand_dims(np.real(var), axis=-1))
                        sig=np.real(np.matmul(transformed_values.conj(), transformed_all_signals) * np.exp(
                            -1j * phi))


                    else:
                        sig = np.matmul(values.conj(), all_signals[:, j_signal:j_signal_next])
                        phi = 0.5 * np.angle(sig ** 2 / np.expand_dims(np.real(var), axis=-1))
                        sig=np.real(np.matmul(values.conj(), all_signals[:, j_signal:j_signal_next]) * np.exp(
                            -1j * phi))

                    lambd = sig / np.expand_dims(np.real(var), axis=-1)


                else:


                    if pca:
                        transformed_all_signals = cp.transpose(
                            pca.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next]))))

                        sig = (cp.matmul(cp.asarray(transformed_values).conj(),
                                                       cp.asarray(transformed_all_signals)))

                        phi = 0.5 * cp.angle(sig ** 2 / cp.expand_dims(cp.real(cp.asarray(var)), axis=-1))
                        sig = cp.real(cp.matmul(cp.asarray(transformed_values).conj(), transformed_all_signals) * cp.exp(
                            -1j * phi))

                    else:

                        sig = (cp.matmul(cp.asarray(values).conj(),
                                                       cp.asarray(all_signals)[:, j_signal:j_signal_next]))

                        phi = 0.5 * cp.angle(sig ** 2 / cp.expand_dims(cp.real(var), axis=-1))
                        sig = cp.real(cp.matmul(values.conj(), all_signals[:, j_signal:j_signal_next] )* cp.exp(
                            -1j * phi))

                    lambd = sig / cp.expand_dims(cp.real(var), axis=-1)

                if self.verbose:
                    end = datetime.now()
                    print(end - start)

                if self.verbose:
                    start = datetime.now()


                #current_sig_ws = current_sig_ws_for_phase.real
                #current_sig_fs = current_sig_fs_for_phase.real

                if self.verbose:
                    end = datetime.now()
                    print(end-start)

                if not(useGPU_dictsearch):
                    #if adj_phase:
                    if self.verbose:
                        print("Adjusting Phase")
                        print("Calculating alpha optim and flooring")

                        ### Testing direct phase solving

                    J_all=lambd**2*np.expand_dims(var,axis=-1) - 2*lambd*sig

                    if not(self.paramDict["return_matched_signals"]):
                        del lambd
                        del phi

                    end = datetime.now()

                else:
                    
                    J_all=lambd**2*cp.expand_dims(cp.asarray(var),axis=-1) - 2*lambd*sig

                    J_all = J_all.get()

                    if not(self.paramDict["return_matched_signals"]):
                        del lambd
                        del phi
                    else:
                        lambd=lambd.get()
                        phi=phi.get()


                    if verbose:
                        end = datetime.now()
                        print(end - start)

                if verbose:
                    print("Extracting index of pattern with max correl")
                    start = datetime.now()

                idx_max_all_current = np.argmin(J_all, axis=0)
                #check_max_correl=np.max(J_all,axis=0)

                if verbose:
                    end = datetime.now()
                    print(end-start)

                if verbose:
                    print("Filling the lists with results for this loop")
                    start = datetime.now()

                idx_max_all_unique.extend(idx_max_all_current)

                if verbose:
                    end = datetime.now()
                    print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all

        else:

            num_group = int(nb_signals / split) + 1

            #idx_max_all_current = []

            idx_max_all_unique=np.zeros(nb_signals,dtype="int64")

            #lambd = []

            for j in tqdm(range(num_group)):
                j_signal = j * split
                j_signal_next = np.minimum((j + 1) * split, nb_signals)

                if j_signal>=nb_signals:
                    break

                if self.verbose:
                    print("PCA transform")
                    start = datetime.now()

                if not (useGPU_dictsearch):

                    transformed_all_signals = np.transpose(
                        pca.transform(np.transpose(all_signals[:, j_signal:j_signal_next])))

                    sig = np.matmul(cluster_signals.conj(),transformed_all_signals)

                    phi = 0.5 * np.angle(sig ** 2 / np.expand_dims(np.real(var_clusters), axis=-1))
                    sig = np.real(np.matmul(cluster_signals.conj(), transformed_all_signals) * np.exp(
                        -1j * phi))

                    lambd = sig / np.expand_dims(np.real(var_clusters), axis=-1)

                    J_all_clusters = lambd ** 2 * np.expand_dims(var_clusters, axis=-1) - 2 * lambd * sig

                    idx_max_all_clusters_current = np.argsort(J_all_clusters, axis=0)[:int(pruning*n_clusters_dico)]

                    retained_clusters = idx_max_all_clusters_current.flatten()
                    retained_signals_indices = np.where(np.in1d(labels, retained_clusters))[0]

                    # if len(idx_max_all_clusters) == 0:
                    #     idx_max_all_clusters = idx_max_all_clusters_current
                    # else:
                    #     idx_max_all_clusters = np.append(idx_max_all_unique, idx_max_all_clusters_current, axis=1)
                    var_current=var[retained_signals_indices]
                    retained_signals_dico=transformed_values[retained_signals_indices]

                    sig = np.matmul(retained_signals_dico.conj(), transformed_all_signals)

                    phi = 0.5 * np.angle(sig ** 2 / np.expand_dims(np.real(var_current), axis=-1))
                    sig = np.real(np.matmul(retained_signals_dico.conj(), transformed_all_signals) * np.exp(
                            -1j * phi))

                    lambd = sig / np.expand_dims(np.real(var_current), axis=-1)

                    J_all = lambd ** 2 * np.expand_dims(var_current, axis=-1) - 2 * lambd * sig
                    idx_max_all_current = np.argmin(J_all, axis=0)
                    idx_max_all_unique[j_signal:j_signal_next]=(retained_signals_indices[idx_max_all_current])

                else:

                    transformed_all_signals = cp.transpose(
                        pca.transform(cp.transpose(cp.asarray(all_signals[:, j_signal:j_signal_next]))))

                    sig = cp.matmul(cp.asarray(cluster_signals.conj()), transformed_all_signals)

                    phi = 0.5 * cp.angle(sig ** 2 / cp.expand_dims(cp.real(var_clusters), axis=-1))
                    sig = cp.real(cp.matmul(cp.asarray(cluster_signals.conj()), transformed_all_signals) * cp.exp(
                        -1j * phi))

                    if self.paramDict["ignore_rho"]:
                        lambd=1
                    else:
                        lambd = sig / cp.expand_dims(cp.real(var_clusters), axis=-1)

                    J_all_clusters = lambd ** 2 * cp.expand_dims(var_clusters, axis=-1) - 2 * lambd * sig

                    idx_max_all_clusters_current = cp.argsort(J_all_clusters, axis=0)[:int(pruning * n_clusters_dico)]

                    retained_clusters = idx_max_all_clusters_current.flatten()
                    retained_signals_indices = cp.where(cp.in1d(labels, retained_clusters))[0]

                    # if len(idx_max_all_clusters) == 0:
                    #     idx_max_all_clusters = idx_max_all_clusters_current
                    # else:
                    #     idx_max_all_clusters = np.append(idx_max_all_unique, idx_max_all_clusters_current, axis=1)
                    var_current = cp.asarray(var[retained_signals_indices.get()])
                    retained_signals_dico =cp.asarray(transformed_values[retained_signals_indices.get()])

                    sig = cp.matmul(retained_signals_dico.conj(), transformed_all_signals)

                    phi = 0.5 * cp.angle(sig ** 2 / cp.expand_dims(cp.real(var_current), axis=-1))
                    sig = cp.real(cp.matmul(retained_signals_dico.conj(), transformed_all_signals) * cp.exp(
                        -1j * phi))

                    if self.paramDict["ignore_rho"]:
                        lambd=1
                    else:
                        lambd = sig / cp.expand_dims(cp.real(var_current), axis=-1)

                    J_all = lambd ** 2 * cp.expand_dims(var_current, axis=-1) - 2 * lambd * sig
                    idx_max_all_current = cp.argmin(J_all, axis=0)
                    idx_max_all_unique[j_signal:j_signal_next] = (retained_signals_indices[idx_max_all_current]).get()

                if verbose:
                    end = datetime.now()
                    print(end - start)

            # idx_max_all_unique = np.argmax(J_all, axis=0)
            del J_all




        print("Building the maps for iteration")

        # del sig_ws_all_unique
        # del sig_fs_all_unique

        params_all_unique = np.array(
            [keys[idx]  for l, idx in enumerate(idx_max_all_unique)])

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

        values_results.append((map_rebuilt, mask))

        if log:
            with open('./log/maps_it_{}_{}.npy'.format(int(i), date_time), 'wb') as f:
                pickle.dump({int(i):(map_rebuilt, mask)}, f)

        if useGPU_dictsearch:#Forcing to free the memory
            mempool = cp.get_default_memory_pool()
            print("Cupy memory usage {}:".format(mempool.used_bytes()))
            mempool.free_all_blocks()

            cp.cuda.set_allocator(None)
            # Disable memory pool for pinned memory (CPU).
            cp.cuda.set_pinned_memory_allocator(None)
            gc.collect()

        if self.paramDict["return_matched_signals"]:
            print(nb_signals)
            print(lambd.shape)
            print(transformed_values.shape)
            print(phi.shape)
            matched_signals=np.expand_dims(lambd,axis=1)*np.expand_dims(transformed_values,axis=-1)*np.expand_dims(np.exp(1j*phi),axis=1)
            print(matched_signals.shape)
            print(np.array(idx_max_all_unique).shape)
            matched_signals = matched_signals[idx_max_all_unique, :, np.arange(nb_signals)].T
            if pca:
                matched_signals = pca.components_ @ matched_signals
            return dict(zip(keys_results, values_results)),matched_signals

        else:

            return dict(zip(keys_results, values_results))


    def search_patterns_new_clustering(self,dictfile,volumes,retained_timesteps=None):


            if self.mask is None:
                mask = build_mask_from_volume(volumes)
            else:
                mask = self.mask


            if "dictfile_light" not in self.paramDict:
                raise ValueError("dictfile_light should be a member for this mathing method")

            split=self.paramDict["split"]
            pca=self.paramDict["pca"]
            threshold_pca=self.paramDict["threshold_pca"]
            #useAdjPred=self.paramDict["useAdjPred"]
            log=self.paramDict["log"]
            useGPU_dictsearch=self.paramDict["useGPU_dictsearch"]


            remove_duplicates = self.paramDict["remove_duplicate_signals"]
            #adj_phase=self.paramDict["adj_phase"]
            self.paramDict["n_clusters_dico"]=None
            self.paramDict["pruning"]=None

            if pca:
                pca_file = str.split(dictfile, ".dict")[0] + "_{}pca_brute.pkl".format(threshold_pca)
                pca_file_name = str.split(pca_file, "/")[-1]

            path = str.split(os.path.realpath(__file__), "/dictoptimizers.py")[0]
           
            
            keys,values=read_mrf_dict(dictfile,self.paramDict["FF"])

            keys=cp.asarray(keys)

            if volumes.ndim > 2:
                signals = volumes[:, mask > 0]
            else:  # already masked
                signals = volumes

            if log:
                now = datetime.now()
                date_time = now.strftime("%Y%m%d_%H%M%S")
                with open('./log/volumes0_{}.npy'.format(date_time), 'wb') as f:
                    np.save(f, volumes)


            del volumes


            #norm_volumes = np.linalg.norm(volumes, 2, axis=0)

            norm_signals=np.linalg.norm(signals, 2, axis=0)
            all_signals = signals/norm_signals



            if retained_timesteps is not None:
                values=values[:,retained_timesteps]

            if pca:
                if pca_file_name not in os.listdir(path):
                    pca = PCAComplex(n_components_=threshold_pca)
                    pca.fit(values)
                    transformed_values = pca.transform(values)
                    with open(pca_file, "wb") as file:
                        pickle.dump((pca,transformed_values), file)
                else:
                    with open(pca_file,"rb") as file:
                        (pca,transformed_values)=pickle.load(file)

            else:
                transformed_values=values

            var_total = np.sum(transformed_values * transformed_values.conj(), axis=1).real


            values_results = []
            keys_results = [0]


            print("Calculating optimal fat fraction and best pattern per signal")
            nb_signals = all_signals.shape[1]

            
            num_group = int(nb_signals / split) + 1
            idx_max_all_unique=np.zeros(nb_signals,dtype="int64")

        #Trick to avoid returning matched signals in the coarse dictionary matching step
            return_matched_signals_backup=self.paramDict["return_matched_signals"]
            self.paramDict["return_matched_signals"]=False

            all_maps_bc_cf_light = self.search_patterns(self.paramDict["dictfile_light"],all_signals)

            self.paramDict["return_matched_signals"] = return_matched_signals_backup


            print("Calculating clusters")
            all_maps = np.array([all_maps_bc_cf_light[0][0][k] for k in list(all_maps_bc_cf_light[0][0].keys())[:-1]]).squeeze()
            unique_keys, labels = np.unique(all_maps, axis=-1, return_inverse=True)

            unique_keys=cp.asarray(unique_keys)
            labels = cp.asarray(labels)
    
            d_T1 = 400
            d_fT1 = 100
            d_B1 = 0.2
            d_DF = 0.015

            nb_clusters = unique_keys.shape[-1]

            print("Performing standard dictionary matching on clustered signals")
            print(nb_clusters)

            for cl in tqdm(range(nb_clusters)):

                indices = cp.argwhere(labels == cl)
                nb_signals_cluster=len(indices)
                num_group = int(nb_signals_cluster / split) + 1
                #print("nb_signals_cluster {}".format(nb_signals_cluster))

                #print(keys[:10])
                # if j_signal==nb_signals:
                #    break
                keys_T1 = (keys[:, 0] < unique_keys[:, cl][0] + d_T1) & ((keys[:, 0] > unique_keys[:, cl][0] - d_T1))
                keys_fT1 = (keys[:, 1] < unique_keys[:, cl][1] + d_fT1) & ((keys[:, 1] > unique_keys[:, cl][1] - d_fT1))
                keys_B1 = (keys[:, 2] < unique_keys[:, cl][2] + d_B1) & ((keys[:, 2] > unique_keys[:, cl][2] - d_B1))
                keys_DF = (keys[:, 3] < unique_keys[:, cl][3] + d_DF) & ((keys[:, 3] > unique_keys[:, cl][3] - d_DF))
                retained_signals_indices = cp.argwhere(keys_T1 & keys_fT1 & keys_B1 & keys_DF).flatten()


                var = var_total[retained_signals_indices.get()]

                all_signals_cluster=cp.asarray(all_signals[:, (indices.get()).flatten()])
                idx_max_all_unique_cluster = cp.zeros(nb_signals_cluster,dtype="int64")
                
                retained_signals_dico =cp.asarray(transformed_values[retained_signals_indices.get()])

                
  
                for j in tqdm(range(num_group)):
                    j_signal = j * split
                    j_signal_next = np.minimum((j + 1) * split, nb_signals_cluster)
                    if j_signal>=nb_signals_cluster:
                        break

                        
                    #print("Applying PCA on clustered signal")
                    transformed_all_signals = cp.transpose(
                                    pca.transform(cp.transpose(cp.asarray(all_signals_cluster[:, j_signal:j_signal_next]))))

                    #print("transformed_all_signals shape {}".format(transformed_all_signals.shape))

                                # if len(idx_max_all_clusters) == 0:
                                #     idx_max_all_clusters = idx_max_all_clusters_current
                                # else:
                                #     idx_max_all_clusters = np.append(idx_max_all_unique, idx_max_all_clusters_current, axis=1)
                    
                    
                    var_current = cp.asarray(var)
                    #print("Var current shape {}".format(var_current.shape))
                    
                    sig = cp.matmul(retained_signals_dico.conj(), transformed_all_signals)
                    #print("sig.shape {}".format(sig.shape))

                    phi = 0.5 * cp.angle(sig ** 2 / cp.expand_dims(cp.real(var_current), axis=-1))
                    #print("phi.shape {}".format(phi.shape))

                    sig = cp.real(cp.matmul(retained_signals_dico.conj(), transformed_all_signals) * cp.exp(
                                    -1j * phi))
                    
                    #print("sig.shape {}".format(sig.shape))

                    lambd = sig / cp.expand_dims(cp.real(var_current), axis=-1)

                    #print("lambd.shape {}".format(lambd.shape))


                    J_all = lambd ** 2 * cp.expand_dims(var_current, axis=-1) - 2 * lambd * sig
                    idx_max_all_current = cp.argmin(J_all, axis=0)
                    idx_max_all_unique_cluster[j_signal:j_signal_next] = idx_max_all_current
                        
                
                        # idx_max_all_unique = np.argmax(J_all, axis=0)
                    del J_all
                    
                idx_max_all_unique[indices.get().flatten()] = (retained_signals_indices[idx_max_all_unique_cluster]).get()




            print("Building the maps for iteration")

            # del sig_ws_all_unique
            # del sig_fs_all_unique

            keys=keys.get()
            keys_for_map = [tuple(k) for k in keys]

            params_all_unique = np.array(
                [keys_for_map[idx]  for l, idx in enumerate(idx_max_all_unique)])


            params_all = params_all_unique


            del params_all_unique

            map_rebuilt = {
                "wT1": params_all[:, 0],
                "fT1": params_all[:, 1],
                "attB1": params_all[:, 2],
                "df": params_all[:, 3],
                "ff": params_all[:, 4]

            }

            values_results.append((map_rebuilt, mask))

            
            return dict(zip(keys_results, values_results))