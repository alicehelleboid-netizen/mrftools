
try:
    import matplotlib.pyplot as plt
except:
    pass
import matplotlib.animation as animation
# from mutools.optim.dictsearch import dictsearch,groupmatch

from functools import reduce
# from mrfsim import *
import numpy as np
import finufft
from scipy import ndimage,signal
#from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.spatial import Voronoi,ConvexHull
from mrftools.Transformers import PCAComplex
import pickle
import twixtools
import os
from scipy.signal import savgol_filter
# from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
#from utils_reco import calculate_displacement,correct_mvt_kdata_zero_filled
#from trajectory import Navigator3D,Radial
import math
try:
    import freud
except:
    pass
try:

    import seaborn as sns
except:
    pass
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import pandas as pd
import itertools
try:
    from sigpy.mri import spiral
except:
    pass
import cv2
import pywt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import binary_erosion,erosion
import collections
import gc
try:
    import pycuda
    import pycuda.autoinit
    from pycuda.gpuarray import GPUArray, to_gpu
    from cufinufft import cufinufft

except:
    pass

from copy import copy,deepcopy
import psutil
from datetime import datetime
import scipy as sp
from mrftools.dictmodel import Dictionary

try:
    import cupy as cp
except:
    print("Could not import cupy - not using gpu")
    useGPU=False

# def read_mrf_dict(dict_file ,FF_list ,aggregate_components=True):

#     mrfdict = Dictionary()
#     mrfdict.load(dict_file, force=True)

#     if aggregate_components :
#         epg_water = mrfdict.values[: ,: ,0]
#         epg_fat = mrfdict.values[: ,: ,1]
#         ff = np.zeros(mrfdict.values.shape[:-1 ] +(len(FF_list),))
#         ff_matrix =np.tile(np.array(FF_list) ,ff.shape[:-1 ] +(1,))

#         water_signal =np.expand_dims(mrfdict.values[: ,: ,0] ,axis=-1 ) *(1-ff_matrix)
#         fat_signal =np.expand_dims(mrfdict.values[: ,: ,1] ,axis=-1 ) *(ff_matrix)

#         signal =water_signal +fat_signal

#         signal_reshaped =np.moveaxis(signal ,-1 ,-2)
#         signal_reshaped =signal_reshaped.reshape((-1 ,signal_reshaped.shape[-1]))

#         keys_with_ff = list(itertools.product(mrfdict.keys, FF_list))
#         keys_with_ff = [(*res, f) for res, f in keys_with_ff]

#         return keys_with_ff,signal_reshaped

#     else:
#         return mrfdict.keys,mrfdict.values

# def combine_mrf_dict_components(mrfdict ,FF_list ,aggregate_components=True):
#     '''
#     Combine the water and fat components of the dictionary with FF_list to create a single dictionary with all the FFs
#     '''
#     ff = np.zeros(mrfdict.values.shape[:-1 ] +(len(FF_list),))
#     ff_matrix =np.tile(np.array(FF_list) ,ff.shape[:-1 ] +(1,))

#     water_signal =np.expand_dims(mrfdict.values[: ,: ,0] ,axis=-1 ) *(1-ff_matrix)
#     fat_signal =np.expand_dims(mrfdict.values[: ,: ,1] ,axis=-1 ) *(ff_matrix)

#     signal =water_signal +fat_signal

#     signal_reshaped =np.moveaxis(signal ,-1 ,-2)
#     signal_reshaped =signal_reshaped.reshape((-1 ,signal_reshaped.shape[-1]))

#     keys_with_ff = list(itertools.product(mrfdict.keys, FF_list))
#     keys_with_ff = [(*res, f) for res, f in keys_with_ff]

#     return keys_with_ff,signal_reshaped

class Optimizer(object):

    def __init__(self,mask=None,verbose=False,useGPU=False,**kwargs):
        self.paramDict=kwargs
        self.paramDict["useGPU"]=useGPU
        self.mask=mask
        self.verbose=verbose


    def search_patterns(self,dictfile,volumes,retained_timesteps=None):
        #takes as input dictionary pattern and an array of images or volumes and outputs parametric maps
        raise ValueError("search_patterns should be implemented in child")
    
def match_signals_v2_clustered_on_dico(all_signals_current,keys,pca_water,pca_fat,transformed_array_water_unique,transformed_array_fat_unique,var_w_total,var_f_total,sig_wf_total,index_water_unique,index_fat_unique,useGPU_dictsearch,unique_keys,d_T1,d_fT1,d_B1,d_DF,labels,split,high_ff=False,return_cost=False):

    nb_clusters = unique_keys.shape[-1]

    # keys = np.array(keys)
    # keys = cp.asnumpy(keys) if 'cupy' in str(type(keys)) else np.asarray(keys)

    # unique_keys = np.array(unique_keys)
    # unique_keys = cp.asnumpy(unique_keys) if 'cupy' in str(type(unique_keys)) else np.asarray(unique_keys)


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
                print(mask.shape)
                print(volumes.shape)
                
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

def combine_mrf_dict_components(mrfdict, FF_list, aggregate_components=True):
    """
    Combine the water and fat components of the dictionary with FF_list
    to create a single dictionary with all the FFs efficiently.
    
    mrfdict.values shape: (n_voxels, n_timepoints, 2)  # last dim: water/fat
    FF_list: list of fat fractions (floats between 0 and 1)
    """
    FF_array = np.array(FF_list, dtype=np.float32)          # (nFF,)
    
    # Broadcasting instead of np.tile
    water_signal = mrfdict.values[:, :, 0, None] * (1 - FF_array)  # (n_voxels, n_timepoints, nFF)
    fat_signal   = mrfdict.values[:, :, 1, None] * FF_array        # (n_voxels, n_timepoints, nFF)
    signal = water_signal + fat_signal                               # same shape
    
    # Optionally reshape to 2D: (n_voxels * n_timepoints, nFF)
    signal_reshaped = np.moveaxis(signal, -1, -2)  # put nFF before timepoints
    signal_reshaped = signal_reshaped.reshape((-1, signal_reshaped.shape[-1]))
    
    # Generate keys with FF using itertools.product efficiently
    keys_with_ff = list(itertools.product(mrfdict.keys, FF_list))
    keys_with_ff = [(*res, f) for res, f in keys_with_ff]
    
    return signal_reshaped, keys_with_ff


def animate_images(images_series,interval=200,metric=np.abs,cmap=None):
    fig, ax = plt.subplots()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i, image in enumerate(images_series):

        im = ax.imshow(metric(image), animated=True,cmap=cmap)
        if i == 0:
            ax.imshow(metric(image),cmap=cmap)  # show an initial one first
        ims.append([im])

    return animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=10 * interval)

def animate_multiple_images(images_series,images_series_rebuilt,interval=200,cmap=None):
    nb_frames=len(images_series)
    fig, ax = plt.subplots()
    fig_rebuilt, ax_rebuilt = plt.subplots()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    ims_rebuilt = []
    for i in range(nb_frames):
        im = ax.imshow(np.abs(images_series[i]), animated=True,cmap=cmap)
        if i == 0:
            ax.imshow(np.abs(images_series[i]),cmap=cmap)  # show an initial one first
        ims.append([im])

        im_rebuilt = ax_rebuilt.imshow(np.abs(images_series_rebuilt[i]), animated=True,cmap=cmap)
        if i == 0:
            ax_rebuilt.imshow(np.abs(images_series_rebuilt[i]),cmap=cmap)  # show an initial one first
        ims_rebuilt.append([im_rebuilt])

    return animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=10 * interval),animation.ArtistAnimation(fig_rebuilt, ims_rebuilt, interval=interval, blit=True,
                                    repeat_delay=10 * interval),

def cartesian_traj_2D(npoint_x,npoint_y,k_max=np.pi):
    #kx = -k_max + np.arange(npoint_x) * 2 * k_max / (npoint_x - 1)
    #ky = -k_max + np.arange(npoint_y) * 2 * k_max / (npoint_y - 1)
    kx = np.arange(-k_max, k_max, 2 * k_max / npoint_x) + k_max / npoint_x
    ky = np.arange(-k_max, k_max, 2 * k_max / npoint_y) + k_max / npoint_y

    KX, KY = np.meshgrid(kx, ky)
    return np.stack([KX.flatten(), KY.flatten()], axis=-1)

def cartesian_traj_3D(total_nspoke, npoint_x, npoint_y, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    base_traj=cartesian_traj_2D(npoint_x,npoint_y)
    traj = np.tile(base_traj, (total_nspoke, 1, 1))

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])




def radial_golden_angle_traj(total_nspoke,npoint,k_max=np.pi):
    golden_angle=111.246*np.pi/180
    #base_spoke = np.arange(-k_max, k_max, 2 * k_max / npoint, dtype=np.complex_)
    #base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    base_spoke = (-k_max+k_max/(npoint)+np.arange(npoint)*2*k_max/(npoint))
    all_rotations = np.exp(1j * np.arange(total_nspoke) * golden_angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes

def distrib_angle_traj(total_nspoke,npoint,k_max=np.pi):
    angle=2*np.pi/total_nspoke
    #base_spoke = np.arange(-k_max, k_max, 2 * k_max / npoint, dtype=np.complex_)
    base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    all_rotations = np.exp(1j * np.arange(total_nspoke) * angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes


def radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,nb_rep_center_part=1):
    timesteps = int(total_nspoke / nspoke)
    print(total_nspoke)
    print(nspoke)


    nb_rep = math.ceil((nb_slices ) / undersampling_factor)+nb_rep_center_part-1
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    k_z = np.zeros((timesteps, nb_rep))
    #all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    

    k_z[0, :] = all_slices[::undersampling_factor]


    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    if nb_rep_center_part>1:
        center_part=all_slices[int(nb_slices/2)]
        k_z_new= np.zeros((timesteps, nb_rep))
        for j in range( k_z.shape[0]):
            num_center_part=np.argwhere(k_z[j]==center_part)[0][0]
            #print(num_center_part)
            k_z_new[j,:num_center_part]=k_z[j,:num_center_part]
            k_z_new[j,(num_center_part+nb_rep_center_part):]=k_z[j,(num_center_part+1):]
        print(k_z_new[0,:])
        k_z=k_z_new


    k_z=np.repeat(k_z, nspoke, axis=0)

    print(k_z.shape)
    k_z = np.expand_dims(k_z, axis=-1)


    traj = np.expand_dims(all_spokes, axis=-2)

    print(traj.shape)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    print(result.shape)
    return result.reshape(result.shape[0],-1,result.shape[-1])


def distrib_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = distrib_angle_traj(total_nspoke, npoint)
    #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    k_z = np.zeros((timesteps, nb_rep))
    #all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]

    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    traj = np.expand_dims(all_spokes, axis=-2)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def spherical_golden_angle_means_traj_3D(total_nspoke, npoint, nb_slices, undersampling_factor=4,k_max=np.pi):
    nb_rep = int(nb_slices / undersampling_factor)
    base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    base_spoke=base_spoke.reshape(1,-1)
    #traj = np.reshape(all_spokes, (-1, nspoke * npoint))
    phi_1=0.4656
    phi_2=0.6823
    all_spokes_count=nb_rep*total_nspoke
    m=np.arange(all_spokes_count)
    alpha=2*np.pi*np.mod(m*phi_1,1).reshape(-1,1)
    beta=np.arccos(np.mod(m*phi_2,1)).reshape(-1,1)
    traj=np.cos(beta)*np.exp(1j*alpha)*base_spoke
    k_z=np.sin(beta)*base_spoke
    print(k_z.shape)
    print(traj.shape)
    k_z, traj = np.broadcast_arrays(k_z, traj)
    result = np.stack([traj.real, traj.imag, k_z], axis=-1)
    #result=result.reshape(total_nspoke,nb_rep,npoint,3)
    #result=np.moveaxis(result,1,0)
    return result.reshape(total_nspoke,-1,3)



# def radial_golden_angle_traj_3D_incoherent(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,mode="old",offset=0):
#     timesteps = int(total_nspoke / nspoke)
#     nb_rep = int(nb_slices / undersampling_factor)
#     all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
#     golden_angle = 111.246 * np.pi / 180
#     if mode=="old":
#         all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
#     elif mode=="new":
#         all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
#     else:
#         raise ValueError("Unknown value for mode")
#     all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
#     traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes
#
#     k_z = np.zeros((timesteps, nb_rep))
#     all_slices = np.linspace(-np.pi, np.pi, nb_slices)
#     k_z[0, :] = all_slices[offset::undersampling_factor]
#     for j in range(1, k_z.shape[0]):
#         k_z[j, :] = np.sort(np.roll(all_slices, -j)[offset::undersampling_factor])
#
#     k_z=np.repeat(k_z, nspoke, axis=0)
#     k_z = np.expand_dims(k_z, axis=-1)
#     k_z, traj = np.broadcast_arrays(k_z, traj)
#
#     result = np.stack([traj.real,traj.imag, k_z], axis=-1)
#     return result.reshape(result.shape[0],-1,result.shape[-1])

def radial_golden_angle_traj_3D_incoherent(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=1,mode="old",offset=0):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = math.ceil(nb_slices / undersampling_factor)

    print(nb_rep)
    print(nb_slices)
    print(undersampling_factor)

    golden_angle = 111.246 * np.pi / 180
    #all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    all_slices = np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)

    # all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    # if mode=="old":
    #     all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
    # elif mode=="new":
    #     all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
    # else:
    #     raise ValueError("Unknown value for mode")
    # all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
    # traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes
    #
    # k_z = np.zeros((timesteps, nb_rep))
    # k_z[0, :] = all_slices[offset::undersampling_factor]
    # for j in range(1, k_z.shape[0]):
    #     k_z[j, :] = np.sort(np.roll(all_slices, -j)[offset::undersampling_factor])
    #
    # k_z=np.repeat(k_z, nspoke, axis=0)
    # k_z = np.expand_dims(k_z, axis=-1)
    # k_z, traj = np.broadcast_arrays(k_z, traj)
    #
    # result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    # return result.reshape(result.shape[0],-1,result.shape[-1])

    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    if mode=="old":
        all_rotations = np.exp(1j * np.arange(nb_slices) * total_nspoke * golden_angle)
    elif mode=="new":
        all_rotations = np.exp(1j * np.arange(nb_slices) * golden_angle)
    else:
        raise ValueError("Unknown value for mode")

    all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_slices, axis=1)
    traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes

    k_z=np.zeros((timesteps, nb_slices))
    k_z[0, :] = all_slices
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j))

    print(traj.shape)
    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)

    print(result.shape)

    if undersampling_factor>1:
        print(result.shape)
        result = result.reshape(timesteps, nspoke, -1, npoint, result.shape[-1])

        result_us=np.zeros((timesteps, nspoke, nb_rep, npoint, 3),
                          dtype=result.dtype)
        
        #result_us[:, :, :, :, 1:] = result[:, :, :nb_rep, :, 1:]
        #print(result_us.shape)
        shift = offset

        for sl in range(nb_slices):

            if int(sl/undersampling_factor)<nb_rep:
                result_us[shift::undersampling_factor, :, int(sl/undersampling_factor), :, :] = result[shift::undersampling_factor, :, sl, :, :]
                shift += 1
                shift = shift % (undersampling_factor)
            else:
                continue

        result=result_us


    return result.reshape(total_nspoke,-1,3)










def radial_golden_angle_traj_random_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,frac_center=0.25,mode="old",incoherent=True):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
        #traj = np.reshape(all_spokes, (-1, nspoke * npoint))

    if incoherent:
        golden_angle = 111.246 * np.pi / 180
        if mode=="old":
            all_rotations = np.exp(1j * np.arange(nb_rep) * total_nspoke * golden_angle)
        elif mode=="new":
            all_rotations = np.exp(1j * np.arange(nb_rep) * golden_angle)
        else:
            raise ValueError("Unknown value for mode")
        all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_rep, axis=1)
        traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes
    else:
        traj = np.expand_dims(all_spokes, axis=-2)

    k_z = np.zeros((timesteps, nb_rep))
    all_slices = np.linspace(-np.pi, np.pi, nb_slices)
    kz_center=all_slices[(int(nb_slices/2)+np.array(range(int(-frac_center*nb_rep/2),int(frac_center*nb_rep/2),1)))]
    kz_border = [k for k in all_slices if k not in kz_center]
    nb_border=nb_rep-len(kz_center)
    for j in range(k_z.shape[0]):
        k_z[j, :] = np.sort(np.concatenate([np.random.choice(kz_border,size=int(nb_border),replace=False),kz_center]))

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)

    k_z, traj = np.broadcast_arrays(k_z, traj)

    # k_z = np.reshape(k_z, (timesteps, -1))
    # traj = np.reshape(traj, (timesteps, -1))

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def spiral_golden_angle_traj(total_spiral,fov, N, f_sampling, R, ninterleaves, alpha, gm, sm):
    golden_angle = 111.246 * np.pi / 180
    base_spiral = spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
    base_spiral = base_spiral[:,0]+1j*base_spiral[:,1]
    all_rotations = np.exp(1j * np.arange(total_spiral) * golden_angle)
    all_spirals = np.matmul(np.diag(all_rotations), np.repeat(base_spiral.reshape(1, -1), total_spiral, axis=0))
    return all_spirals

def spiral_golden_angle_traj_v2(total_spiral,nspiral,fov, N, f_sampling, R, ninterleaves, alpha, gm, sm):
    golden_angle = 111.246 * np.pi / 180
    angle = 2*np.pi/nspiral
    base_spiral = spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm)
    base_spiral = base_spiral[:,0]+1j*base_spiral[:,1]
    all_rotations = np.exp(1j * np.arange(total_spiral) * angle)
    all_spirals = np.matmul(np.diag(all_rotations), np.repeat(base_spiral.reshape(1, -1), total_spiral, axis=0))

    disk_rotations = np.exp(1j * np.arange(int(total_spiral/nspiral)) * golden_angle)
    disk_rotations = np.repeat(disk_rotations,nspiral)
    all_spirals = np.matmul(np.diag(disk_rotations),all_spirals )

    return all_spirals

def create_random_map(list_params,region_size,size,mask):
    basis = np.random.choice(list_params,(int(size[0]/region_size),int(size[1]/region_size)))
    map = np.repeat(np.repeat(basis, region_size, axis=1), region_size, axis=0) * mask
    return map

def create_map(list_params,region_size,mask):
    map = np.repeat(np.repeat(list_params, region_size, axis=1), region_size, axis=0) * mask
    return map

def compare_patterns(pixel_number,images_1,images_2,title_1="image_1",title_2="image_2"):

    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.plot(np.real(images_1[:,pixel_number[0],pixel_number[1]]),label=title_1+" - real part")
    ax1.plot(np.real(images_2[:, pixel_number[0],pixel_number[1]]), label=title_2+" - real part")
    ax1.legend()
    ax2.plot(np.imag(images_1[:, pixel_number[0], pixel_number[1]]),
             label=title_1+" - imaginary part")
    ax2.plot(np.imag(images_2[:, pixel_number[0], pixel_number[1]]),
             label=title_2+" - imaginary part")
    ax2.legend()

    ax3.plot(np.abs(images_1[:, pixel_number[0], pixel_number[1]]),
             label=title_1+" - norm")
    ax3.plot(np.abs(images_2[:, pixel_number[0], pixel_number[1]]),
             label=title_2+" - norm")
    ax3.legend()

    plt.show()


def translation_breathing(t,direction,T=4000,frac_expiration=0.7):
    def base_pattern(t):
        lambda1=5/(frac_expiration*T)
        lambda2=20/((1-frac_expiration)*T)

        return ((1-np.exp(-lambda1*t))*((t<(frac_expiration*T))*1) + ((t>=(frac_expiration*T))*1)*((1-np.exp(-lambda1*frac_expiration*T))* np.exp(-lambda2*t)/np.exp(-lambda2*frac_expiration*T)))*direction

    return base_pattern(t-(t/T).astype(int)*T)

def find_klargest_freq(ft, k=1, remove_central_peak=True):
    n_max = len(ft) - 1
    n_min = 0
    if remove_central_peak:
        # Removing the central peak in fourier transform (corresponds roughly to PSF)
        while (ft[n_max - 1] <= ft[n_max]):
            n_max = n_max - 1

        while (ft[n_min + 1] <= ft[n_min]):
            n_min = n_min + 1

    freq_image = np.argsort((ft)[n_min:n_max])[-k]

    if freq_image + n_min < 175 / 2:
        freq_image = freq_image + n_min
    else:
        freq_image = len(ft) - 1 - (freq_image + n_min)

    return freq_image

def SearchMrf(kdata,trajectory, dictfile, niter, method, metric, shape,density_adj=False, setup_opts={}, search_opts= {}):
    """ Estimate parameters """
    # constants
    shape = tuple(shape)

    nspoke=trajectory.paramDict["nspoke"]
    npoint=trajectory.paramDict["npoint"]
    traj = trajectory.get_traj()

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
    else:
        density=np.ones(npoint)

    # printer(f"Load dictionary: {dictfile}")
    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dictfile, force=True)

    print(f"Init solver ({method})")
    if method == "brute":
        solver = dictsearch.DictSearch()
        setupopts = {"pca": True, **parse_options(setup_opts)}
        searchopts = {"metric": metric, "parallel": True, **parse_options(search_opts)}
    elif method == "group":
        solver = groupmatch.GroupMatch()
        setupopts = {"pca": True, "group_ratio": 0.05, **parse_options(setup_opts)}
        searchopts = {"metric": metric, "parallel": True, "group_threshold": 1e-1, **parse_options(search_opts)}
    solver.setup(mrfdict.keys, mrfdict.values, **setupopts)

    # group trajectories and kspace
    #traj = np.reshape(groupby(traj, nspoke), (-1, npoint * nspoke))
    kdata = np.array([(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata])

    #kdata = np.reshape(groupby(kdata * density, nspoke), (-1, npoint * nspoke))

    print(f"Build volumes ({nspoke} groups)")

    # NUFFT
    kdata /= np.sum(np.abs(kdata)**2)**0.5 / len(kdata)
    volumes = [
        finufft.nufft2d1(t.real, t.imag, s, shape)
        for t, s in zip(traj, kdata)
    ]

    # init mask
    mask = False
    volumes0 = volumes
    kdata0 = kdata
    info = {}
    for i in range(niter + 1):

        # auto mask
        unique = np.histogram(np.abs(volumes), 100)[1]
        mask = mask | (np.mean(np.abs(volumes), axis=0) > unique[len(unique) // 10])
        mask = ndimage.binary_closing(mask, iterations=3)

        print(f"Search data (iteration {i})")
        obs = np.transpose([vol[mask] for vol in volumes])
        res = solver.search(obs, **searchopts)

        info[f"iteration {i}"] = solver.info

        if i == niter:
            break

        # generate prediction volumes
        pred = np.asarray(solver.predict(res)).T

        # predict spokes
        kdata = [
            finufft.nufft2d2(t.real, t.imag, makevol(p, mask))
            for t, p in zip(traj, pred)
        ]
        kdatai = np.array([(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata])

        # NUFFT
        kdatai /= np.sum(np.abs(kdatai)**2)**0.5 / len(kdatai)
        volumesi = [
            finufft.nufft2d1(t.real, t.imag, s, shape)
            for t, s in zip(traj, kdatai)
        ]

        # correct volumes
        volumes = [2 * vol0 - voli for vol0, voli in zip(volumes0, volumesi)]


    # make maps
    wt1map = makevol([p[0] for p in res.parameters], mask)
    ft1map = makevol([p[1] for p in res.parameters], mask)
    b1map = makevol([p[2] for p in res.parameters], mask)
    dfmap = makevol([p[3] for p in res.parameters], mask)
    wmap =  makevol([s[0] for s in res.scales], mask)
    fmap =  makevol([s[1] for s in res.scales], mask)
    ffmap = makevol([s[1]/(s[0] + s[1]) for s in res.scales], mask)



    return {
        "mask": mask,
        "wt1map": wt1map,
        "ft1map": ft1map,
        "b1map": b1map,
        "dfmap": dfmap,
        "wmap": wmap,
        "fmap": fmap,
        "ffmap": ffmap,
        "info": {"search": info, "options": solver.options},
    }

def basicDictSearch(all_signals,dictfile):
    #Basic dic search with component separation
    #
    mrfdict = dictsearch.Dictionary()
    mrfdict.load(dictfile, force=True)

    array_water = mrfdict.values[:, :, 0]
    array_fat = mrfdict.values[:, :, 1]

    var_w = np.sum(array_water * array_water.conj(), axis=1).real
    var_f = np.sum(array_fat * array_fat.conj(), axis=1).real
    sig_wf = np.sum(array_water * array_fat.conj(), axis=1).real

    print("Removing duplicate dictionary entries and signals")
    array_water_unique, index_water_unique = np.unique(array_water, axis=0, return_inverse=True)
    array_fat_unique, index_fat_unique = np.unique(array_fat, axis=0, return_inverse=True)
    all_signals_unique, index_signals_unique = np.unique(all_signals, axis=1, return_inverse=True)

    print("Calculating correlations")
    sig_ws_all_unique = np.matmul(array_water_unique, all_signals_unique[:, :].conj()).real
    sig_fs_all_unique = np.matmul(array_fat_unique, all_signals_unique[:, :].conj()).real
    sig_ws_all = sig_ws_all_unique[index_water_unique, :]
    sig_fs_all = sig_fs_all_unique[index_fat_unique, :]

    print("Calculating optimal fat fraction and best pattern per signal")
    var_w = np.reshape(var_w, (-1, 1))
    var_f = np.reshape(var_f, (-1, 1))
    sig_wf = np.reshape(sig_wf, (-1, 1))

    alpha_all_unique = (sig_wf * sig_ws_all-var_w*sig_fs_all) / ((sig_ws_all + sig_fs_all) * sig_wf-var_w*sig_fs_all-var_f*sig_ws_all)
    one_minus_alpha_all = 1 - alpha_all_unique
    J_all = (one_minus_alpha_all * sig_ws_all + alpha_all_unique * sig_fs_all) / np.sqrt(
        one_minus_alpha_all ** 2 * var_w + alpha_all_unique ** 2 * var_f + 2 * alpha_all_unique * one_minus_alpha_all * sig_wf)
    idx_max_all_unique = np.argmax(J_all, axis=0)
    #idx_max_all = idx_max_all_unique[index_signals_unique]

    print("Building the maps")
    #alpha_all = alpha_all_unique[:, index_signals_unique]

    params_all_unique = np.array([mrfdict.keys[idx] + (alpha_all_unique[idx, i],) for i, idx in enumerate(idx_max_all_unique)])
    params_all = params_all_unique[index_signals_unique]



    return {
            "wT1": params_all[:, 0],
            "fT1": params_all[:, 1],
            "attB1": params_all[:, 2],
            "df": params_all[:, 3],
            "ff": params_all[:, 4]

            }







def compare_paramMaps(map1,map2,mask1,mask2=None,fontsize=5,title1="Orig Map",title2="Rebuilt Map",adj_wT1=False,fat_threshold=0.8,proj_on_mask1=False,save=False,figsize=(30,10),units=None,vmax_error=None,extent=None,kept_keys=None):
    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())

    if kept_keys is not None:
        keys_1 = keys_1 & set(kept_keys)
        keys_2 = keys_2 & set(kept_keys)
    if mask2 is None:
        mask2 = mask1
    for k in (keys_1 & keys_2):
        fig,axes=plt.subplots(1,3,figsize=figsize)
        vol1 = makevol(map1[k],mask1)
        vol2= makevol(map2[k],mask2)

        if proj_on_mask1 is not None:
            if type(proj_on_mask1) is bool:#Projection on mask 1
                vol2=vol2*(mask1*1)
            else:#projection on external mask
                vol1 = vol1*proj_on_mask1
                vol2 = vol2*proj_on_mask1

        if adj_wT1 and k=="wT1":
            ff = makevol(map2["ff"],mask2)
            vol2[ff>fat_threshold]=vol1[ff>fat_threshold]

        error=vol2-vol1

        if units is None:
            graph_title1 = title1+" {}".format(k)
            graph_title2 = title2 + " {}".format(k)
            error_title = "Error {}".format(k)
        else:
            graph_title1 = title1+" {} ({})".format(k,units[k])
            graph_title2 = title2 + " {} ({})".format(k, units[k])
            error_title = "Error {} ({})".format(k,units[k])


        if extent is not None:
            centrum_x=int(vol1.shape[0]/2)
            centrum_y = int(vol2.shape[1] / 2)
            vol1=vol1[centrum_x-extent:centrum_x+extent,centrum_y-extent:centrum_y+extent]
            vol2 = vol2[centrum_x - extent:centrum_x + extent, centrum_y - extent:centrum_y + extent]
            error = error[centrum_x - extent:centrum_x + extent, centrum_y - extent:centrum_y + extent]

        minmin=np.min([np.min(vol1),np.min(vol2)])
        maxmax=np.max([np.max(vol1),np.max(vol2)])

        im1=axes[0].imshow(vol1,vmin=minmin,vmax=maxmax,aspect="auto")
        axes[0].set_title(graph_title1,fontdict={"fontsize":fontsize})
        axes[0].xaxis.set_visible(False)
        axes[0].yaxis.set_visible(False)
        #cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        #cbar1.ax.tick_params(labelsize=fontsize)

        im2 = axes[1].imshow(vol2,vmin=minmin,vmax=maxmax,aspect="auto")
        axes[1].set_title(graph_title2,fontdict={"fontsize":fontsize})
        axes[1].xaxis.set_visible(False)
        axes[1].yaxis.set_visible(False)


        if (vmax_error is None):
            im3 = axes[2].imshow(error,aspect="auto")
        else:
            im3 = axes[2].imshow(error,vmin=-vmax_error[k],vmax=vmax_error[k],aspect="auto")
        axes[2].set_title(error_title,fontdict={"fontsize":fontsize})
        axes[2].xaxis.set_visible(False)
        axes[2].yaxis.set_visible(False)

        #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
        #                    wspace=0.08, hspace=0.02)

        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=fontsize)

        cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=fontsize)
        if save:
            fig.savefig("./figures/{}_vs_{}_{}".format(title1,title2,k))

def compare_paramMaps_3D(map1,map2,mask1,mask2=None,slice=0,fontsize=5,title1="Orig Map",title2="Rebuilt Map",adj_wT1=False,fat_threshold=0.8,proj_on_mask1=False,save=False):
    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    if mask2 is None:
        mask2 = mask1
    for k in (keys_1 & keys_2):
        fig,axes=plt.subplots(1,3)
        vol1 = makevol(map1[k],mask1)[slice,:,:]
        vol2= makevol(map2[k],mask2)[slice,:,:]
        if proj_on_mask1:
            vol2=vol2*(mask1[slice,:,:]*1)
        if adj_wT1 and k=="wT1":
            ff = makevol(map2["ff"],mask2)[slice,:,:]
            vol2[ff>fat_threshold]=vol1[ff>fat_threshold]

        error=(vol1-vol2)

        im1=axes[0].imshow(vol1)
        axes[0].set_title(title1+" "+k)
        axes[0].tick_params(axis='x', labelsize=fontsize)
        axes[0].tick_params(axis='y', labelsize=fontsize)
        cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=fontsize)

        im2 = axes[1].imshow(vol2)
        axes[1].set_title(title2+" "+k)
        axes[1].tick_params(axis='x', labelsize=fontsize)
        axes[1].tick_params(axis='y', labelsize=fontsize)
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=fontsize)

        im3 = axes[2].imshow(error)
        axes[2].set_title("Error {}".format(k))
        axes[2].tick_params(axis='x', labelsize=fontsize)
        axes[2].tick_params(axis='y', labelsize=fontsize)
        cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=fontsize)
        if save:
            plt.savefig("./figures/{}_vs_{}_Slice_{}_{}".format(title1,title2,slice,k))

def regression_paramMaps(map1,map2,mask1=None,mask2=None,title="Maps regression plots",fontsize=5,adj_wT1=False,fat_threshold=0.8,mode="Standard",proj_on_mask1=False,save=False):

    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    nb_keys=len(keys_1 & keys_2)
    fig,ax = plt.subplots(1,nb_keys)

    for i,k in enumerate(keys_1 & keys_2):
        obs = map1[k]
        pred = map2[k]

        if mask1 is not None:
            if mask2 is None:
                mask2 = mask1
            mask_union = mask1 | mask2
            mat_obs = makevol(map1[k],mask1)
            mat_pred = makevol(map2[k],mask2)
            if proj_on_mask1:
                mat_pred = mat_pred*(mask1*1)

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]

        if adj_wT1 and k=="wT1":
            ff = map2["ff"]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]

        x_min = np.min(obs)
        x_max = np.max(obs)

        if x_min==x_max:
            fig.delaxes(ax[i])
            continue

        mean=np.mean(obs)
        ss_tot = np.sum((obs-mean)**2)
        ss_res = np.sum((obs-pred)**2)
        bias = np.mean((pred-obs))
        r_2 = 1-ss_res/ss_tot

        dx = (x_max - x_min) / 10
        x_ = np.arange(x_min, x_max+dx,dx )

        if mode=="Standard":
            ax[i].scatter(obs,pred,s=1)
            ax[i].plot(x_, x_, "r")
        elif mode=="Boxplot":
            unique_obs=np.unique(obs)
            sns.boxplot(ax=ax[i],x=obs,y=pred)
            locs=ax[i].get_xticks()
            print(locs)
            sns.lineplot(ax=ax[i],x=locs,y=unique_obs)
        else:
            raise ValueError("mode should be Standard/Boxplot")

        ax[i].set_title(k+" R2:{} Bias:{}".format(np.round(r_2,2),np.round(bias,2)),fontsize=2*fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].tick_params(axis='y', labelsize=fontsize)

    plt.suptitle(title)

    if save:
        plt.savefig("./figures/{}".format(title))


def regression_paramMaps_ROI(map1, map2, mask1=None, mask2=None, maskROI=None, title="Maps regression plots",
                             fontsize=5, adj_wT1=False, fat_threshold=0.8, mode="Standard", proj_on_mask1=True,plt_std=False,
                             figsize=(20, 5),save=False,kept_keys=None,min_ROI_count=15,units=None,fontsize_axis=None,marker_size=1):

    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    if kept_keys is not None:
        keys_1 = keys_1 & set(kept_keys)
        keys_2 = keys_2 & set(kept_keys)

    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map1)

    #Removing ROIs with less than min_ROI_count values
    freq = collections.Counter(maskROI)
    maskROI = np.array([ele if freq[ele] > min_ROI_count else 0 for ele in maskROI])

    fig, ax = plt.subplots(1, nb_keys, figsize=figsize)

    for i, k in enumerate(sorted(keys_1 & keys_2)):
        obs = map1[k]
        pred = map2[k]

        if mask1 is not None:
            if mask2 is None:
                mask2 = mask1
            mask_union = mask1 | mask2
            mat_obs = makevol(map1[k], mask1)
            mat_pred = makevol(map2[k], mask2)
            mat_ROI = makevol(maskROI, mask1)
            if proj_on_mask1 is not None:
                if type(proj_on_mask1) is bool:#Projection on mask1
                    mat_pred = mat_pred * (mask1 * 1)
                    mat_ROI = mat_ROI * (mask1 * 1)
                    mat_obs = mat_obs * (mask1 * 1)
                    mask_union = mask1

                else : #projection on externally provided mask
                    mat_pred = mat_pred * (proj_on_mask1 * 1)
                    mat_ROI = mat_ROI * (proj_on_mask1 * 1)
                    mat_obs = mat_obs * (proj_on_mask1 * 1)
                    mask_union = proj_on_mask1

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]
            maskROI_current = mat_ROI[mask_union]

            # print(obs)

        if adj_wT1 and k == "wT1":
            ff = makevol(map1["ff"], mask1)
            ff = ff[mask_union]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]
            maskROI_current = maskROI_current[ff < fat_threshold]

        df_obs = pd.DataFrame(columns=["Data", "Groups"],
                              data=np.stack([obs.flatten(), maskROI_current.flatten()], axis=-1))
        df_pred = pd.DataFrame(columns=["Data", "Groups"],
                               data=np.stack([pred.flatten(), maskROI_current.flatten()], axis=-1))
        obs = np.array(df_obs.groupby("Groups").mean())[1:]
        pred = np.array(df_pred.groupby("Groups").mean())[1:]
        # obs_std = np.array(df_obs.groupby("Groups").std())[1:]

        # print(list(pred_std.reshape(1,-1)))

        x_min = np.min(obs)
        x_max = np.max(obs)

        if x_min == x_max:
            fig.delaxes(ax[i])
            continue

        mean = np.mean(obs)
        ss_tot = np.sum((obs - mean) ** 2)
        ss_res = np.sum((obs - pred) ** 2)
        bias = np.mean((pred - obs))
        r_2 = 1 - ss_res / ss_tot

        dx = (x_max - x_min) / 10
        x_ = np.arange(x_min, x_max + dx, dx)

        if mode == "Standard":
            if plt_std:
                pred_std = np.array(df_pred.groupby("Groups").std())[1:]
                ax[i].errorbar(obs, pred, list(pred_std.flatten()), linestyle='None', marker='^')
            else:
                ax[i].scatter(obs,pred,s=marker_size)
            ax[i].plot(x_, x_, "r")
            ax[i].set_xlabel('obs')
            ax[i].set_ylabel('pred')
        elif mode == "Boxplot":
            unique_obs = np.unique(obs)
            sns.boxplot(ax=ax[i], x=obs, y=pred)
            locs = ax[i].get_xticks()
            sns.lineplot(ax=ax[i], x=locs, y=unique_obs)
        else:
            raise ValueError("mode should be Standard/Boxplot")
        if units is None:
            graph_title = k + " R2:{} Bias:{}".format(np.round(r_2, 4), np.round(bias, 3))
        else:
            graph_title = k + " R2:{} Bias:{} ({})".format(np.round(r_2, 4), np.round(bias, 3), units[k])

        ax[i].set_title(graph_title, fontsize=2 * fontsize)

        if fontsize_axis is None:
            ax[i].tick_params(axis='x', labelsize=fontsize)
            ax[i].tick_params(axis='y', labelsize=fontsize)

        else:
            ax[i].tick_params(axis='x', labelsize=fontsize_axis)
            ax[i].tick_params(axis='y', labelsize=fontsize_axis)


    plt.suptitle(title)
    if save:
        fig.savefig("./{}.jpeg".format(title))


def process_ROI_values(all_results, save=False,mode="Standard",title="Results comparison all ROIs",plt_std=False,figsize=(15, 10),fontsize=5,fontsize_axis=None,units=None):

    nb_keys=len(all_results.keys())
    fig, ax = plt.subplots(1, nb_keys, figsize=figsize)

    for i,k in enumerate(all_results.keys()):
        obs=all_results[k][:,0]
        pred = all_results[k][:, 1]

        x_min = np.min(obs)
        x_max = np.max(obs)

        if x_min == x_max:
            fig.delaxes(ax[i])
            continue

        mean = np.mean(obs)
        ss_tot = np.sum((obs - mean) ** 2)
        ss_res = np.sum((obs - pred) ** 2)
        bias = np.mean((pred - obs))
        r_2 = 1 - ss_res / ss_tot

        dx = (x_max - x_min) / 10
        x_ = np.arange(x_min, x_max + dx, dx)

        if mode == "Standard":
            if plt_std:
                pred_std = np.array(df_pred.groupby("Groups").std())[1:]
                ax[i].errorbar(obs, pred, list(pred_std.flatten()), linestyle='None', marker='^')
            else:
                ax[i].scatter(obs, pred, s=1)
            ax[i].plot(x_, x_, "r")
        elif mode == "Boxplot":
            unique_obs = np.unique(obs)
            sns.boxplot(ax=ax[i], x=obs, y=pred)
            locs = ax[i].get_xticks()
            sns.lineplot(ax=ax[i], x=locs, y=unique_obs)
        else:
            raise ValueError("mode should be Standard/Boxplot")

        if units is None:
            graph_title = k + " R2:{} Bias:{}".format(np.round(r_2, 4), np.round(bias, 3))
        else :
            graph_title = k + " R2:{} Bias:{} ({})".format(np.round(r_2, 4), np.round(bias, 3),units[k])
        ax[i].set_title(graph_title, fontsize=2 * fontsize)

        if fontsize_axis is None:
            ax[i].tick_params(axis='x', labelsize=fontsize)
            ax[i].tick_params(axis='y', labelsize=fontsize)
        else:
            ax[i].tick_params(axis='x', labelsize=fontsize_axis)
            ax[i].tick_params(axis='y', labelsize=fontsize_axis)

    plt.suptitle(title)
    if save:
        plt.savefig("./figures/{}".format(title))

def metrics_ROI_values(all_results,units=None,name="Results"):

    nb_keys=len(all_results.keys())
    df = pd.DataFrame(columns=[name])
    for i,k in enumerate(all_results.keys()):
        obs=all_results[k][:,0]
        pred = all_results[k][:, 1]


        mean = np.mean(obs)
        ss_tot = np.sum((obs - mean) ** 2)
        ss_res = np.sum((obs - pred) ** 2)
        bias = np.mean((pred - obs))
        r_2 = 1 - ss_res / ss_tot

        error = np.linalg.norm(obs-pred)/np.sqrt(len(obs))

        if units is None:
            r2_label = "R2 {}".format(k)
            bias_label = "Bias {}".format(k)
            rmse_label ="RMSE {}".format(k)

        else :
            r2_label = "R2 {} (a.u)".format(k)
            bias_label = "Bias {} ({})".format(k,units[k])
            rmse_label = "RMSE {} ({})".format(k,units[k])

        df=df.append(pd.DataFrame(columns=[name],index=[r2_label],data=r_2))
        df=df.append(pd.DataFrame(columns=[name], index=[bias_label], data=bias))
        df=df.append(pd.DataFrame(columns=[name], index=[rmse_label], data=error))

    return df

def compare_ROI_values(map1, map2, mask1=None, mask2=None, maskROI=None, adj_wT1=False, fat_threshold=0.8,proj_on_mask1=True,return_std=False,
                             kept_keys=None,min_ROI_count=15):

    keys_1 = set(map1.keys())
    keys_2 = set(map2.keys())
    if kept_keys is not None:
        keys_1 = keys_1 & set(kept_keys)
        keys_2 = keys_2 & set(kept_keys)

    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map1)

    #Removing ROIs with less than min_ROI_count values
    freq = collections.Counter(maskROI)
    maskROI = np.array([ele if freq[ele] > min_ROI_count else 0 for ele in maskROI])


    results={}
    for i, k in enumerate(sorted(keys_1 & keys_2)):
        obs = map1[k]
        pred = map2[k]

        if mask1 is not None:
            if mask2 is None:
                mask2 = mask1
            mask_union = mask1 | mask2
            mat_obs = makevol(map1[k], mask1)
            mat_pred = makevol(map2[k], mask2)
            mat_ROI = makevol(maskROI, mask1)
            if proj_on_mask1 is not None:
                if type(proj_on_mask1) is bool:#Projection on mask1
                    mat_pred = mat_pred * (mask1 * 1)
                    mat_ROI = mat_ROI * (mask1 * 1)
                    mat_obs = mat_obs * (mask1 * 1)
                    mask_union = mask1

                else : #projection on externally provided mask
                    mat_pred = mat_pred * (proj_on_mask1 * 1)
                    mat_ROI = mat_ROI * (proj_on_mask1 * 1)
                    mat_obs = mat_obs * (proj_on_mask1 * 1)
                    mask_union = proj_on_mask1

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]
            maskROI_current = mat_ROI[mask_union]

            # print(obs)

        if adj_wT1 and k == "wT1":
            ff = makevol(map1["ff"], mask1)
            ff = ff[mask_union]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]
            maskROI_current = maskROI_current[ff < fat_threshold]

        df_obs = pd.DataFrame(columns=["Data", "Groups"],
                              data=np.stack([obs.flatten(), maskROI_current.flatten()], axis=-1))
        df_pred = pd.DataFrame(columns=["Data", "Groups"],
                               data=np.stack([pred.flatten(), maskROI_current.flatten()], axis=-1))

        obs = np.array(df_obs.groupby("Groups").mean())[1:]
        pred = np.array(df_pred.groupby("Groups").mean())[1:]

        if return_std:
            obs_std = np.array(df_obs.groupby("Groups").std())[1:]
            pred_std=np.array(df_pred.groupby("Groups").std())[1:]

            results[k] = pd.DataFrame(data=np.concatenate([obs,obs_std, pred,pred_std], axis=1),columns=["Obs Mean","Obs Std","Pred Mean","Pred Std"])


        else:
            # print(list(pred_std.reshape(1,-1)))
            results[k]=pd.DataFrame(data=np.concatenate([obs, pred], axis=1),columns=["Obs Mean","Pred Mean"])

    return results

def metrics_paramMaps_ROI(map_ref, map2, mask_ref=None, mask2=None, maskROI=None,
                              adj_wT1=False, fat_threshold=0.8, proj_on_mask1=True,name="Result",min_ROI_count=15,units=None
                             ):

    df = pd.DataFrame(columns=[name])

    keys_1 = set(map_ref.keys())
    keys_2 = set(map2.keys())
    nb_keys = len(keys_1 & keys_2)

    if maskROI is None:
        maskROI = buildROImask(map_ref)

    # Removing ROIs with less than min_ROI_count values
    freq = collections.Counter(maskROI)
    maskROI = np.array([ele if freq[ele] > min_ROI_count else 0 for ele in maskROI])

    for i, k in enumerate(keys_1 & keys_2):
        print(i)

        mask_union = mask_ref | mask2
        mat_obs = makevol(map_ref[k], mask_ref)
        mat_pred = makevol(map2[k], mask2)
        mat_ROI = makevol(maskROI, mask_ref)
        if proj_on_mask1 is not None:
            if type(proj_on_mask1) is bool:  # Projection on mask1
                mat_pred = mat_pred * (mask_ref * 1)
                mat_ROI = mat_ROI * (mask_ref * 1)
                mat_obs = mat_obs * (mask_ref * 1)
                mask_union = mask_ref

            else:  # projection on externally provided mask
                mat_pred = mat_pred * (proj_on_mask1 * 1)
                mat_ROI = mat_ROI * (proj_on_mask1 * 1)
                mat_obs = mat_obs * (proj_on_mask1 * 1)
                mask_union = proj_on_mask1

        obs = mat_obs[mask_union]
        pred = mat_pred[mask_union]
        maskROI_current = mat_ROI[mask_union]

        if adj_wT1 and k == "wT1":
            ff = makevol(map_ref["ff"], mask_ref)
            ff = ff[mask_union]
            obs = obs[ff < fat_threshold]
            pred = pred[ff < fat_threshold]
            maskROI_current = maskROI_current[ff < fat_threshold]

        df_all = pd.DataFrame(columns=["Data_Obs", "Data_Pred", "Groups"],
                              data=np.stack([obs.flatten(), pred.flatten(), maskROI_current.flatten()], axis=-1))

        ssim_values = df_all.groupby("Groups").apply(lambda x: ssim(x.Data_Obs, x.Data_Pred))
        mean_ssim = ssim_values.mean()
        std_ssim = ssim_values.std()

        mean_obs = np.array(df_all[["Data_Obs","Groups"]].groupby("Groups").mean())[1:]
        mean_pred = np.array(df_all[["Data_Pred", "Groups"]].groupby("Groups").mean())[1:]


        x_min = np.min(mean_obs)
        x_max = np.max(mean_obs)

        if x_min == x_max:
            continue

        mean = np.mean(mean_obs)
        ss_tot = np.sum((mean_obs - mean) ** 2)
        ss_res = np.sum((mean_obs - mean_pred) ** 2)
        bias = np.mean((mean_pred - mean_obs))
        r_2 = 1 - ss_res / ss_tot


        df_error = pd.DataFrame(columns=["Data", "Groups"],
                                data=np.stack(
                                    [(pred.flatten() - obs.flatten()) ** 2, maskROI_current.flatten()],
                                    axis=-1))
        errors = np.sqrt(np.array(df_error.groupby("Groups").mean())[1:])
        error = np.mean(errors)
        std_error = np.std(errors)

        if units is None:
            r2_label = "R2 {}".format(k)
            bias_label = "Bias {}".format(k)
            rmse_label ="mean RMSE {}".format(k)
            std_rmse_label = "std RMSE {}".format(k)
            ssim_label = "mean SSIM {}".format(k)
            std_ssim_label ="std SSIM {}".format(k)
        else :
            r2_label = "R2 {} (a.u)".format(k)
            bias_label = "Bias {} ({})".format(k,units[k])
            rmse_label = "mean RMSE {} ({})".format(k,units[k])
            std_rmse_label = "std RMSE {} ({})".format(k, units[k])
            ssim_label = "mean SSIM {} (a.u)".format(k)
            std_ssim_label = "std SSIM {} (a.u)".format(k)

        df=df.append(pd.DataFrame(columns=[name],index=[r2_label],data=r_2))
        df=df.append(pd.DataFrame(columns=[name], index=[bias_label], data=bias))
        df=df.append(pd.DataFrame(columns=[name], index=[rmse_label], data=error))
        df=df.append(pd.DataFrame(columns=[name], index=[std_rmse_label], data=std_error))
        df = df.append(pd.DataFrame(columns=[name], index=[ssim_label], data=mean_ssim))
        df = df.append(pd.DataFrame(columns=[name], index=[std_ssim_label], data=std_ssim))

    return df


def get_ROI_values(map_,mask,maskROI,kernel_size=5,adj_wT1=False, fat_threshold=0.7,wT1_threshold=1700,kept_keys=None,min_ROI_count=15,return_std=True,excluded_border_slices=2):

    keys = set(map_.keys())
    if kept_keys is not None:
        keys = keys & set(kept_keys)
    
    nb_keys = len(keys)

    print(maskROI.shape)
    print(mask.shape)

    if mask.ndim==2:
        excluded_border_slices=None
        final_mask=mask[None,...]
        for k in keys:
            curr_map = makevol(map_[k], mask > 0)
            curr_map=curr_map[None,...]
            map_[k]=curr_map[final_mask>0]
        mask=final_mask

    if excluded_border_slices is not None:
        print("Excluding {} border slices on each extremity".format(excluded_border_slices))
        final_mask=mask[excluded_border_slices:-excluded_border_slices]
        maskROI=maskROI[excluded_border_slices:-excluded_border_slices]
        for k in keys:
            curr_map=makevol(map_[k],mask>0)
            map_[k]=curr_map[excluded_border_slices:-excluded_border_slices][final_mask>0]
        mask=final_mask


    if kernel_size is not None:
        print("Eroding ROIs kernel size {}".format(kernel_size))
        mask_from_roi=(maskROI>0)*1
        mask_from_roi_eroded=np.zeros_like(mask_from_roi)
        for sl in range(mask_from_roi.shape[0]):
            mask_from_roi_eroded[sl]=erosion(mask_from_roi[sl],footprint=np.ones((kernel_size,kernel_size),np.uint8))

        final_mask=mask_from_roi_eroded*mask
        for k in keys:
            curr_map=makevol(map_[k],mask>0)
            map_[k]=curr_map[final_mask>0]

        mask=final_mask        
    



    maskROI=maskROI[mask>0]
    print(maskROI.shape)

    if wT1_threshold is not None:
        print("Filtering High wT1 values")
        wT1 = map_["wT1"]
        for k in keys:
            map_[k]=map_[k][wT1 < wT1_threshold]
        
        maskROI=maskROI[wT1 < wT1_threshold]
        

    #Removing ROIs with less than min_ROI_count values
    freq = collections.Counter(maskROI)
    maskROI = np.array([ele if freq[ele] > min_ROI_count else 0 for ele in maskROI])




    results=pd.DataFrame()
    for i, k in enumerate(sorted(keys)):
        obs=map_[k]


        if adj_wT1 and k == "wT1":
            print("Filtering wT1 map for high FF values")
            ff = map_["ff"]
            obs = obs[ff < fat_threshold]
            maskROI_current = maskROI[ff < fat_threshold]
        else:
            maskROI_current=maskROI
        
        

        df_obs = pd.DataFrame(columns=["Data", "Groups"],
                              data=np.stack([obs.flatten(), maskROI_current.flatten()], axis=-1))

        obs = np.array(df_obs.groupby("Groups").mean())[1:]
        print(obs)

        


            # print(list(pred_std.reshape(1,-1)))
        results[k+" Mean"]=obs.flatten()

        if return_std:
            obs_std = np.array(df_obs.groupby("Groups").std())[1:]
 
            results[k+" Std"] = obs_std.flatten()

    return results

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def voronoi_volumes_freud(points,box_size=2*np.pi):

    box = freud.box.Box.cube(box_size + 0.00001)

    voro = freud.locality.Voronoi()
    voro.compute(system=(box, points))

    volumes = np.array(voro.volumes)
    #volumes /= volumes.sum()

    return volumes

def voronoi_volumes(points,min_x=None,min_y=None,max_x=None,max_y=None,min_z=None,max_z=None,eps=0.1):
    dim =points.shape[1]

    vor = Voronoi(points)

    if dim==3:
        regions, vertices = voronoi_finite_polygons_2d(vor)

    else:
        regions, vertices = voronoi_finite_polygons_3d(vor)

    if min_x is None:
        min_x = vor.min_bound[0] - eps

    if max_x is None:
        max_x = vor.max_bound[0] + eps


    if min_y is None:
        min_y = vor.min_bound[1] - eps

    if max_y is None:
        max_y = vor.max_bound[1] + eps


    if dim==3:
        if min_z is None:
            min_z = vor.min_bound[2] - eps

        if max_z is None:
            max_z = vor.max_bound[2] + eps


        mins = np.tile((min_x, min_y,min_z), (vertices.shape[0], 1))
        maxs = np.tile((max_x, max_y,min_z), (vertices.shape[0], 1))

    else:
        mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
        maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))

    bounded_vertices = np.max((vertices, mins), axis=0)
    bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

    vol = np.zeros(len(regions))
    for i, indices in enumerate(regions):
        if -1 in indices:  # some regions can be opened - SHOULD NOT HAPPEN WITH NEW IMPLEMENTATION
            # indices.remove(-1)
            print("Warning : Had to set a voronoi volume to 0 for {}".format(np.round(v.points[i], 2)))
            vol[i] = 0
        else:
            vol[i] = ConvexHull(bounded_vertices[indices]).volume
        # except:
        #     print("Warning : Had to set a voronoi volume to 0 for {}".format(np.round(v.points[i],2)))
        #     vol[i]=0

    return vol, vor


def normalize_image_series(images_series):
    shapes=list(images_series.shape)
    last_dimensions_collapse =reduce(lambda x, y: x*y, shapes[1:])
    normalization = np.reshape(np.sum(np.abs(np.reshape(images_series,(-1,last_dimensions_collapse)) ** 2), axis=-1) ** 0.5, (-1,)+tuple(np.ones(len(shapes[1:])).astype(int)))
    images_series /= normalization
    return images_series


def transform_py_map(res,mask):
    map_py = res.copy()
    map_py.pop("info")
    map_py.pop("mask")
    keys = list(map_py.keys()).copy()

    for k in keys:
        map_py[str.split(k, "map")[0]] = map_py.pop(k)[mask > 0]
    map_py["attB1"] = map_py.pop("b1")
    map_py["fT1"] = map_py.pop("ft1")
    map_py["wT1"] = map_py.pop("wt1")
    return map_py

def build_mask_from_volume(volumes,threshold_factor=0.05,iterations=3):
    mask = False
    unique = np.histogram(np.abs(volumes), 100)[1]
    mask = mask | (np.abs(volumes) > unique[int(len(unique) * threshold_factor)])
    mask = ndimage.binary_closing(mask, iterations=iterations)
    return mask*1


def build_mask_single_image(kdata,trajectory,size,useGPU=False,eps=1e-6,threshold_factor=1/7):
    mask = False

    npoint=trajectory.paramDict["npoint"]
    traj = trajectory.get_traj_for_reconstruction()

    # kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)

    density = np.abs(np.linspace(-1, 1, npoint))
    kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]
    # kdata = (normalize_image_series(np.array(kdata)))
    kdata_all = np.concatenate(kdata)
    traj_all = np.concatenate(list(traj))
    if traj_all.shape[-1]==2: # For slices

        if not(useGPU):
            volume_rebuilt = finufft.nufft2d1(traj_all[:,0], traj_all[:,1], kdata_all, size)
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)


            c_retrieved = kdata_all
            kx = traj_all[:, 0]
            ky = traj_all[:, 1]

            # Cast to desired datatype.
            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            c_retrieved = c_retrieved.astype(complex_dtype)

            kx_gpu = to_gpu(kx)
            ky_gpu = to_gpu(ky)
            c_retrieved_gpu = to_gpu(c_retrieved)

            # Allocate memory for the uniform grid on the GPU.

            # Initialize the plan and set the points.
            plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
            plan.set_pts(kx_gpu, ky_gpu)

            # Execute the plan, reading from the strengths array c and storing the
            # result in fk_gpu.
            plan.execute(c_retrieved_gpu, fk_gpu)

            fk = np.squeeze(fk_gpu.get())
            volume_rebuilt = np.array(fk)
            kx_gpu.gpudata.free()
            ky_gpu.gpudata.free()
            fk_gpu.gpudata.free()
            c_retrieved_gpu.gpudata.free()

            plan.__del__()


        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique)*threshold_factor)])
        #mask = ndimage.binary_closing(mask, iterations=10)


    elif traj_all.shape[-1]==3: # For volumes
        if not(useGPU):
            volume_rebuilt = finufft.nufft3d1(traj_all[:, 2],traj_all[:, 0], traj_all[:, 1], kdata_all, size)
        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)

            c_retrieved = kdata_all
            kx = traj_all[:, 0]
            ky = traj_all[:, 1]
            kz = traj_all[:, 2]

            # Cast to desired datatype.
            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            c_retrieved = c_retrieved.astype(complex_dtype)

            # Allocate memory for the uniform grid on the GPU.

            # Initialize the plan and set the points.
            plan = cufinufft(1, (N1, N2, N3), 1, eps=eps, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

            # Execute the plan, reading from the strengths array c and storing the
            # result in fk_gpu.
            plan.execute(to_gpu(c_retrieved), fk_gpu)

            fk = np.squeeze(fk_gpu.get())
            volume_rebuilt = np.array(fk)
            plan.__del__()

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique)*threshold_factor)])
        mask = ndimage.binary_closing(mask, iterations=3)

    return mask

def build_mask_single_image_multichannel(kdata,trajectory,size,density_adj=True,eps=1e-6,b1=None,threshold_factor=None,useGPU=False,normalize_kdata=True,light_memory_usage=False,is_theta_z_adjusted=False,selected_spokes=None,normalize_volumes=True):
    '''

    :param kdata: shape nchannels*ntimesteps*point_per_timestep
    :param trajectory: shape ntimesteps * point_per_timestep * ndim (2 or 3)
    :param size: image size
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
        volume_rebuilt = build_single_image_multichannel(kdata[:,selected_spokes,:,:],trajectory_for_mask,size,density_adj,eps,b1,useGPU=useGPU,normalize_kdata=normalize_kdata,light_memory_usage=light_memory_usage,is_theta_z_adjusted=is_theta_z_adjusted,normalize_volumes=normalize_volumes)
    else:
        volume_rebuilt = build_single_image_multichannel(kdata, trajectory_for_mask, size,
                                                         density_adj, eps, b1, useGPU=useGPU,
                                                         normalize_kdata=normalize_kdata,
                                                         light_memory_usage=light_memory_usage,
                                                         is_theta_z_adjusted=is_theta_z_adjusted,normalize_volumes=normalize_volumes)

    traj = trajectory.get_traj_for_reconstruction(timesteps=1)


    if traj.shape[-1]==2: # For slices

        if threshold_factor is None:
            threshold_factor = 1/7

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique) *threshold_factor)])
        #mask = ndimage.binary_closing(mask, iterations=3)


    elif traj.shape[-1]==3: # For volumes

        if threshold_factor is None:
            threshold_factor = 1/20

        unique = np.histogram(np.abs(volume_rebuilt), 100)[1]
        mask = mask | (np.abs(volume_rebuilt) > unique[int(len(unique) *threshold_factor)])
        mask = ndimage.binary_closing(mask, iterations=3)

    return mask


def build_single_image_multichannel(kdata,trajectory,size,density_adj=True,eps=1e-6,b1=None,useGPU=False,normalize_kdata=False,light_memory_usage=False,is_theta_z_adjusted=False,normalize_volumes=False):
    '''

    :param kdata: shape nchannels*ntimesteps*point_per_timestep
    :param trajectory: shape ntimesteps * point_per_timestep * ndim (2 or 3)
    :param size: image size
    :param density_adj:
    :param eps:
    :param b1: coil sensitivity map
    :return: mask of size size
    '''
    volume_rebuilt=simulate_radial_undersampled_images_multi(kdata,trajectory,size,density_adj,eps,is_theta_z_adjusted,b1,1,useGPU,None,normalize_kdata,light_memory_usage,True)[0]
    return volume_rebuilt

def generate_kdata(volumes,trajectory,useGPU=False,eps=1e-6,ntimesteps=None):
    if ntimesteps is None:
        traj=trajectory.get_traj()
    else:
        traj = trajectory.get_traj_for_reconstruction(ntimesteps)

    if volumes.dtype=="complex64":
        traj=traj.astype("float32")

    ntimesteps=volumes.shape[0]
    if traj.shape[-1]==2:# For slices
        if not(useGPU):
            kdata = [
                    finufft.nufft2d2(t[:,0], t[:,1], p)
                    for t, p in zip(traj, volumes)
                ]
        else:
            # Allocate memory for the nonuniform coefficients on the GPU.
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            N1,N2 = volumes.shape[1],volumes.shape[2]
            M = traj.shape[1]
            c_gpu = GPUArray((1, M), dtype=complex_dtype)
            # Initialize the plan and set the points.
            kdata=[]
            for i in list(range(volumes.shape[0])):
                fk = volumes[i, :, :]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]

                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                fk = fk.astype(complex_dtype)

                plan = cufinufft(2, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))
                plan.execute(c_gpu, to_gpu(fk))
                c = np.squeeze(c_gpu.get())
                kdata.append(c)
                plan.__del__()

    elif traj.shape[-1]==3:# For volumes
        if not (useGPU):

            kdata = [
                finufft.nufft3d2(t[:, 2],t[:, 0], t[:, 1], p)
                for t, p in zip(traj, volumes)
            ]

            # traj=traj.reshape(-1,3)
            # kdata = finufft.nufft3d2(traj[:, 2],traj[:, 0], traj[:, 1], volumes)
            # kdata=kdata.reshape(ntimesteps,-1)
        else:
            # Allocate memory for the nonuniform coefficients on the GPU.
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            N1, N2,N3 = volumes.shape[1], volumes.shape[2],volumes.shape[3]
            M = traj.shape[1]
            c_gpu = GPUArray((M), dtype=complex_dtype)
            # Initialize the plan and set the points.
            kdata = []
            for i in list(range(volumes.shape[0])):
                fk = volumes[i, :, :]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]
                kz = traj[i, :, 2]

                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                fk = fk.astype(complex_dtype)

                plan = cufinufft(2, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))
                plan.execute(c_gpu, to_gpu(fk))
                c = np.squeeze(c_gpu.get())
                kdata.append(c)
                plan.__del__()
    return kdata


def generate_kdata_new(volumes,trajectory,useGPU=False,eps=1e-6,ntimesteps=None,retained_timesteps=None):
    if ntimesteps is None:
        traj=trajectory.get_traj()
    else:
        traj = trajectory.get_traj_for_reconstruction(ntimesteps)

    if retained_timesteps is not None:
        traj=traj[retained_timesteps]

    ntimesteps=volumes.shape[0]
    if not(len(traj)==ntimesteps):
        raise ValueError("Mismatch between image series size and retained timesteps")
    if traj.shape[-1]==2:# For slices
        if not(useGPU):
            kdata = [
                    finufft.nufft2d2(t[:,0], t[:,1], p)
                    for t, p in zip(traj, volumes)
                ]
        else:
            # Allocate memory for the nonuniform coefficients on the GPU.
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            N1,N2 = volumes.shape[1],volumes.shape[2]
            M = traj.shape[1]
            c_gpu = GPUArray((1, M), dtype=complex_dtype)
            # Initialize the plan and set the points.
            kdata=[]
            for i in list(range(volumes.shape[0])):
                fk = volumes[i, :, :]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]

                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                fk = fk.astype(complex_dtype)

                plan = cufinufft(2, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))
                plan.execute(c_gpu, to_gpu(fk))
                c = np.squeeze(c_gpu.get())
                kdata.append(c)
                plan.__del__()

    elif traj.shape[-1]==3:# For volumes
        if not (useGPU):

            kdata = [
                finufft.nufft3d2(t[:, 2],t[:, 0], t[:, 1], p)
                for t, p in zip(traj, volumes)
            ]

            # traj=traj.reshape(-1,3)
            # kdata = finufft.nufft3d2(traj[:, 2],traj[:, 0], traj[:, 1], volumes)
            # kdata=kdata.reshape(ntimesteps,-1)
        else:
            # Allocate memory for the nonuniform coefficients on the GPU.
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            N1, N2,N3 = volumes.shape[1], volumes.shape[2],volumes.shape[3]
            M = traj.shape[1]
            c_gpu = GPUArray((M), dtype=complex_dtype)
            # Initialize the plan and set the points.
            kdata = []
            for i in list(range(volumes.shape[0])):
                fk = volumes[i, :, :]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]
                kz = traj[i, :, 2]

                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                fk = fk.astype(complex_dtype)

                plan = cufinufft(2, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))
                plan.execute(c_gpu, to_gpu(fk))
                c = np.squeeze(c_gpu.get())
                kdata.append(c)
                plan.__del__()
    return kdata

def generate_kdata_singular(volumes, trajectory, useGPU=False, eps=1e-6):
    """
    generate L0 singular kdata for all the spokes in the trajectory
    """

    traj = trajectory.get_traj()
    ntimesteps = volumes.shape[0]
    if traj.shape[-1] == 2:  # For slices
        raise ValueError("kdata singular not implemented in 2D")

    elif traj.shape[-1] == 3:  # For volumes
        if not (useGPU):
            traj=traj.reshape(-1,3)
            print('vol',volumes.shape)
            print('traj',traj.shape)
            kdata = finufft.nufft3d2(traj[:, 2],traj[:, 0], traj[:, 1], volumes.astype(np.complex128))


        else:
            raise ValueError("kdata singular not implemented for GPU")
            # # Allocate memory for the nonuniform coefficients on the GPU.
            # dtype = np.float32  # Datatype (real)
            # complex_dtype = np.complex64
            # N1, N2, N3 = volumes.shape[1], volumes.shape[2], volumes.shape[3]
            # M = traj.shape[1]
            # c_gpu = GPUArray((M), dtype=complex_dtype)
            # # Initialize the plan and set the points.
            # kdata = []
            # for i in list(range(volumes.shape[0])):
            #     fk = volumes[i, :, :]
            #     kx = traj[i, :, 0]
            #     ky = traj[i, :, 1]
            #     kz = traj[i, :, 2]
            #
            #     kx = kx.astype(dtype)
            #     ky = ky.astype(dtype)
            #     kz = kz.astype(dtype)
            #     fk = fk.astype(complex_dtype)
            #
            #     plan = cufinufft(2, (N1, N2, N3), 1, eps=eps, dtype=dtype)
            #     plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
            #     plan.execute(c_gpu, to_gpu(fk))
            #     c = np.squeeze(c_gpu.get())
            #     kdata.append(c)
            #     plan.__del__()
    return kdata

def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new

def generate_kdata_singular_multi(volumes,trajectory,b1_all_slices,useGPU=False,eps=1e-6):
    kdata=[]

    for i in tqdm(range(b1_all_slices.shape[1])):
        kdata.append(generate_kdata_singular(volumes*np.expand_dims(b1_all_slices[:,i],axis=0),trajectory, useGPU=useGPU, eps=eps))

    kdata = np.array(kdata)

    return kdata

def generate_kdata_multi(volumes,trajectory,b1_all_slices,useGPU=False,eps=1e-6,ntimesteps=None):
    kdata=[]
    if volumes.dtype=="complex64":
        b1_all_slices=b1_all_slices.astype("complex64")

    for i in tqdm(range(b1_all_slices.shape[0])):
        kdata.append(generate_kdata(volumes*np.expand_dims(b1_all_slices[i],axis=0),trajectory, useGPU=useGPU, eps=eps,ntimesteps=ntimesteps))

    kdata = np.array(kdata)

    return kdata

def generate_kdata_multi_new(volumes,trajectory,b1_all_slices,useGPU=False,eps=1e-6,ntimesteps=None,retained_timesteps=None):
    kdata=[]
    for i in tqdm(range(b1_all_slices.shape[0])):
        kdata.append(generate_kdata_new(volumes*np.expand_dims(b1_all_slices[i],axis=0),trajectory, useGPU=useGPU, eps=eps,ntimesteps=ntimesteps,retained_timesteps=retained_timesteps))

    kdata = np.array(kdata)

    return kdata

def buildROImask(map,max_clusters=5):
    # ROI using KNN clustering
    if "wT1" not in map:
        raise ValueError("wT1 should be in the param Map to build the ROI")

    #print(map["wT1"].shape)
    orig_data = map["wT1"].reshape(-1, 1)
    data = orig_data + np.random.normal(size=orig_data.shape,scale=0.1)
    model = KMeans(n_clusters=np.minimum(len(np.unique(orig_data)),max_clusters))
    model.fit(data)
    groups = model.labels_ + 1
    #print(groups.shape)
    return groups

def buildROImask_unique(map,key="wT1"):
    # ROI using regions with same value of key
    if key not in map:
        raise ValueError("{} should be in the param Map to build the ROI".format(key))

    unique_wT1 = np.unique(map[key])
    maskROI = np.zeros(map[key].shape)
    for i, value in enumerate(unique_wT1):
        maskROI[map[key] == value] = i + 1

    return maskROI

def simulate_radial_undersampled_images(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6,is_theta_z_adjusted=False,ntimesteps=175):
#Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)
    traj=trajectory.get_traj_for_reconstruction(ntimesteps)
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes = trajectory.paramDict["total_nspokes"]
    nspoke = int(nb_allspokes / ntimesteps)

    if not(is_theta_z_adjusted):
        dtheta = np.pi / nspoke
        dz = 1/trajectory.paramDict["nb_rep"]

    else:
        dtheta=1
        dz = 1/(2*np.pi)


    if not(len(kdata)==len(traj)):
        kdata=np.array(kdata).reshape(len(traj),-1)

    if not(kdata[0].shape[0]==traj[0].shape[0]):
        raise ValueError("Incompatible Kdata and Trajectory shapes")

    kdata = [k / (2*npoint)*dz * dtheta for k in kdata]



    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj[0].shape[-1] == 2:  # 2D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:,0], t[:,1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)
            images_GPU = []
            for i in list(range(len(kdata))):

                c_retrieved = kdata[i]
                kx = traj[i][:, 0]
                ky = traj[i][:, 1]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.


                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt=np.array(images_GPU)
    elif traj[0].shape[-1] == 3:  # 3D
        if not(useGPU):
            #images_series_rebuilt = [
            #    finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
            #    for t, s in zip(traj, kdata)
            #]

            images_series_rebuilt=[]
            for t,s in tqdm(zip(traj,kdata)):
                s=s.astype(np.complex64)
                t=t.astype(np.float32)
                images_series_rebuilt.append(finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size))

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            images_GPU=[]
            for i in list(range(len(kdata))):
                fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                c_retrieved = kdata[i]
                kx = traj[i][ :, 0]
                ky = traj[i][ :, 1]
                kz = traj[i][ :, 2]


                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.
                c_retrieved_gpu = to_gpu(c_retrieved)

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(c_retrieved_gpu, fk_gpu)

                fk = np.squeeze(fk_gpu.get())

                fk_gpu.gpudata.free()
                c_retrieved_gpu.gpudata.free()

                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)


def simulate_radial_undersampled_images(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6,is_theta_z_adjusted=False,ntimesteps=175,nthreads=1,fftw=0):
#Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)
    traj=trajectory.get_traj_for_reconstruction(ntimesteps)
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes = trajectory.paramDict["total_nspokes"]
    nspoke = int(nb_allspokes / ntimesteps)

    if not(is_theta_z_adjusted):
        dtheta = np.pi / nspoke
        dz = 1/trajectory.paramDict["nb_rep"]

    else:
        dtheta=1
        dz = 1/(2*np.pi)


    if not(len(kdata)==len(traj)):
        kdata=np.array(kdata).reshape(len(traj),-1)

    if not(kdata[0].shape[0]==traj[0].shape[0]):
        raise ValueError("Incompatible Kdata and Trajectory shapes")

    kdata = [k / (2*npoint)*dz * dtheta for k in kdata]



    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj[0].shape[-1] == 2:  # 2D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:,0], t[:,1], s, size,nthreads=nthreads,fftw=fftw)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)
            images_GPU = []
            for i in list(range(len(kdata))):

                c_retrieved = kdata[i]
                kx = traj[i][:, 0]
                ky = traj[i][:, 1]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.


                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt=np.array(images_GPU)
    elif traj[0].shape[-1] == 3:  # 3D
        if not(useGPU):
            #images_series_rebuilt = [
            #    finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
            #    for t, s in zip(traj, kdata)
            #]

            images_series_rebuilt=[]
            for t,s in tqdm(zip(traj,kdata)):
                s=s.astype(np.complex64)
                t=t.astype(np.float32)
                images_series_rebuilt.append(finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size))

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            images_GPU=[]
            for i in list(range(len(kdata))):
                fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                c_retrieved = kdata[i]
                kx = traj[i][ :, 0]
                ky = traj[i][ :, 1]
                kz = traj[i][ :, 2]


                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.
                c_retrieved_gpu = to_gpu(c_retrieved)

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(c_retrieved_gpu, fk_gpu)

                fk = np.squeeze(fk_gpu.get())

                fk_gpu.gpudata.free()
                c_retrieved_gpu.gpudata.free()

                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)

def simulate_radial_undersampled_images_density_optim(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6,is_theta_z_adjusted=False,ntimesteps=175):
#Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)
    traj=trajectory.get_traj_for_reconstruction()
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes=trajectory.paramDict["total_nspokes"]
    nspoke=int(nb_allspokes/ntimesteps)

    if not(is_theta_z_adjusted):
        dtheta = np.pi / nspoke
        dz = 1/trajectory.paramDict["nb_rep"]

    else:
        dtheta=np.pi
        dz = 1


    if not(len(kdata)==len(traj)):
        kdata=np.array(kdata).reshape(len(traj),-1)

    if not(kdata[0].shape[0]==traj[0].shape[0]):
        raise ValueError("Incompatible Kdata and Trajectory shapes")

    kdata = [k / (npoint)*dz * dtheta for k in kdata]

    if density_adj:
        traj_left=np.moveaxis(traj,1,-1)
        traj_right=np.expand_dims(traj_left,axis=-1)
        traj_right=np.moveaxis(traj_right,2,-1)
        traj_left = np.expand_dims(traj_left,axis=-1)
        for ts in tqdm(range(traj.shape[0])):
            density=traj_left[ts]-traj_right[ts]
            density=np.sinc(density)**2
            density=np.prod(density,axis=0)
            density = 1/np.sum(density,axis=1)
            kdata[ts] = kdata[ts]*density

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj[0].shape[-1] == 2:  # 2D
        if not(useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:,0], t[:,1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)
            images_GPU = []
            for i in list(range(len(kdata))):

                c_retrieved = kdata[i]
                kx = traj[i][:, 0]
                ky = traj[i][:, 1]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.


                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt=np.array(images_GPU)
    elif traj[0].shape[-1] == 3:  # 3D
        if not(useGPU):
            #images_series_rebuilt = [
            #    finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
            #    for t, s in zip(traj, kdata)
            #]

            images_series_rebuilt=[]
            for t,s in tqdm(zip(traj,kdata)):
                images_series_rebuilt.append(finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size))

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            images_GPU=[]
            for i in list(range(len(kdata))):
                fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                c_retrieved = kdata[i]
                kx = traj[i][ :, 0]
                ky = traj[i][ :, 1]
                kz = traj[i][ :, 2]


                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                kz = kz.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.
                c_retrieved_gpu = to_gpu(c_retrieved)

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(c_retrieved_gpu, fk_gpu)

                fk = np.squeeze(fk_gpu.get())

                fk_gpu.gpudata.free()
                c_retrieved_gpu.gpudata.free()

                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    #images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)


def convolution_kernel_radial_single_channel(traj,dk,npoint,size,density_adj=False):
    dtheta = 1
    dz = 1 / (2 * np.pi)


    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        #density=np.expand_dims(axis=0)

        dk =(np.reshape(dk, (-1, npoint)) * density).flatten()


    dk *= dz * dtheta / (2*npoint)

    if dk.dtype == "complex64":
        traj=traj.astype("float32")
        print(traj.dtype)

    fk = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], dk, size)
    return fk





def simulate_radial_undersampled_images_multi(kdata, trajectory, size, density_adj=True, eps=1e-6,
                                              is_theta_z_adjusted=False, b1=None, ntimesteps=175, useGPU=False,
                                              memmap_file=None, normalize_kdata=False, light_memory_usage=False,
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

    elif density_adj=="Voronoi":
        print("Calculating Voronoi Density Adj")
        density=[]
        for i in tqdm(range(len(traj))):
            curr_dens=voronoi_volumes_freud(traj[i])
            curr_dens_shape=curr_dens.shape
            curr_dens=curr_dens.reshape(-1,npoint)
            curr_dens[:,0]=curr_dens[:,1]
            curr_dens[:, npoint-1] = curr_dens[:, npoint-2]
            curr_dens=curr_dens.reshape(curr_dens_shape)
            curr_dens /= curr_dens.sum()
            density.append(curr_dens)

        # density = [
        #     voronoi_volumes_freud(traj[i]) for i in
        #     tqdm(range(len(traj)))]
        for j in tqdm(range(nb_channels)):
            kdata[j] = [k * density[i] for i, k in enumerate(kdata[j])]

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
            fk = finufft.nufft2d1(t[:, 0], t[:, 1], np.squeeze(kdata[:, i, :]), size)

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

                    fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[:, i, :], size)
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
                            fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[j][i], size)
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
                            kdata_current = kdata[j][i]
                            #print(t_current.shape)
                            #print(kdata_current.shape)
                            fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata_current, size)
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


def simulate_radial_undersampled_images_multi_new(kdata, trajectory, size, density_adj=True, eps=1e-6,
                                            b1=None, ntimesteps=175, useGPU=False, light_memory_usage=False,
                                               weights=None,
                                              retained_timesteps=None,ntimesteps_final=None):
    # Deals with single channel data / howver kdata can be a list of arrays (meaning each timestep does not need to have the same number of spokes/partitions)

    # if light_memory_usage and not(useGPU):
    #    print("Warning : light memory usage is not used without GPU")
    traj = trajectory.get_traj_for_reconstruction(ntimesteps)

    nb_channels = len(kdata)

    if not (len(kdata[0]) == len(traj)):
        kdata = kdata.reshape(nb_channels, len(traj), -1)

    if retained_timesteps is not None:
        traj = traj[retained_timesteps]
        kdata = kdata[:, retained_timesteps]
        ntimesteps=len(retained_timesteps)

    # print(traj[0].shape)
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes = trajectory.paramDict["total_nspokes"]
    nb_rep=trajectory.paramDict["nb_rep"]

    # print(kdata.shape)
    nspoke = int(nb_allspokes / ntimesteps)

    num_samples = traj.shape[1]


    if type(density_adj) is bool:
        if density_adj:
            density_adj = "Radial"

    if density_adj == "Radial":
        density = np.abs(np.linspace(-1, 1, npoint))
        #density=np.expand_dims(axis=0)
        for j in tqdm(range(nb_channels)):
            kdata[j] = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata[j]]
        kdata=np.array(kdata)

    if weights is not None:
        kdata=kdata.reshape(nb_channels,ntimesteps,nspoke,nb_rep,-1)
        weights=np.expand_dims(weights,axis=(0,-1))
        kdata*=weights
        kdata=kdata.reshape(nb_channels,ntimesteps,-1)




    if kdata[0][0].dtype == "complex64":
        try:
            traj = traj.astype("float32")
        except:
            for i in range(traj.shape[0]):
                traj[i] = traj[i].astype("float32")
        print(traj[0].dtype)


    for i in tqdm(range(nb_channels)):
        kdata[i] /= num_samples

    # kdata = (normalize_image_series(np.array(kdata)))

    if ntimesteps_final is not None:
        ntimesteps=ntimesteps_final
        traj=traj.reshape(ntimesteps_final,-1,3)
        kdata=kdata.reshape(nb_channels,ntimesteps_final,-1)

    output_shape = (ntimesteps,) + size

    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    print("Performing NUFFT")
    if traj[0].shape[-1] == 2:  # 2D

        for i, t in tqdm(enumerate(traj)):
            fk = finufft.nufft2d1(t[:, 0], t[:, 1], np.squeeze(kdata[:, i, :]), size)

            # images_series_rebuilt = np.moveaxis(images_series_rebuilt, 0, 1)
            if b1 is None:
                print(fk.shape)
                if fk.ndim > 2:
                    images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                else:
                    print('Taking abs of image')
                    images_series_rebuilt[i] = np.abs(fk)
            else:
                images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

    elif traj[0].shape[-1] == 3:  # 3D
        if not (useGPU):

            for i, t in tqdm(enumerate(traj)):
                if not (light_memory_usage):

                    fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[:, i, :], size)
                    if b1 is None:
                        images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                    else:
                        images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=0)

                else:

                    for j in tqdm(range(nb_channels)):

                        fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kdata[j,i],
                                                  size)
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
                        c_retrieved = kdata[j,i]
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

    return images_series_rebuilt


def simulate_radial_undersampled_singular_images_multi(kdata, trajectory, size, density_adj=True, eps=1e-6,
                                              is_theta_z_adjusted=False, b1=None, useGPU=False,
                                              memmap_file=None, light_memory_usage=False,
                                              normalize_volumes=True):
    """
    simulate radial undersampled singular images from singular kdata
    Input : nb_channels x L0 x size total_nb_spokes
    Output : L0 x image_size

    """

    L0=kdata.shape[1]
    traj = trajectory.get_traj_for_reconstruction(1)
    #print(traj[0].shape)
    npoint = trajectory.paramDict["npoint"]
    nb_allspokes = trajectory.paramDict["total_nspokes"]
    nb_channels = len(kdata)
    #print(kdata.shape)






    if type(density_adj) is bool:
        if density_adj:
            density_adj="Radial"

    if density_adj=="Radial":
        kdata=kdata.reshape(nb_channels,L0,-1,npoint)
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata.ndim - 2)))

        #density=np.expand_dims(axis=0)
        for j in tqdm(range(nb_channels)):
            kdata[j]*=density


    if kdata[0][0].dtype == "complex64":
        try:
            traj=traj.astype("float32")
        except:
            for i in range(traj.shape[0]):
                traj[i] = traj[i].astype("float32")
        print(traj[0].dtype)


    #for i in tqdm(range(nb_channels)):
    #    kdata[i] *= dz * dtheta / (2*npoint)

    # kdata = (normalize_image_series(np.array(kdata)))

    output_shape = (L0,) + size

    traj=traj.reshape(-1,3)
    kdata=kdata.reshape(nb_channels,L0,-1)

    num_k_samples = traj.shape[0]

    kdata /= num_k_samples

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

        raise ValueError("simulate singular images not implemented for 2D")

    elif traj[0].shape[-1] == 3:  # 3D
        if not (useGPU):

            if not (light_memory_usage):
                fk = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], kdata, size)
                if b1 is None:
                    images_series_rebuilt[i] = np.sqrt(np.sum(np.abs(fk) ** 2, axis=0))
                else:
                    images_series_rebuilt[i] = np.sum(b1.conj() * fk, axis=1)

            else:

                for j in tqdm(range(nb_channels)):

                    kdata_current=kdata[j]
                    fk = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], kdata_current, size)
                    if b1 is None:
                        images_series_rebuilt += np.abs(fk) ** 2
                    else:
                        images_series_rebuilt += np.expand_dims(b1[:,j].conj(),axis=0) * fk

                if b1 is None:
                    images_series_rebuilt = np.sqrt(images_series_rebuilt[i])
        else:
            raise ValueError("simulate singular images not implemented for GPU")

        del kdata
        gc.collect()

        if flushed:
            images_series_rebuilt.flush()
            del images_series_rebuilt
            images_series_rebuilt = np.memmap(memmap_file, dtype="complex64", mode="r", shape=output_shape)




    # images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return images_series_rebuilt

def undersampling_operator_singular(volumes,trajectory,b1_all_slices,density_adj=True,weights=None):
    """
    returns A.H @ W @ A @ volumes where A=F Fourier + sampling operator and W correspond to radial density adjustment
    """

    L0=volumes.shape[0]
    size=volumes.shape[1:]
    nb_channels=b1_all_slices.shape[0]

    nb_allspokes = trajectory.paramDict["total_nspokes"]
    traj = trajectory.get_traj()
    traj = traj.reshape(-1, 3)
    npoint = trajectory.paramDict["npoint"]

    num_k_samples = traj.shape[0]
    num_k_samples=1
    output_shape = (L0,) + size
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    for k in tqdm(range(nb_channels)):

        curr_volumes = volumes * np.expand_dims(b1_all_slices[:,k], axis=0)
        print(curr_volumes.shape)
        curr_kdata = finufft.nufft3d2(traj[:, 2], traj[:, 0], traj[:, 1], curr_volumes)
        print(curr_kdata.shape)
        if density_adj:
            curr_kdata=curr_kdata.reshape(L0,-1,npoint)
            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
            curr_kdata*=density

        if weights is not None:
            weights = weights.reshape(1, -1, 1)
            curr_kdata *= weights


        curr_kdata = curr_kdata.reshape(L0, -1)
        #np.save("kdata_test.npy",curr_kdata)

        fk = finufft.nufft3d1(traj[:, 2], traj[:, 0], traj[:, 1], curr_kdata.squeeze(), size)
        print(fk.shape)
        images_series_rebuilt += b1_all_slices[:,k].conj()* fk

    images_series_rebuilt /= num_k_samples

    return images_series_rebuilt


def undersampling_operator_singular_new(volumes,trajectory,b1_all_slices=None,ntimesteps=175,density_adj=True,weights=None,retained_timesteps=None):
    """
    returns A.H @ W @ A @ volumes where A=F Fourier + sampling operator and W correspond to radial density adjustment
    """

    L0=volumes.shape[0]
    size=volumes.shape[1:]

    print("LO {}".format(L0))
    print("Image size {}".format(size))

    if b1_all_slices is None:
        b1_all_slices=np.ones((1,)+size,dtype="complex64")

    nb_channels=b1_all_slices.shape[0]

    nb_slices=size[0]

    print("Nb channels {}".format(nb_channels))

    #nb_allspokes = trajectory.paramDict["total_nspokes"]

    if not(type(trajectory)==np.ndarray):
        traj = trajectory.get_traj_for_reconstruction(1)
        if retained_timesteps is not None:
            traj=traj[retained_timesteps]
        if not((type(weights)==int)or(weights is None)):
            weights=weights.flatten()
        traj = traj.reshape(-1, 2).astype("float32")
        npoint = trajectory.paramDict["npoint"]
    
    else:#testing fully sampled cartesian trajectory 
        traj=trajectory
        weights=None
        density_adj=False
        traj=traj.reshape(-1,2)

    #print(traj.shape)
    #print(weights.shape)

    num_k_samples = traj.shape[0]

    output_shape = (L0,) + size
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)


    if (weights is not None) and not(type(weights)==int):
        weights = np.expand_dims(weights, axis=(0, -1))
        print(weights.shape)
        

    

    for k in tqdm(range(nb_channels)):
        #print("Ch {}".format(k))
        #print(volumes.shape)
        #print(b1_all_slices[k].shape)
        curr_volumes = volumes * np.expand_dims(b1_all_slices[:,k], axis=0)
        #print(curr_volumes.shape)
        curr_kdata_slice=np.fft.fftshift(sp.fft.fft(
            np.fft.ifftshift(curr_volumes, axes=1),
            axis=1,workers=24), axes=1).astype("complex64")
        #curr_kdata=np.zeros((L0,nb_slices,traj.shape[0]), dtype="complex64")
        print(curr_kdata_slice.shape)
        curr_kdata = finufft.nufft2d2(traj[:, 0],traj[:, 1],curr_kdata_slice.reshape((L0*nb_slices,)+size[1:])).reshape(L0,nb_slices,-1)
        #print(curr_kdata.shape)
        if density_adj:
            curr_kdata=curr_kdata.reshape(L0,-1,npoint)
            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
            curr_kdata*=density

        #print(weights.shape)
        if weights is not None:
            curr_kdata = curr_kdata.reshape((L0,-1,npoint))
            curr_kdata *= weights

        curr_kdata = curr_kdata.reshape(L0, nb_slices,traj.shape[0])
        curr_kdata = np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(curr_kdata,axes=1),axis=1,workers=24),axes=1).astype(np.complex64)

        images_series_rebuilt+=np.expand_dims(b1_all_slices[:,k].conj(), axis=0)* (finufft.nufft2d1(traj[:, 0],traj[:, 1],curr_kdata.reshape(L0*nb_slices,-1),size[1:])).reshape((L0,)+size)


    images_series_rebuilt /= num_k_samples
    return images_series_rebuilt


def undersampling_operator(volumes,trajectory,b1_all_slices,density_adj=True,light_memory_usage=False):
    """
    returns A.H @ W @ A @ volumes where A=F Fourier + sampling operator and W correspond to radial density adjustment
    """

    ntimesteps=volumes.shape[0]
    size=volumes.shape[1:]
    nb_channels=b1_all_slices.shape[0]

    traj = trajectory.get_traj()
    if len(size)==3:
        traj = traj.reshape(ntimesteps,-1, 3)
    else:
        traj = traj.reshape(ntimesteps, -1, 2)
    npoint = trajectory.paramDict["npoint"]

    if volumes.dtype == "complex64":
        traj=traj.astype("float32")
        b1_all_slices=b1_all_slices.astype("complex64")

    num_k_samples = traj.shape[1]

    output_shape = (ntimesteps,) + size
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    for k in tqdm(range(nb_channels)):
        if not(light_memory_usage):
            curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
            #print(curr_volumes.shape)
            if len(size)==3:
                curr_kdata = [finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], v) for (t,v) in zip (traj,curr_volumes)]
            else:
                curr_kdata = [finufft.nufft2d2( t[:, 0], t[:, 1], v) for (t, v) in zip(traj, curr_volumes)]
            del curr_volumes
            curr_kdata=np.array(curr_kdata)
            #print(curr_kdata.shape)
            if density_adj:
                curr_kdata=curr_kdata.reshape(ntimesteps,-1,npoint)
                density = np.abs(np.linspace(-1, 1, npoint))
                density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
                curr_kdata*=density


            curr_kdata = curr_kdata.reshape(ntimesteps, -1)

            if len(size) == 3:
                fk = [finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kd, size) for (t,kd) in zip(traj,curr_kdata)]
            else:
                fk = [finufft.nufft2d1(t[:, 0], t[:, 1], kd, size) for (t, kd) in zip(traj, curr_kdata)]
            del curr_kdata
            fk=np.array(fk)
            #print(fk.shape)
            images_series_rebuilt += b1_all_slices[k].conj()* fk

        else:
            for ts in tqdm(range(ntimesteps)):
                curr_volumes = volumes[ts] * b1_all_slices[k]
                #print(curr_volumes.shape)
                if len(size) == 3:
                    t=traj[ts]
                    curr_kdata = finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], curr_volumes)
                else:
                    t = traj[ts]
                    curr_kdata = finufft.nufft2d2(t[:, 0], t[:, 1], curr_volumes)
                del curr_volumes
                #print(curr_kdata.shape)
                if density_adj:
                    curr_kdata = curr_kdata.reshape(-1, npoint)
                    density = np.abs(np.linspace(-1, 1, npoint))
                    density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
                    curr_kdata *= density

                curr_kdata = curr_kdata.flatten()

                if len(size) == 3:
                    t=traj[ts]
                    fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], curr_kdata, size)
                else:
                    t = traj[ts]
                    fk = finufft.nufft2d1(t[:, 0], t[:, 1], curr_kdata, size)
                del curr_kdata

                #print(fk.shape)
                images_series_rebuilt[ts] += b1_all_slices[k].conj() * fk


    images_series_rebuilt /= num_k_samples
    return images_series_rebuilt


# def undersampling_operator_new(volumes,trajectory,b1_all_slices,density_adj=True,weights=None,retained_timesteps=None,ntimesteps=None):
#     """
#     returns A.H @ W @ A @ volumes where A=F Fourier + sampling operator and W correspond to radial density adjustment
#     """
#     if ntimesteps is None:
#         ntimesteps=volumes.shape[0]
#     size=volumes.shape[1:]
#     nb_channels=b1_all_slices.shape[0]
#
#     traj = trajectory.get_traj()
#     traj = traj.reshape(ntimesteps,-1, 3)
#     if retained_timesteps is not None:
#         traj=traj[retained_timesteps]
#         ntimesteps=len(retained_timesteps)
#
#     npoint = trajectory.paramDict["npoint"]
#
#     num_k_samples = traj.shape[1]
#
#     output_shape = (ntimesteps,) + size
#     images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)
#
#     for k in tqdm(range(nb_channels)):
#
#         curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
#         print(curr_volumes.shape)
#         if len(size)==3:
#             curr_kdata = [finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], v) for (t,v) in zip (traj,curr_volumes)]
#         else:
#             curr_kdata = [finufft.nufft2d2( t[:, 0], t[:, 1], v) for (t, v) in zip(traj, curr_volumes)]
#         curr_kdata=np.array(curr_kdata)
#         print(curr_kdata.shape)
#         if density_adj:
#             curr_kdata=curr_kdata.reshape(ntimesteps,-1,npoint)
#             density = np.abs(np.linspace(-1, 1, npoint))
#             density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
#             curr_kdata*=density
#         curr_kdata = curr_kdata.reshape(ntimesteps, -1)
#         if weights is not None:
#             curr_kdata = curr_kdata.reshape(weights.shape +(-1,))
#             weights = np.expand_dims(weights, axis=-1)
#             curr_kdata *= weights
#             curr_kdata = curr_kdata.reshape(ntimesteps, -1)
#
#         if len(size) == 3:
#             fk = [finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kd, size) for (t,kd) in zip(traj,curr_kdata)]
#         else:
#             fk = [finufft.nufft2d1(t[:, 0], t[:, 1], kd, size) for (t, kd) in zip(traj, curr_kdata)]
#         fk=np.array(fk)
#         print(fk.shape)
#         images_series_rebuilt += b1_all_slices[k].conj()* fk
#
#     images_series_rebuilt /= num_k_samples
#     return images_series_rebuilt

def undersampling_operator_new(volumes, trajectory, b1_all_slices, density_adj=True, weights=None,
                               retained_timesteps=None, ntimesteps=None, light_memory_usage=True):
    """
    returns A.H @ W @ A @ volumes where A=F Fourier + sampling operator and W correspond to radial density adjustment
    """
    if ntimesteps is None:
        ntimesteps = volumes.shape[0]
    size = volumes.shape[1:]
    nb_channels = b1_all_slices.shape[0]

    traj = trajectory.get_traj()
    traj = traj.reshape(ntimesteps, -1, 3)
    if retained_timesteps is not None:
        traj = traj[retained_timesteps]
        ntimesteps = len(retained_timesteps)

    npoint = trajectory.paramDict["npoint"]

    if volumes.dtype == "complex64":
        traj = traj.astype("float32")
        b1_all_slices = b1_all_slices.astype("complex64")
    num_k_samples = traj.shape[1]

    output_shape = (ntimesteps,) + size
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    for k in tqdm(range(nb_channels)):
        if not (light_memory_usage):
            curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
            print(curr_volumes.shape)
            if len(size) == 3:
                curr_kdata = [finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], v) for (t, v) in zip(traj, curr_volumes)]
            else:
                curr_kdata = [finufft.nufft2d2(t[:, 0], t[:, 1], v) for (t, v) in zip(traj, curr_volumes)]
            curr_kdata = np.array(curr_kdata)
            print(curr_kdata.shape)
            if density_adj:
                curr_kdata = curr_kdata.reshape(ntimesteps, -1, npoint)
                density = np.abs(np.linspace(-1, 1, npoint))
                density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
                curr_kdata *= density
            curr_kdata = curr_kdata.reshape(ntimesteps, -1)
            if weights is not None:
                curr_kdata = curr_kdata.reshape(weights.shape + (-1,))
                weights = np.expand_dims(weights, axis=-1)
                curr_kdata *= weights
                curr_kdata = curr_kdata.reshape(ntimesteps, -1)

            if len(size) == 3:
                fk = [finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kd, size) for (t, kd) in zip(traj, curr_kdata)]
            else:
                fk = [finufft.nufft2d1(t[:, 0], t[:, 1], kd, size) for (t, kd) in zip(traj, curr_kdata)]
            fk = np.array(fk)
            print(fk.shape)
            images_series_rebuilt += b1_all_slices[k].conj() * fk

        else:
            for ts in tqdm(range(ntimesteps)):
                curr_volumes = volumes[ts] * b1_all_slices[k]
                print(curr_volumes.shape)
                if len(size) == 3:
                    t = traj[ts]
                    curr_kdata = finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], curr_volumes)
                else:
                    t = traj[ts]
                    curr_kdata = finufft.nufft3d2(t[:, 0], t[:, 1], curr_volumes)
                del curr_volumes
                print(curr_kdata.shape)

                if density_adj:
                    curr_kdata = curr_kdata.reshape(-1, npoint)
                    density = np.abs(np.linspace(-1, 1, npoint))
                    density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
                    curr_kdata *= density

                if weights is not None:
                    curr_weights = weights[ts]
                    curr_kdata = curr_kdata.reshape(curr_weights.shape + (-1,))
                    curr_weights = np.expand_dims(curr_weights, axis=-1)
                    curr_kdata *= curr_weights
                    # curr_kdata = curr_kdata.reshape(ntimesteps, -1)

                curr_kdata = curr_kdata.flatten()

                if len(size) == 3:
                    t = traj[ts]
                    fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], curr_kdata, size)
                else:
                    t = traj[ts]
                    fk = finufft.nufft3d1(t[:, 0], t[:, 1], curr_kdata, size)
                del curr_kdata

                print(fk.shape)
                images_series_rebuilt[ts] += b1_all_slices[k].conj() * fk
    images_series_rebuilt /= num_k_samples
    return images_series_rebuilt



def undersampling_operator_new(volumes, trajectory, b1_all_slices, density_adj=True, weights=None,
                               retained_timesteps=None, ntimesteps=None, light_memory_usage=True):
    """
    returns A.H @ W @ A @ volumes where A=F Fourier + sampling operator and W correspond to radial density adjustment
    """
    if ntimesteps is None:
        ntimesteps = volumes.shape[0]
    size = volumes.shape[1:]
    nb_channels = b1_all_slices.shape[0]

    traj = trajectory.get_traj()
    traj = traj.reshape(ntimesteps, -1, 3)
    if retained_timesteps is not None:
        traj = traj[retained_timesteps]
        ntimesteps = len(retained_timesteps)

    npoint = trajectory.paramDict["npoint"]

    if volumes.dtype == "complex64":
        traj = traj.astype("float32")
        b1_all_slices = b1_all_slices.astype("complex64")
    num_k_samples = traj.shape[1]

    output_shape = (ntimesteps,) + size
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    for k in tqdm(range(nb_channels)):
        if not (light_memory_usage):
            curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
            print(curr_volumes.shape)
            if len(size) == 3:
                curr_kdata = [finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], v) for (t, v) in zip(traj, curr_volumes)]
            else:
                curr_kdata = [finufft.nufft2d2(t[:, 0], t[:, 1], v) for (t, v) in zip(traj, curr_volumes)]
            curr_kdata = np.array(curr_kdata)
            print(curr_kdata.shape)
            if density_adj:
                curr_kdata = curr_kdata.reshape(ntimesteps, -1, npoint)
                density = np.abs(np.linspace(-1, 1, npoint))
                density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
                curr_kdata *= density
            curr_kdata = curr_kdata.reshape(ntimesteps, -1)
            if weights is not None:
                curr_kdata = curr_kdata.reshape(weights.shape + (-1,))
                weights = np.expand_dims(weights, axis=-1)
                curr_kdata *= weights
                curr_kdata = curr_kdata.reshape(ntimesteps, -1)

            if len(size) == 3:
                fk = [finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], kd, size) for (t, kd) in zip(traj, curr_kdata)]
            else:
                fk = [finufft.nufft2d1(t[:, 0], t[:, 1], kd, size) for (t, kd) in zip(traj, curr_kdata)]
            fk = np.array(fk)
            print(fk.shape)
            images_series_rebuilt += b1_all_slices[k].conj() * fk

        else:
            for ts in tqdm(range(ntimesteps)):
                curr_volumes = volumes[ts] * b1_all_slices[k]
                print(curr_volumes.shape)
                if len(size) == 3:
                    t = traj[ts]
                    curr_kdata = finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], curr_volumes)
                else:
                    t = traj[ts]
                    curr_kdata = finufft.nufft3d2(t[:, 0], t[:, 1], curr_volumes)
                del curr_volumes
                print(curr_kdata.shape)

                if density_adj:
                    curr_kdata = curr_kdata.reshape(-1, npoint)
                    density = np.abs(np.linspace(-1, 1, npoint))
                    density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
                    curr_kdata *= density

                if weights is not None:
                    curr_weights = weights[ts]
                    curr_kdata = curr_kdata.reshape(curr_weights.shape + (-1,))
                    curr_weights = np.expand_dims(curr_weights, axis=-1)
                    curr_kdata *= curr_weights
                    # curr_kdata = curr_kdata.reshape(ntimesteps, -1)

                curr_kdata = curr_kdata.flatten()

                if len(size) == 3:
                    t = traj[ts]
                    fk = finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], curr_kdata, size)
                else:
                    t = traj[ts]
                    fk = finufft.nufft3d1(t[:, 0], t[:, 1], curr_kdata, size)
                del curr_kdata

                print(fk.shape)
                images_series_rebuilt[ts] += b1_all_slices[k].conj() * fk
    images_series_rebuilt /= num_k_samples
    return images_series_rebuilt


def simulate_undersampled_images(kdata,trajectory,size,density_adj=True,useGPU=False,eps=1e-6):
    # Strong Assumption : from one time step to the other, the sampling is just rotated, hence voronoi volumes can be calculated only once
    print("Simulating Undersampled Images")
    traj=trajectory.get_traj_for_reconstruction()

    if not(len(kdata)==len(traj)):
        kdata=np.array(kdata).reshape(len(traj),-1)

    kdata = np.array(kdata) / (2*np.pi) **2

    if density_adj:
        print("Performing density adjustment using Voronoi cells")
        #density = voronoi_volumes(np.transpose(np.array([traj[0,:, 0], traj[0,:, 1]])),min_x=-np.pi,min_y=-np.pi,max_x=np.pi,max_y=np.pi)[0]
        #if traj[0].shape[-1] == 2:#2D
        #    density = [voronoi_volumes(traj[i],min_x=-np.pi,min_y=-np.pi,max_x=np.pi,max_y=np.pi)[0] for i in tqdm(range(len(kdata)))]
        #else:#3D
        density = [
                voronoi_volumes(traj[i], min_x=-np.pi, min_y=-np.pi,
                                max_x=np.pi, max_y=np.pi, min_z=-np.pi, max_z=np.pi)[0] for i in
                tqdm(range(len(kdata)))]
        kdata = np.array([k * density[i] for i, k in enumerate(kdata)])

    #kdata = (normalize_image_series(np.array(kdata)))

    if traj[0].shape[-1] == 2:  # 2D
        if not (useGPU):
            images_series_rebuilt = [
                finufft.nufft2d1(t[:, 0], t[:, 1], s, size)
                for t, s in zip(traj, kdata)
            ]
        else:
            N1, N2 = size[0], size[1]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64
            fk_gpu = GPUArray((N1, N2), dtype=complex_dtype)
            images_GPU = []
            for i in list(range(len(kdata))):
                c_retrieved = kdata[i]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]

                # Cast to desired datatype.
                kx = kx.astype(dtype)
                ky = ky.astype(dtype)
                c_retrieved = c_retrieved.astype(complex_dtype)

                # Allocate memory for the uniform grid on the GPU.

                # Initialize the plan and set the points.
                plan = cufinufft(1, (N1, N2), 1, eps=eps, dtype=dtype)
                plan.set_pts(to_gpu(kx), to_gpu(ky))

                # Execute the plan, reading from the strengths array c and storing the
                # result in fk_gpu.
                plan.execute(to_gpu(c_retrieved), fk_gpu)

                fk = np.squeeze(fk_gpu.get())
                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)
    elif traj[0].shape[-1] == 3:  # 3D
        if not (useGPU):
            # images_series_rebuilt = [
            #    finufft.nufft3d1(t[:,2],t[:, 0], t[:, 1], s, size)
            #    for t, s in zip(traj, kdata)
            # ]

            images_series_rebuilt = []
            for t, s in tqdm(zip(traj, kdata)):
                images_series_rebuilt.append(finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], s, size))

        else:
            N1, N2, N3 = size[0], size[1], size[2]
            dtype = np.float32  # Datatype (real)
            complex_dtype = np.complex64

            images_GPU = []
            for i in list(range(len(kdata))):
                fk_gpu = GPUArray((N1, N2, N3), dtype=complex_dtype)
                c_retrieved = kdata[i]
                kx = traj[i, :, 0]
                ky = traj[i, :, 1]
                kz = traj[i, :, 2]

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

                images_GPU.append(fk)
                plan.__del__()
            images_series_rebuilt = np.array(images_GPU)

    # images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))

    return np.array(images_series_rebuilt)

def plot_evolution_params(map_ref, mask_ref, all_maps, maskROI=None, adj_wT1=True, title="Evolution",metric="R2", fat_threshold=0.7,
                          proj_on_mask1=True, fontsize=5, figsize=(15, 40),save=False):
    keys_1 = set(map_ref.keys())
    keys_2 = set(all_maps[0][0].keys())
    nb_keys = len(keys_1 & keys_2)
    fig, ax = plt.subplots(nb_keys, figsize=figsize)

    if maskROI is None:
        maskROI = buildROImask(map_ref)

    for i, k in enumerate(keys_1 & keys_2):
        print(i)
        result_list = []
        std_list = []
        it_list = []
        for it, value in all_maps.items():

            map2 = value[0]
            mask2 = value[1] > 0

            mask_union = mask_ref | mask2
            mat_obs = makevol(map_ref[k], mask_ref)
            mat_pred = makevol(map2[k], mask2)
            mat_ROI = makevol(maskROI, mask_ref)
            if proj_on_mask1:
                mat_pred = mat_pred * (mask_ref * 1)
                mat_obs = mat_obs * (mask_ref * 1)
                mat_ROI = mat_ROI * (mask_ref * 1)
                mask_union = mask_ref

            obs = mat_obs[mask_union]
            pred = mat_pred[mask_union]
            maskROI_current = mat_ROI[mask_union]

            if adj_wT1 and k == "wT1":
                ff = makevol(map_ref["ff"], mask_ref)
                ff = ff[mask_union]
                obs = obs[ff < fat_threshold]
                pred = pred[ff < fat_threshold]
                maskROI_current = maskROI_current[ff < fat_threshold]

            df_obs = pd.DataFrame(columns=["Data", "Groups"],
                                  data=np.stack([obs.flatten(), maskROI_current.flatten()], axis=-1))
            df_pred = pd.DataFrame(columns=["Data", "Groups"],
                                   data=np.stack([pred.flatten(), maskROI_current.flatten()], axis=-1))
            mean_obs = np.array(df_obs.groupby("Groups").mean())[1:]
            mean_pred = np.array(df_pred.groupby("Groups").mean())[1:]

            x_min = np.min(mean_obs)
            x_max = np.max(mean_pred)

            if x_min == x_max:
                fig.delaxes(ax[i])
                break

            if metric == "R2":
                mean = np.mean(mean_obs)
                ss_tot = np.sum((mean_obs - mean) ** 2)
                ss_res = np.sum((mean_obs - mean_pred) ** 2)
                bias = np.mean((mean_pred - mean_obs))
                r_2 = 1 - ss_res / ss_tot
                result_list.append(r_2)
                it_list.append(it)



            elif metric == "RMSE":
                print(obs.shape)
                print(pred.shape)
                df_error = pd.DataFrame(columns=["Data", "Groups"],
                                        data=np.stack(
                                            [(pred.flatten() - obs.flatten()) ** 2, maskROI_current.flatten()],
                                            axis=-1))
                errors = np.sqrt(np.array(df_error.groupby("Groups").mean())[1:])
                error = np.mean(errors)
                std_error = np.std(errors)
                result_list.append(error)
                std_list.append(std_error)
                it_list.append(it)


            else:
                raise ValueError("Metric should be RMSE or R2")

        if x_min == x_max:
            continue

        if metric == "R2":
            it_list, result_list = tuple(zip(*sorted(zip(it_list, result_list))))
            ax[i].plot(it_list, result_list, "r")
        elif metric == "RMSE":
            # ax[i].plot(range(n_it), result_list, "r")
            it_list, result_list,std_list=tuple(zip(*sorted(zip(it_list, result_list,std_list))))
            ax[i].errorbar(it_list, result_list, std_list)

        ax[i].set_title(k + " Evolution over Iteration", fontsize=2 * fontsize)
        ax[i].tick_params(axis='x', labelsize=fontsize)
        ax[i].tick_params(axis='y', labelsize=fontsize)

    plt.suptitle("{} : {}".format(title,metric))

    if save :
        plt.savefig("./figures/{} : {}".format(title,metric))


def create_cuda_context():
    pycuda.driver.init()
    dev=pycuda.driver.Device(0)
    context=dev.make_context()
    return context

def wavelet_denoising(image,retained_coef=0.99,level=3):
    c = pywt.wavedec2(image, 'db2', level=level)
    arr, slices = pywt.coeffs_to_array(c)

    #Selection of the appropriate cut off
    sorted_coef = np.sort(np.abs(arr.flatten()))[::-1]
    cum_sum = np.cumsum(sorted_coef)
    cum_sum = cum_sum / cum_sum[-1]
    index_cut = (cum_sum > retained_coef).sum()
    value = sorted_coef[index_cut]
    # Cut off
    arr_cut = arr.copy()
    arr_cut[np.abs(arr_cut) < value] = 0

    # sorted_coef = np.sort(np.abs(arr_cut.flatten()))
    # plt.plot(sorted_coef)

    #Image reconstruction
    coef_cut = pywt.array_to_coeffs(arr_cut, slices, output_format='wavedec2')
    image_cut = pywt.waverec2(coef_cut, 'db2')
    return image_cut

def calculate_condition_mvt_correction(t,transf,perc):


    shifts = transf(t.flatten().reshape(-1, 1))[:, 1]

    # traj_for_selection=traj_for_selection.reshape(t.shape+traj_for_selection.shape[-2:])
    # traj_for_selection=traj_for_selection.reshape((m.paramDict["nb_rep"],ntimesteps,-1)+traj_for_selection.shape[-2:])
    #
    # kdata_for_selection=kdata_for_selection.reshape(t.shape+kdata_for_selection.shape[-1:])
    # kdata_for_selection=kdata_for_selection.reshape((m.paramDict["nb_rep"],ntimesteps,-1)+kdata_for_selection.shape[-1:])

    threshold = np.percentile(shifts, perc)
    cond = (shifts > threshold)
    return cond


def correct_mvt_kdata(kdata,trajectory,cond,ntimesteps,density_adj=True,log=False):

    kdata=np.array(kdata)
    traj=trajectory.get_traj()

    mode=trajectory.paramDict["mode"]
    incoherent=trajectory.paramDict["incoherent"]


    nb_rep = int(cond.shape[0]/traj.shape[0])
    npoint = int(traj.shape[1] / nb_rep)
    nspoke = int(traj.shape[0] / ntimesteps)

    traj_for_selection = np.array(groupby(traj, npoint, axis=1))
    kdata_for_selection = np.array(groupby(kdata, npoint, axis=1))

    traj_for_selection = traj_for_selection.reshape(cond.shape[0], -1, 3)
    kdata_for_selection = kdata_for_selection.reshape(cond.shape[0], -1)

    indices = np.unravel_index(np.argwhere(cond).T,(nb_rep,ntimesteps,nspoke))
    retained_indices = np.squeeze(np.array(indices).T)

    traj_retained = traj_for_selection[cond, :, :]
    kdata_retained = kdata_for_selection[cond, :]

    ## DENSITY CORRECTION
    if density_adj:
        df = pd.DataFrame(columns=["rep", "ts", "spoke", "kz", "theta"], index=range(nb_rep * ntimesteps * nspoke))
        df["rep"] = np.repeat(list(range(nb_rep)), ntimesteps * nspoke)
        df["ts"] = list(np.repeat(list(range(ntimesteps)), (nspoke))) * nb_rep
        df["spoke"] = list(range(nspoke)) * nb_rep * ntimesteps

        df["kz"] = traj_for_selection[:, :, 2][:, 0]
        golden_angle = 111.246 * np.pi / 180

        if not(incoherent):
            df["theta"] = np.array(list(np.mod(np.arange(0, int(df.shape[0] / nb_rep)) * golden_angle, np.pi)) * nb_rep)
        elif incoherent:
            if mode=="old":
                df["theta"] = np.array(list(np.mod(np.arange(0, int(df.shape[0])) * golden_angle, np.pi)))
            elif mode=="new":
                df["theta"] = np.mod((np.array(list(np.arange(0, nb_rep) * golden_angle)).reshape(-1,1)+np.array(list(np.arange(0, int(df.shape[0] / nb_rep)) * golden_angle)).reshape(1,-1)).flatten(),np.pi)


        df_retained = df.iloc[np.nonzero(cond)]
        kz_by_timestep = df_retained.groupby("ts")["kz"].unique()
        theta_by_rep_timestep = df_retained.groupby(["ts", "rep"])["theta"].unique()

        df_retained = df_retained.join(kz_by_timestep, on="ts", rsuffix="_s")
        df_retained = df_retained.join(theta_by_rep_timestep, on=["ts", "rep"], rsuffix="_s")

        # Theta weighting
        df_retained["theta_s"] = df_retained["theta_s"].apply(lambda x: np.sort(x))
        #df_retained["theta_s"] = df_retained["theta_s"].apply(
        #    lambda x: np.concatenate([[x[-1] - np.pi], x, [x[0] + np.pi]]))
        df_retained["theta_s"] = df_retained["theta_s"].apply(
                lambda x: np.unique(np.concatenate([[0], x, [np.pi]])))
        diff_theta=(df_retained.theta - df_retained["theta_s"])
        theta_inside_boundary=(df_retained["theta"]!=0)*(df_retained["theta"]!=np.pi)
        df_retained["theta_inside_boundary"] = theta_inside_boundary

        min_theta = df_retained.groupby(["ts", "rep"])["theta"].min()
        max_theta = df_retained.groupby(["ts", "rep"])["theta"].max()
        df_retained = df_retained.join(min_theta, on=["ts", "rep"], rsuffix="_min")
        df_retained = df_retained.join(max_theta, on=["ts", "rep"], rsuffix="_max")
        is_min_theta = (df_retained["theta"]==df_retained["theta_min"])
        is_max_theta = (df_retained["theta"] == df_retained["theta_max"])
        df_retained["is_min_theta"] = is_min_theta
        df_retained["is_max_theta"] = is_max_theta

        df_retained["theta_weight"] = theta_inside_boundary*diff_theta.apply(lambda x: (np.sort(x[x>=0])[1]+np.sort(-x[x<=0])[1])/2 if ((x>=0).sum()>1) and ((x<=0).sum()>1) else 0)+\
                                      (1-theta_inside_boundary)*diff_theta.apply(lambda x: np.sort(np.abs(x))[1]/2)

        df_retained["theta_weight_before_correction"] = df_retained["theta_weight"]

        df_retained["theta_weight"] = df_retained["theta_weight"]+ (theta_inside_boundary)* ((is_min_theta)*df_retained["theta"]+(is_max_theta)*(np.pi-df_retained["theta"]))/2

        df_retained.loc[df_retained["theta_weight"].isna(), "theta_weight"] = 1.0
        sum_weights = df_retained.groupby(["ts", "rep"])["theta_weight"].sum()
        df_retained = df_retained.join(sum_weights, on=["ts", "rep"], rsuffix="_sum")
        #df_retained["theta_weight"] = df_retained["theta_weight"] / df_retained["theta_weight_sum"]

        # KZ weighting
        #df_retained.loc[df_retained.ts == 138].to_clipboard()
        df_retained["kz_s"] = df_retained["kz_s"].apply(lambda x: np.unique(np.concatenate([[-np.pi], x, [np.pi]])))
        diff_kz=(df_retained.kz - df_retained["kz_s"])
        kz_inside_boundary=(df_retained["kz"].abs()!=np.pi)
        df_retained["kz_inside_boundary"]=kz_inside_boundary

        min_kz = df_retained.groupby(["ts"])["kz"].min()
        max_kz = df_retained.groupby(["ts"])["kz"].max()
        df_retained = df_retained.join(min_kz, on=["ts"], rsuffix="_min")
        df_retained = df_retained.join(max_kz, on=["ts"], rsuffix="_max")

        is_min_kz = (df_retained["kz"] == df_retained["kz_min"])
        is_max_kz = (df_retained["kz"] == df_retained["kz_max"])

        df_retained["is_min_kz"] = is_min_kz
        df_retained["is_max_kz"] = is_max_kz

        df_retained["kz_weight"] = kz_inside_boundary*diff_kz.apply(lambda x: (np.sort(x[x >= 0])[1] + np.sort(-x[x <= 0])[1]) / 2 if ((x >= 0).sum() > 1) and (
            ((x <= 0).sum() > 1)) else 0)+(1-kz_inside_boundary)*diff_kz.apply(lambda x: np.sort(np.abs(x))[1]/2)



        df_retained["kz_weight"] = df_retained["kz_weight"] + (kz_inside_boundary) * (
                    (is_min_kz) * (df_retained["kz"]+np.pi) + (is_max_kz) * (np.pi - df_retained["kz"])) / 2

        df_retained.loc[df_retained["kz_weight"].isna(), "kz_weight"] = 1.0
        sum_weights = df_retained.drop_duplicates(subset=["kz","ts"])
        sum_weights = sum_weights.groupby(["ts"])["kz_weight"].apply(lambda x: x.sum())
        df_retained = df_retained.join(sum_weights, on=["ts"], rsuffix="_sum")
        #df_retained["kz_weight"] = df_retained["kz_weight"] / df_retained["kz_weight_sum"]

        if log:
            now = datetime.now()
            df_retained.to_csv("./log/df_density_correction_{}.csv".format(now))

    dico_traj = {}
    dico_kdata = {}
    for i, index in enumerate(retained_indices):
        curr_slice = index[0]
        ts = index[1]
        curr_spoke = index[2]
        if ts not in dico_traj:
            dico_traj[ts] = []
            dico_kdata[ts] = []

        # dico_traj[ts]=[*dico_traj[ts],*traj_retained[i]]
        # dico_kdata[ts]=[*dico_kdata[ts],*kdata_retained[i]]



        dico_traj[ts].append(traj_retained[i])
        if density_adj:
            theta_weight = df_retained.iloc[i]["theta_weight"]
            kz_weight = df_retained.iloc[i]["kz_weight"]
            dico_kdata[ts].append(kdata_retained[i]*theta_weight*kz_weight)
        else:
            dico_kdata[ts].append(kdata_retained[i])

    retained_timesteps = list(dico_traj.keys())
    retained_timesteps.sort()

    traj_retained_final = []
    kdata_retained_final = []

    # for ts in tqdm(range(len(retained_timesteps))):
    #     traj_retained_final.append(np.array(dico_traj[ts]))
    #     kdata_retained_final.append(np.array(dico_kdata[ts]))

    for ts in tqdm(retained_timesteps):
        traj_retained_final.append(np.array(dico_traj[ts]).flatten().reshape(-1, 3))
        kdata_retained_final.append(np.array(dico_kdata[ts]).flatten())

    traj_retained_final = np.array(traj_retained_final)
    kdata_retained_final = np.array(kdata_retained_final)

    return kdata_retained_final,traj_retained_final,retained_timesteps





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
        plt.show()



def J_fourier(m, traj, kdata):

    Fu_m = finufft.nufft2d2(traj[:, 0], traj[:, 1], m)
    return np.linalg.norm(Fu_m - kdata) ** 2


def J_sparse(m, typ="db4", mode="periodization", mu=1e-6):
    return N(coef_to_array(psi(m, typ, mode=mode)), mu)


def grad_J_sparse(m, typ="db4", mode="periodization", mu=1e-6):
    psi_m = coef_to_array(psi(m, typ, mode=mode))
    return np.real(inv_psi(array_to_coef(W(psi_m, mu) * psi_m)))


def grad_J_fourier(m, traj, kdata, npoint=512, density_adj=True):
    image_size = m.shape
    Fu_m = finufft.nufft2d2(traj[:, 0], traj[:, 1], m)
    kdata_error = Fu_m - kdata
    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        kdata_error = (np.reshape(kdata_error, (-1, npoint)) * density).flatten()
    error_volume = finufft.nufft2d1(traj[:, 0], traj[:, 1], kdata_error, image_size)

    return 2 * np.real(error_volume)

def psi(m,typ='db4',mode="periodization"):
    coef = pywt.dwt2(m, typ,mode=mode)
    return coef

def inv_psi(c,typ='db4',mode="periodization"):
    m = pywt.idwt2(c, typ,mode=mode)
    return m

def N(x,mu=1e-6):
    return np.sum(np.sqrt(np.abs(x)**2+mu))

def W(x,mu=1e-6):
    return 1/np.sqrt(np.abs(x)**2+mu)

def coef_to_array(c):
    cA, (cH, cV, cD) = c
    image_1 = np.concatenate([cA,cH],axis=1)
    image_2 = np.concatenate([cV,cD],axis=1)
    psi_m = np.concatenate([image_1,image_2],axis=0)
    return psi_m

def array_to_coef(array):
    N = array.shape[0]
    mid = int(N/2)
    cA = array[:mid,:mid]
    cH = array[:mid,mid:N]
    cV = array[mid:N,:mid]
    cD = array[mid:N,mid:N]
    c=cA, (cH, cV, cD)
    return c


def conjgrad(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False,plot=False,filename_save=None,folder_logs="./logs",folder_figures="./figures"):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log or plot:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
    
    if log:
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    if plot:
        plt.ion()
        fig, axs = plt.subplots(1, 2, figsize=(30, 10))
        axs[0].set_title("Evolution of cost function")
    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)
            if filename_save is not None:
                np.save(filename_save,m)
        t = t0
        J_m = J(m)
        print("J for iter {}: {}".format(k,J_m))
        J_m_next = J(m+t*d_m)
        slope = np.real(np.dot(g.flatten(),d_m.flatten()))
        if plot:
            axs[0].scatter(k,J_m,c="r",marker="+")
            axs[1].cla()
            axs[1].set_title("Line search for iteration {}".format(k))
            t_array = np.arange(0.,t0,t0/100)
            axs[1].plot(t_array,J_m+t_array*slope)
            axs[1].scatter(0,J_m,c="b",marker="x")
            plt.savefig(folder_figures+'/conjgrad_plot_{}.png'.format(date_time))

        while(J_m_next>J_m+alpha*t*slope):
            print(t)
            t = beta*t
            if plot:
                axs[1].scatter(t,J_m_next,c="b",marker="x")
                plt.savefig(folder_figures+'/conjgrad_plot_{}.png'.format(date_time))
            J_m_next=J(m+t*d_m)




        m = m + t*d_m
        g_prev = g
        g = grad_J(m)
        gamma = np.linalg.norm(g)**2/np.linalg.norm(g_prev)**2
        d_m = -g + gamma*d_m
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save(folder_logs+'/conjgrad_{}.npy'.format(date_time),norm_g_list)

    return m


def graddesc(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)
        t = t0
        J_m = J(m)
        print("J for iter {}: {}".format(k,J_m))
        while(J(m+t*d_m)>J_m+alpha*t*np.real(np.dot(g.flatten(),d_m.flatten()))):
            print(t)
            t = beta*t

        m = m + t*d_m
        g = grad_J(m)
        d_m = -g
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save('./logs/graddesc_{}.npy'.format(date_time),norm_g_list)

    return m

def graddesc_linsearch(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.05,beta=0.6,t0=1,log=False):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)

        J_m = J(m)
        t = t0
        print("J for iter {}: {}".format(k,J_m))
        while(J(m+t*d_m)>J_m+alpha*t*np.real(np.dot(g.flatten(),d_m.flatten()))):
            print(t)
            t = beta*t

        m = m + t*d_m
        g = grad_J(m)
        d_m = -g
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save('./logs/graddesc_linsearch_{}.npy'.format(date_time),norm_g_list)

    return m


def graddesc(J,grad_J,m0,tolgrad=1e-4,maxiter=100,alpha=0.1,log=False):
    '''
    J : function from W (domain of m) to R
    grad_J : function from W to W - gradient of J
    m0 : initial value of m
    '''
    k=0
    m=m0
    if log:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        norm_g_list=[]

    g=grad_J(m)
    d_m=-g
    #store = [m]

    while (np.linalg.norm(g)>tolgrad)and(k<maxiter):
        norm_g = np.linalg.norm(g)
        if log:
            print("################ Iter {} ##################".format(k))
            norm_g_list.append(norm_g)
        print("Grad norm for iter {}: {}".format(k,norm_g))
        if k%10==0:
            print(k)

        J_m = J(m)
        print("J for iter {}: {}".format(k,J_m))

        m = m + alpha*d_m/norm_g*np.linalg.norm(J_m)
        g = grad_J(m)
        d_m = -g
        k=k+1
        #store.append(m)

    if log:
        norm_g_list=np.array(norm_g_list)
        np.save('./logs/graddesc_{}.npy'.format(date_time),norm_g_list)

    return m

def simulate_image_series_from_maps(map_rebuilt,mask_rebuilt,window=8):
    keys_simu = list(map_rebuilt.keys())
    values_simu = [makevol(map_rebuilt[k], mask_rebuilt > 0) for k in keys_simu]
    map_for_sim = dict(zip(keys_simu, values_simu))

    map_ = MapFromDict("RebuiltMapFromParam", paramMap=map_for_sim)
    map_.buildParamMap()

    map_.build_ref_images(seq=seq)
    rebuilt_image_series = map_.images_series
    rebuilt_image_series= [np.mean(gp, axis=0) for gp in groupby(rebuilt_image_series, window)]
    rebuilt_image_series=np.array(rebuilt_image_series)
    return rebuilt_image_series,map_for_sim



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




def mrisensesim(size, ncoils=8, array_cent=None, coil_width=2, n_rings=None, phi=0):
    """Apply simulated sensitivity maps. Based on a script by Florian Knoll.
    Args:
        size (tuple): Size of the image array for the sensitivity coils.
        nc_range (int, default: 8): Number of coils to simulate.
        array_cent (tuple, default: 0): Location of the center of the coil
            array.
        coil_width (double, default: 2): Parameter governing the width of the
            coil, multiplied by actual image dimension.
        n_rings (int, default: ncoils // 4): Number of rings for a
            cylindrical hardware set-up.
        phi (double, default: 0): Parameter for rotating coil geometry.
    Returns:
        list: A list of dimensions (ncoils, (N)), specifying spatially-varying
            sensitivity maps for each coil.
    """
    if array_cent is None:
        c_shift = [0, 0, 0]
    elif len(array_cent) < 3:
        c_shift = array_cent + (0,)
    else:
        c_shift = array_cent

    c_width = coil_width * min(size)

    if len(size) > 2:
        if n_rings is None:
            n_rings = ncoils // 4

    c_rad = min(size[0:1]) / 2
    smap = []
    if len(size) > 2:
        zz, yy, xx = np.meshgrid(
            range(size[2]), range(size[1]), range(size[0]), indexing="ij"
        )
    else:
        yy, xx = np.meshgrid(range(size[1]), range(size[0]), indexing="ij")

    if ncoils > 1:
        x0 = np.zeros((ncoils,))
        y0 = np.zeros((ncoils,))
        z0 = np.zeros((ncoils,))

        for i in range(ncoils):
            if len(size) > 2:
                theta = np.radians((i - 1) * 360 / (ncoils + n_rings) + phi)
            else:
                theta = np.radians((i - 1) * 360 / ncoils + phi)
            x0[i] = c_rad * np.cos(theta) + size[0] / 2
            y0[i] = c_rad * np.sin(theta) + size[1] / 2
            if len(size) > 2:
                z0[i] = (size[2] / (n_rings + 1)) * (i // n_rings)
                smap.append(
                    np.exp(
                        -1
                        * ((xx - x0[i]) ** 2 + (yy - y0[i]) ** 2 + (zz - z0[i]) ** 2)
                        / (2 * c_width)
                    )
                )
            else:
                smap.append(
                    np.exp(-1 * ((xx - x0[i]) ** 2 + (yy - y0[i]) ** 2) / (2 * c_width))
                )
    else:
        x0 = c_shift[0]
        y0 = c_shift[1]
        z0 = c_shift[2]
        if len(size) > 2:
            smap = np.exp(
                -1 * ((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2) / (2 * c_width)
            )
        else:
            smap = np.exp(-1 * ((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * c_width))

    side_mat = np.arange(int(size[0] // 2) - 20, 1, -1)
    side_mat = np.reshape(side_mat, (1,) + side_mat.shape) * np.ones(shape=(size[1], 1))
    cent_zeros = np.zeros(shape=(size[1], size[0] - side_mat.shape[1] * 2))

    ph = np.concatenate((side_mat, cent_zeros, side_mat), axis=1) / 10
    if len(size) > 2:
        ph = np.reshape(ph, (1,) + ph.shape)

    for i, s in enumerate(smap):
        smap[i] = s * np.exp(i * 1j * ph * np.pi / 180)

    return smap

def calc_grad_entropy(v,w=None):
    ndim = v.ndim
    grad_norm = 0
    if w is None:
        w=v.ndim*[1]
    for axis in range(ndim):
        pad = v.ndim * [(0, 0)]
        pad[axis] = (0, 1)
        pad = tuple(pad)
        grad = np.diff(np.pad(v, pad, mode="constant"), axis=axis)

        grad_norm += (w[axis]*np.abs(grad)) ** 2

    grad_norm = np.sqrt(grad_norm)
    p = grad_norm / np.sum(grad_norm)
    p[p==0]=1 #pixel with value 0 are crashing the algo
    return np.sum(-np.log2(p) * p)

def calc_entropy(v):
    ndim = v.ndim
    #grad_norm = 0
    p = np.abs(v)
    p/=np.max(p)
    #p[p==0]=1 #pixel with value 0 are crashing the algo
    return np.sum(-np.log2(p) * p)


def build_phi(dictfile,L0):

    #mrfdict = dictsearch.Dictionary()
    keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.1))

    import dask.array as da
    u,s,vh = da.linalg.svd(da.asarray(values))

    vh=np.array(vh)
    s=np.array(s)

    phi=vh[:L0]
    #phi=vh

    filename_phi=str.split(dictfile, ".dict")[0] + "_phi_L0_{}.npy".format(L0)
    np.save(filename_phi,phi)

    return phi



def calc_grad_entropy(v,w=None):
    ndim = v.ndim
    grad_norm = 0
    if w is None:
        w=v.ndim*[1]
    for axis in range(ndim):
        pad = v.ndim * [(0, 0)]
        pad[axis] = (0, 1)
        pad = tuple(pad)
        grad = np.diff(np.pad(v, pad, mode="constant"), axis=axis)

        grad_norm += (w[axis]*np.abs(grad)) ** 2

    grad_norm = np.sqrt(grad_norm)
    p = grad_norm / np.sum(grad_norm)
    p[p==0]=1 #pixel with value 0 are crashing the algo
    return np.sum(-np.log2(p) * p)

def calc_entropy(v):
    ndim = v.ndim
    #grad_norm = 0
    p = np.abs(v)
    p/=np.max(p)
    #p[p==0]=1 #pixel with value 0 are crashing the algo
    return np.sum(-np.log2(p) * p)

def build_dico_seqParams(filename,folder):
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

        twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock", "sKSpace"],
                                   optional_additional_arrays=["SliceThickness"])

        if np.max(np.argwhere(np.array(twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]) > 0)) >= 16:
            use_navigator_dll = True
        else:
            use_navigator_dll = False

        alFree = twix[-1]["hdr"]["Meas"]["sWipMemBlock"]["alFree"]
        x_FOV = twix[-1]["hdr"]["Meas"]["RoFOV"]
        y_FOV = twix[-1]["hdr"]["Meas"]["PeFOV"]
        z_FOV = twix[-1]["hdr"]["Meas"]["SliceThickness"][0]

        nb_part = twix[-1]["hdr"]["Meas"]["Partitions"]

        dico_seqParams = {"alFree": alFree, "x_FOV": x_FOV, "y_FOV": y_FOV, "z_FOV": z_FOV,
                          "use_navigator_dll": use_navigator_dll, "nb_part": nb_part}

        del alFree

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

    return dico_seqParams


def build_data(filename,folder, nb_segments, nb_gating_spokes):
    filename_save = str.split(filename, ".dat")[0] + ".npy"
    # filename_nav_save=str.split(base_folder+"/phantom.001.v1/phantom.001.v1.dat",".dat") [0]+"_nav.npy"
    filename_nav_save = str.split(filename, ".dat")[0] + "_nav.npy"



    if str.split(filename_save, "/")[-1] not in os.listdir(folder):
        print("Building data")
        if 'twix' not in locals():
            print("Re-loading raw data")
            twix = twixtools.read_twix(filename)

        mdb_list = twix[-1]['mdb']
        if nb_gating_spokes == 0:
            data = []

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    data.append(mdb)

            data_for_nav = None


        else:
            print("Reading Navigator Data....")
            data_for_nav = []
            data = []
            nav_size_initialized = False
            # k = 0
            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh[14][9]):
                        mdb_data_shape = mdb.data.shape
                        mdb_dtype = mdb.data.dtype
                        nav_size_initialized = True
                        break

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh[14][9]):
                        data.append(mdb)
                    else:
                        data_for_nav.append(mdb)
                        data.append(np.zeros(mdb_data_shape, dtype=mdb_dtype))

                    # print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                    # k += 1
            data_for_nav = np.array([mdb.data for mdb in data_for_nav])
            data_for_nav = data_for_nav.reshape((-1, int(nb_gating_spokes)) + data_for_nav.shape[1:])

            if data_for_nav.ndim == 3:
                data_for_nav = np.expand_dims(data_for_nav, axis=-2)

            data_for_nav = np.moveaxis(data_for_nav, -2, 0)
            np.save(filename_nav_save, data_for_nav)

        data = np.array([mdb.data for mdb in data])
        data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
        data = np.moveaxis(data, 2, 0)
        data = np.moveaxis(data, 2, 1)

        del mdb_list

        ##################################################
        try:
            del twix
        except:
            pass

        np.save(filename_save, data)
        #
        ##################################################
        #
        # Parsed_File = rT.map_VBVD(filename)
        # idx_ok = rT.detect_TwixImg(Parsed_File)
        # start_time = time.time()
        # data = Parsed_File[str(idx_ok)]["image"].readImage()
        # elapsed_time = time.time()
        # elapsed_time = elapsed_time - start_time
        #
        # progress_str = "Data read in %f s \n" % round(elapsed_time, 2)
        # print(progress_str)
        #
        # data = np.squeeze(data)
        #
        # if nb_gating_spokes>0:
        #     data = np.moveaxis(data, 0, -1)
        #     data = np.moveaxis(data, -2, 0)
        #
        # else:
        #     data = np.moveaxis(data, 0, -1)

        # np.save(filename_save, data)


    else:
        print("Loading previously built data")
        data = np.load(filename_save)
        if nb_gating_spokes > 0:
            data_for_nav = np.load(filename_nav_save)
        else:
            data_for_nav=None

    return data, data_for_nav

def select_patch(k,volume,window=(2,5,5)):
    shape=volume.shape[-3:]
    d_z=window[0]
    d_x = window[1]
    d_y = window[2]
    #print(k)

    k_min_z_pixel=k[0]-d_z
    k_max_z_pixel = k[0] + d_z
    k_min_x_pixel = k[1] - d_x
    k_max_x_pixel = k[1] + d_x
    k_min_y_pixel = k[2] - d_y
    k_max_y_pixel = k[2] + d_y

    k_min_z=np.maximum(0,k_min_z_pixel)
    k_max_z = np.minimum(shape[0], k_max_z_pixel)
    pad_z_left = k_min_z - k_min_z_pixel
    pad_z_right=k_max_z_pixel-k_max_z

    k_min_x = np.maximum(0, k_min_x_pixel)
    k_max_x = np.minimum(shape[1], k_max_x_pixel)
    pad_x_left = k_min_x - k_min_x_pixel
    pad_x_right = k_max_x_pixel - k_max_x

    k_min_y = np.maximum(0, k_min_y_pixel)
    k_max_y = np.minimum(shape[2], k_max_y_pixel)
    pad_y_left = k_min_y - k_min_y_pixel
    pad_y_right = k_max_y_pixel - k_max_y

    patch=volume[...,k_min_z:k_max_z,k_min_x:k_max_x,k_min_y:k_max_y]
    #print(patch.shape)
    non_padded_dims=[(0,0)]*(patch.ndim-3)
    padding=tuple(non_padded_dims+[(pad_z_left,pad_z_right),(pad_x_left,pad_x_right),(pad_y_left,pad_y_right)])
    #print(padding)
    patch=np.pad(patch,padding,mode="edge")

    pixels=np.mgrid[k_min_z_pixel:k_max_z_pixel,k_min_x_pixel:k_max_x_pixel,k_min_y_pixel:k_max_y_pixel].reshape(3,-1)
    #print(pixels)
    return patch,pixels

def select_similar_patches(k,volume,volume_ref,window=(2,5,5),L=10,sliding_window=(2,10,10),steps=(3,3,3),quantile=None):
    original_patch,_=select_patch(k,volume_ref,window)
    array_k=np.array(k).reshape(-1,1)
    zxy = np.mgrid[-sliding_window[0]:(sliding_window[0]+1):steps[0], -sliding_window[1]:(sliding_window[1]+1):steps[1], -sliding_window[2]:(sliding_window[2]+1):steps[2]].reshape(3, -1)
    k_s = array_k + zxy
    shape=volume.shape[-3:]
    k_s[0, :] = np.maximum(0, k_s[0, :])
    k_s[0,:]=np.minimum(shape[0]-1,k_s[0, :])
    k_s[1, :] = np.maximum(0, k_s[1, :])
    k_s[1, :] = np.minimum(shape[1] - 1, k_s[1, :])
    k_s[2, :] = np.maximum(0, k_s[2, :])
    k_s[2, :] = np.minimum(shape[0] - 1, k_s[2, :])

    #print(k_s)
    k_s=k_s.T
    #print(k_s.shape)
    k_s=np.unique(k_s,axis=0)
    #print(k_s.shape)
    all_patches=[]
    for curr_k in k_s:
        #print(curr_k)
        curr_patch,_=select_patch(curr_k,volume_ref,window)
        all_patches.append(curr_patch)

    all_patches=np.array(all_patches)
    original_patch=np.expand_dims(original_patch,axis=0)
    #print(all_patches.shape)
    distances=original_patch-all_patches
    distances=distances.reshape(distances.shape[0],-1)
    distances=np.linalg.norm(distances,axis=-1)
    if quantile is None:
        retained_k_indices=np.argsort(distances)[:L]
    else:
        retained_k_indices=np.argwhere(distances<(np.percentile(distances,quantile))).flatten()
    #print(retained_k_indices)
    #print(distances.shape)
    #print(k_s.shape)
    retained_k_s = k_s[retained_k_indices]

    all_patches_retained = []
    pixels=[]
    #print(retained_k_s)
    for curr_k in retained_k_s:
        curr_patch,curr_pixels = select_patch(curr_k, volume, window)
        #print(curr_patch.shape)
        all_patches_retained.append(curr_patch)
        pixels.append(curr_pixels)

    return np.array(all_patches_retained),np.array(pixels)


def compute_low_rank_tensor(Sk_cur,variance_explained):
    Sk_1 = Sk_cur.reshape(Sk_cur.shape[0],-1)
    u_1, s_1, vh_1 = np.linalg.svd(Sk_1, full_matrices=False)

    Sk_2 = np.moveaxis(Sk_cur,1,0).reshape(Sk_cur.shape[1],-1)
    u_2, s_2, vh_2 = np.linalg.svd(Sk_2, full_matrices=False)

    Sk_3 = np.moveaxis(Sk_cur, 2, 0).reshape(Sk_cur.shape[2], -1)
    u_3, s_3, vh_3 = np.linalg.svd(Sk_3, full_matrices=False)



    cum_1=np.cumsum(s_1)/np.sum(s_1)
    ind_1 = (cum_1<variance_explained).sum()
    u_1 = u_1[:,:ind_1]

    cum_2=np.cumsum(s_2)/np.sum(s_2)
    ind_2 = (cum_2<variance_explained).sum()
    u_2 = u_2[:, :ind_2]

    cum_3 = np.cumsum(s_3) / np.sum(s_3)
    ind_3 = (cum_3 < variance_explained).sum()
    u_3 = u_3[:, :ind_3]

    res = tensor_product(Sk_cur, u_1.T.conj(), mode=1)
    res = tensor_product(res, u_2.T.conj(), mode=2)
    res = tensor_product(res, u_3.T.conj(), mode=3)
    res = tensor_product(res, u_1, mode=1)
    res = tensor_product(res, u_2, mode=2)
    res = tensor_product(res, u_3, mode=3)

    return res




def tensor_product(T,U,mode):
    T_indexing="ijk"
    index_summed=T_indexing[mode-1]
    T_new_indexing=T_indexing.replace(index_summed,"l")
    U_indexing="l"+index_summed

    result=np.einsum("{},{}->{}".format(T_indexing,U_indexing,T_new_indexing),T,U)
    return result



def J(m,traj,data,npoint,density_adj=True,useGPU=True,scaling=None):


    ntimesteps=traj.shape[0]
    if scaling is None:
        scaling=np.prod(m.shape[1:])


    FU=[]
    if m.dtype=="complex64":
        traj_kdata=traj.astype("float32")
    else:
        traj_kdata = copy(traj)

    if not(useGPU):
        for j in tqdm(range(ntimesteps)):
            FU.append(finufft.nufft3d2(traj_kdata[j,:, 2],traj_kdata[j,:, 0], traj_kdata[j,:, 1],m[j]))

    else:
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64
        N1, N2, N3 = m.shape[1], m.shape[2], m.shape[3]
        #M = traj.shape[0]
        #c_gpu = GPUArray((M), dtype=complex_dtype)
        FU = []
        for i in tqdm(range(ntimesteps)):
            fk = m[i, :, :, :]
            kx = traj[i,:, 0]
            ky = traj[i,:, 1]
            kz = traj[i,:, 2]

            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            fk = fk.astype(complex_dtype)
            c_gpu = GPUArray((kx.shape[0]), dtype=complex_dtype)

            plan = cufinufft(2, (N1, N2, N3), 1, eps=1e-6, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
            plan.execute(c_gpu, to_gpu(fk))
            c = np.squeeze(c_gpu.get())
            c_gpu.gpudata.free()
            FU.append(c)
            plan.__del__()

    FU=np.array(FU)

    kdata_error = FU - data.reshape(ntimesteps,-1)
    #kdata_ratio = FU/data.reshape(ntimesteps,-1)[:curr_ntimesteps]

    # return np.linalg.norm(kdata_error)**2
    kdata_error/=np.prod(m.shape[1:])
    #kdata_error /= np.sqrt(np.prod(data.shape[1:]))
    if density_adj:
        kdata_error = kdata_error.reshape(-1, npoint)
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_error.ndim - 1)))
        kdata_error *= np.sqrt(density)


    return np.linalg.norm(kdata_error) ** 2


def grad_J(m,traj,data,npoint,nspoke,nb_slices,density_adj=True,useGPU=False,scaling=None):
    ntimesteps = traj.shape[0]
    if scaling is None:
        scaling = np.prod(m.shape[1:])



    image_size=m.shape[1:]
    FU = []
    if m.dtype == "complex64":
        traj_kdata = traj.astype("float32")
    else:
        traj_kdata=copy(traj)

    if not (useGPU):
        for j in tqdm(range(ntimesteps)):
            FU.append(finufft.nufft3d2(traj_kdata[j, :, 2], traj_kdata[j, :, 0], traj_kdata[j, :, 1], m[j]))
    else:
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64
        N1, N2, N3 = m.shape[1], m.shape[2], m.shape[3]
        #M = traj[0].shape[0]
        #c_gpu = GPUArray((M), dtype=complex_dtype)
        FU = []
        for i in tqdm(range(ntimesteps)):
            fk = m[i, :, :, :]
            kx = traj_kdata[i,:, 0]
            ky = traj_kdata[i,:, 1]
            kz = traj_kdata[i,:, 2]

            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            fk = fk.astype(complex_dtype)

            c_gpu = GPUArray((kx.shape[0]), dtype=complex_dtype)

            plan = cufinufft(2, (N1, N2, N3), 1, eps=1e-6, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
            plan.execute(c_gpu, to_gpu(fk))
            c = np.squeeze(c_gpu.get())
            c_gpu.gpudata.free()
            FU.append(c)
            plan.__del__()

    FU = np.array(FU)

    kdata_error = FU - data.reshape(ntimesteps,-1)

    kdata_error/=scaling**2
    #kdata_error /= data.shape[1:]
    # return np.linalg.norm(kdata_error)**2
    if density_adj:
        kdata_error = kdata_error.reshape(-1, npoint)
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_error.ndim - 1)))
        kdata_error*=density
        kdata_error=kdata_error.reshape(ntimesteps,-1)


    dm=[]

    if not(useGPU):
        for j in tqdm(range(ntimesteps)):
            dm.append(finufft.nufft3d1(traj[j,:, 2], traj[j,:, 0], traj[j,:, 1], kdata_error[j], image_size))

    else:
        N1, N2, N3 = image_size[0], image_size[1], image_size[2]
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64

        for i in tqdm(range(ntimesteps)):
            fk_gpu = GPUArray(( N1, N2, N3), dtype=complex_dtype)
            c_retrieved = kdata_error[i]
            kx = traj[i,:, 0]
            ky = traj[i,:, 1]
            kz = traj[i,:, 2]
            #
            # print(fk_gpu.shape)
            # print(kx.shape)
            # print(c_retrieved.shape)

            # Cast to desired datatype.
            kx = kx.astype(dtype)
            ky = ky.astype(dtype)
            kz = kz.astype(dtype)
            c_retrieved = c_retrieved.astype(complex_dtype)

            # Allocate memory for the uniform grid on the GPU.
            c_retrieved_gpu = to_gpu(c_retrieved)

            # Initialize the plan and set the points.
            plan = cufinufft(1, (N1, N2, N3),1, eps=1e-6, dtype=dtype)
            plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

            # Execute the plan, reading from the strengths array c and storing the
            # result in fk_gpu.
            plan.execute(c_retrieved_gpu, fk_gpu)

            dm.append(np.squeeze(fk_gpu.get()))

            fk_gpu.gpudata.free()
            c_retrieved_gpu.gpudata.free()

            plan.__del__()


    return 2*np.array(dm)



def psiTV(t,alpha=1e-13):
    return np.sqrt(t+alpha**2)

def delta(m,axis,shift=1):
    return np.diff(m,axis=shift+axis,append=0)

def delta_adjoint(m,axis,shift=1):
    return -np.diff(m, axis=shift + axis, prepend=0)

def J_TV(m,axis,alpha=1e-13,is_weighted=True,mask=None,weights=None,shift=1):

    #global delta_m
    delta_m = delta(m, axis,shift)



    if is_weighted:
        if (mask is None)and(weights is None):
            raise ValueError("should provide mask or weights for calculating weighing mask")

        if mask is not None:
            bound_inf = np.min(np.argwhere(mask > 0)[:, axis])
            bound_sup = np.max(np.argwhere(mask > 0)[:, axis])
            weights = np.ones(delta_m.shape, dtype=delta_m.dtype)

            idx = [np.s_[:]] * weights.ndim
            idx[axis + shift] = np.s_[:bound_inf]
            weights[tuple(idx)] = 0

            idx = [np.s_[:]] * weights.ndim
            idx[axis + shift] = np.s_[bound_sup:]
            weights[tuple(idx)] = 0

    else:
        weights=1


    return np.sum(weights*psiTV(np.abs(delta_m)**2,alpha))

def grad_J_TV(m,axis,alpha=1e-13,is_weighted=True,mask=None,weights=None,shift=1):



    delta_m = delta(m,axis,shift)

    if is_weighted:
        if (mask is None)and(weights is None):
            raise ValueError("should provide mask or weights for calculating weighing mask")

        if mask is not None:
            bound_inf = np.min(np.argwhere(mask > 0)[:, axis])
            bound_sup = np.max(np.argwhere(mask > 0)[:, axis])
            weights = np.ones(delta_m.shape, dtype=delta_m.dtype)

            idx = [np.s_[:]]*weights.ndim
            idx[axis + shift] = np.s_[:bound_inf]
            weights[tuple(idx)] = 0


            idx = [np.s_[:]]*weights.ndim
            idx[axis + shift] = np.s_[bound_sup:]
            weights[tuple(idx)] = 0

    else:
        weights = 1

    W = psiTV(np.abs(delta_m)**2,alpha)

    return delta_adjoint(weights*delta_m/W,axis,shift)


def getPreScanImages(twix):

    '''
    from twix object read from twix tools, extract pre-scan data:
    data_noise : noise data reshaped into (nb_channels,-1) (to calculate noise channels correl)
    data_body : body coil prescan data reshape into (128,2,32,32)
    data_multicoil : body coil prescan data reshape into (128,nb_channels,32,32)
    '''

    mdb_list = twix[0]['mdb']

    dico_data_set0 = {}
    dico_data_set1 = {}
    noise_adj_0 = []
    noise_adj_1 = []

    for i, mdb in enumerate(mdb_list):
        if mdb.is_image_scan():
            if mdb.mdh[14][7] == 0:
                if mdb.mdh[14][0] in dico_data_set0.keys():
                    dico_data_set0[mdb.mdh[14][0]][mdb.mdh[14][3]] = mdb
                else:
                    dico_data_set0[mdb.mdh[14][0]] = {}
                    dico_data_set0[mdb.mdh[14][0]][mdb.mdh[14][3]] = mdb
            else:
                if mdb.mdh[14][0] in dico_data_set1.keys():
                    dico_data_set1[mdb.mdh[14][0]][mdb.mdh[14][3]] = mdb
                else:
                    dico_data_set1[mdb.mdh[14][0]] = {}
                    dico_data_set1[mdb.mdh[14][0]][mdb.mdh[14][3]] = mdb
        elif mdb.is_flag_set("NOISEADJSCAN"):
            if mdb.mdh[14][1] == 0:
                noise_adj_0.append(mdb.data)
            else:
                noise_adj_1.append(mdb.data)

    noise_adj_0 = np.array(noise_adj_0)
    noise_adj_1 = np.array(noise_adj_1)
    data_noise = np.stack([noise_adj_0, noise_adj_1])
    data_noise = np.moveaxis(data_noise, 2, 0)
    nb_chan = data_noise.shape[0]
    data_noise = data_noise.reshape(nb_chan, -1)

    kdata_3D_set0 = np.zeros(shape=(32, 32, nb_chan, 128), dtype="complex64")
    for j in range(kdata_3D_set0.shape[1] - 1):
        if (j + 1) in dico_data_set0.keys():
            for i in range(kdata_3D_set0.shape[0] - 1):
                if (i + 1) in dico_data_set0[j + 1].keys():
                    kdata_3D_set0[i + 1, j + 1, :] = dico_data_set0[j + 1][i + 1].data

    kdata_3D_set1 = np.zeros(shape=(32, 32, 2, 128), dtype="complex64")
    for j in range(kdata_3D_set1.shape[1] - 1):
        if (j + 1) in dico_data_set1.keys():

            for i in range(kdata_3D_set1.shape[0] - 1):
                if (i + 1) in dico_data_set1[j + 1].keys():
                    kdata_3D_set1[i + 1, j + 1, :] = dico_data_set1[j + 1][i + 1].data

    data_body = kdata_3D_set1.T
    data_multicoil = kdata_3D_set0.T

    return data_body, data_multicoil, data_noise


def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    axis_names=["x","y","z"]
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),colors)):
        axlabel = axis_names[i]
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,va="center", ha="center")
        ax.text(*offset, name, color="k", va="center", ha="center",bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})



def gamma_transform(volume,gamma):
    target=copy(volume)
    for sl in range(target.shape[0]):
        target[sl]=((np.abs(volume[sl])-np.min(np.abs(volume[sl])))/(np.max(np.abs(volume[sl]))-np.min(np.abs(volume[sl]))))**gamma
    return target
