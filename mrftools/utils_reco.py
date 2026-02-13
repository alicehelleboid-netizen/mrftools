
try:
    import matplotlib.pyplot as plt
except:
    pass

# import voxelmorph as vxm
import numpy as np
from scipy.signal import savgol_filter
# from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
from mrftools.trajectory import Navigator3D,Radial

from copy import copy,deepcopy

from datetime import datetime
import scipy as sp
from tqdm import tqdm
import pandas as pd
from mrftools.utils_mrf import groupby
from mrftools.Transformers import PCAComplex

import finufft
try:
    import pycuda
    import pycuda.autoinit
    from pycuda.gpuarray import GPUArray, to_gpu
    from cufinufft import cufinufft

except:
    pass

try:
    import cupy as cp
except:
    print("Could not import cupy")
import cv2

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import neurite as ne
from skimage.transform import resize
from numpy.lib import stride_tricks
import dask.array as da
import pywt
from scipy.interpolate import RegularGridInterpolator
# import ot


def calculate_sensitivity_map_3D_for_nav(kdata, trajectory, res=16, image_size=(400,)):
    traj = trajectory.get_traj()
    nb_channels = kdata.shape[0]
    npoint = kdata.shape[-1]
    nb_slices = kdata.shape[1]
    nb_gating_spokes = kdata.shape[2]
    center_res = int(npoint / 2 - 1)
    kdata = kdata.reshape((nb_channels, -1, npoint))
    b1_nav = np.zeros((nb_channels, kdata.shape[1],) + image_size, dtype=np.complex128)
    kdata_for_sensi = np.zeros(kdata.shape[1:], dtype=np.complex128)
    for i in tqdm(range(nb_channels)):
        kdata_for_sensi[:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[i, :,
                                                                                      (center_res - int(res / 2)):(
                                                                                                  center_res + int(
                                                                                              res / 2))]
        b1_nav[i] = finufft.nufft1d1(traj[0, :, 2], kdata_for_sensi, image_size)

        print("Normalizing sensi")

        # b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
        # del coil_sensitivity
        # b1 = b1 / np.max(np.abs(b1.flatten()))
    b1_nav /= np.linalg.norm(b1_nav, axis=0)
    b1_nav /= np.max(np.abs(b1_nav.flatten()))

    b1_nav = b1_nav.reshape((nb_channels, nb_slices, nb_gating_spokes, int(npoint / 2)))
    return b1_nav

def correct_mvt_kdata_zero_filled(trajectory,cond,ntimesteps):

    traj=trajectory.get_traj()

    mode=trajectory.paramDict["mode"]
    incoherent=trajectory.paramDict["incoherent"]

    nb_rep = int(cond.shape[0]/traj.shape[0])
    npoint = int(traj.shape[1] / nb_rep)
    nspoke = int(traj.shape[0] / ntimesteps)


    traj_for_selection = np.array(groupby(traj, npoint, axis=1))
    traj_for_selection = traj_for_selection.reshape(cond.shape[0], -1, 3)
    indices = np.unravel_index(np.argwhere(cond).T,(nb_rep,ntimesteps,nspoke))
    retained_indices = np.squeeze(np.array(indices).T)
    retained_timesteps = np.unique(retained_indices[:, 1])
    ## DENSITY CORRECTION

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

    df_retained["theta_weight"] = theta_inside_boundary*diff_theta.apply(lambda x: (np.sort(x[x>=0])[1]+np.sort(-x[x<=0])[1])/2 if ((x>=0).sum()>1) and ((x<=0).sum()>1) else 0)+ \
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
    theta_weight = df_retained["theta_weight"]
    kz_weight = df_retained["kz_weight"]
    weights = np.zeros(cond.shape[0])
    weights[cond] = theta_weight * kz_weight
    weights = weights.reshape(nb_rep, -1)
    weights = np.moveaxis(weights, 0, 1)
    weights = weights.reshape(ntimesteps, nspoke, -1)
    weights=weights[retained_timesteps]

    return weights,retained_timesteps


def calculate_displacement(image, bottom, top, shifts,lambda_tv=0.001,randomize=False,dct_frequency_filter=None,seasonal_adj=False,interp_bad_correl=False):
    np.save("./log/image_nav.npy",image)
    nb_gating_spokes = image.shape[1]
    nb_slices = image.shape[0]
    npoint_image = image.shape[-1]


    if seasonal_adj:
        from statsmodels.tsa.seasonal import seasonal_decompose

        image_reshaped = image.reshape(-1, npoint_image)
        decomposition = seasonal_decompose(image_reshaped,
                                           model='multiplicative', period=nb_gating_spokes)
        image=image_reshaped/decomposition.seasonal
        image=image.reshape(-1,nb_gating_spokes,npoint_image)
        print(image.shape)

    all_images = image
    if randomize:
        image_reshaped=image.reshape(-1, npoint_image)
        for ind in range(2, nb_gating_spokes):
            shifted_image = np.concatenate([image_reshaped[ind:], image_reshaped[:ind]],
                                           axis=0).reshape(nb_slices, nb_gating_spokes, -1)
            all_images = np.concatenate([all_images, shifted_image], axis=0)
    ft = np.mean(all_images, axis=0)
    # ft=np.mean(image_nav_best_channel,axis=0)
    # ft=image[0]
    image_nav_for_correl = image.reshape(-1, npoint_image)
    nb_images = image_nav_for_correl.shape[0]
    max_correl=0
    max_correls = []
    mvt = []
    # adj=[]
    all_correls=[]
    for j in tqdm(range(nb_images)):
        if (j % nb_gating_spokes == 0)or(max_correl<0.5):
            used_shifts=shifts
        else:
            used_shifts=np.arange(mvt[j - 1]-10,(mvt[j - 1]+10)).astype(int)
        corrs = np.zeros(len(used_shifts))
        bottom = np.maximum(-used_shifts[0],int(npoint_image/4))
        top = np.minimum(int(npoint_image) -used_shifts[-1],int(3*npoint_image/4))
        for i, shift in enumerate(used_shifts):
            
            corr = np.corrcoef(np.concatenate([ft[j % nb_gating_spokes, bottom:top].reshape(1, -1),
                                               image_nav_for_correl[j, (bottom + shift):(top + shift)].reshape(1, -1)],
                                              axis=0))[0, 1]
            # corr = np.linalg.norm(image_nav_for_correl[0, bottom:top]-image_nav_for_correl[j + 1, (bottom + shift):(top + shift)])
            corrs[i] = corr
        # adjustment=np.sum(dft_x[j+1]*dft_t[j])/np.sum(dft_x[j+1]**2)
        # adj.append(adjustment)
        if (j % nb_gating_spokes == 0):
            J = corrs


        else:
            J = corrs - lambda_tv * (np.array(used_shifts) - mvt[j - 1]) ** 2  # penalty to not be too far from last disp

        ind_max_J=np.argmax(J)
        current_mvt = used_shifts[ind_max_J]
        max_correl=J[ind_max_J]
        max_correls.append(max_correl)
        mvt.append(current_mvt)
        all_correls.append(corrs)
    correls_array = np.array(all_correls)
    np.save("log_all_correls_displacement.npy",correls_array)
    # mvt = [shifts[i] for i in np.argmax(correls_array, axis=-1)]
    # mvt=[shifts[i] for i in np.argmin(correls_array,axis=-1)]
    # mvt=np.array(mvt)+np.array(adj)
    # mvt=np.concatenate([[0],mvt]).astype(int)
    # mvt=np.array(mvt).reshape(int(nb_slices),int(nb_gating_spokes))
    # displacement=-np.cumsum(mvt,axis=-1).flatten()
    displacement = np.array(mvt)
    max_correls=np.array(max_correls)
    if dct_frequency_filter is not None:
        displacements_reshaped = displacement.reshape(nb_slices, nb_gating_spokes)
        displacements_smooth = np.zeros_like(displacements_reshaped)
        for sl in range(nb_slices):
            transf_disp = sp.fft.dct(displacements_reshaped[sl])
            transf_disp[-dct_frequency_filter:] = 0
            displacements_smooth[sl] = sp.fft.idct(transf_disp)
        displacement = displacements_smooth.flatten()

    if interp_bad_correl:
        ind_bad_correl=np.argwhere(max_correls<0.2)
        displacement_new=copy(displacement)
        for i in ind_bad_correl.flatten():
            if ((np.abs(displacement[i]-displacement[i-1])/np.abs(displacement[i])>0.5)and(np.abs(displacement[i]-displacement[i+1])/np.abs(displacement[i])>0.5)):
                displacement_new[i]=np.mean([displacement[i-1],displacement[i+1]])

        displacement=displacement_new


    return displacement


def calculate_displacement_ml(data_for_nav,nb_segments,window=5,device="cuda",ch=None):
    nb_allspokes = nb_segments
    nb_slices = data_for_nav.shape[1]
    nb_gating_spokes = data_for_nav.shape[2]
    npoint_nav = data_for_nav.shape[-1]

    all_timesteps = np.arange(nb_allspokes)
    nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                           applied_timesteps=list(nav_timesteps))

    nav_image_size = (int(npoint_nav / 2),)

    if ch is None:
        b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
        b1_nav_mean = np.mean(b1_nav, axis=(1, 2))
        image_full = np.abs(simulate_nav_images_multi(data_for_nav, nav_traj, nav_image_size, b1_nav_mean))

    else:
        image_full = np.abs(simulate_nav_images_multi(np.expand_dims(data_for_nav[ch], axis=0), nav_traj,
                                                                 nav_image_size, b1=None)).squeeze()


    image_full=image_full.reshape(-1,image_full.shape[-1]).T

    np.save("./log/image_nav_ml.npy",image_full)
    
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"


    print("Segment anything model : {}".format(sam_checkpoint) )

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Navigator images shape : {}".format(image_full.shape))

    nb_images = image_full.shape[1]
    disp = []
    for j in tqdm(range(int(nb_images / (window * nb_gating_spokes))+1)):
        start_cycle = j * window * nb_gating_spokes
        end_cycle = np.minimum(start_cycle + window * nb_gating_spokes, nb_images)
        image = image_full[:, start_cycle:end_cycle]
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        diffs = np.abs(np.diff(masks[0]["segmentation"] * 1, axis=0))
        diffs[:5] = 0
        diffs[-5:] = 0
        for l in range(diffs.shape[1]):
            if len(np.argwhere((diffs[:, l] > 0))) > 0:
                disp.append(np.min(np.argwhere((diffs[:, l] > 0))))
            else:
                disp.append(0)
    print("Displacement ML shape {}".format(np.array(disp).shape))

    return np.array(disp)


def simulate_nav_images_multi(kdata, trajectory, image_size=(400,), b1=None):
    traj = trajectory.get_traj()
    nb_channels = kdata.shape[0]
    npoint = kdata.shape[-1]
    nb_slices = kdata.shape[1]
    nb_gating_spokes = kdata.shape[2]
    npoint_image=image_size[0]

    print(kdata.dtype)
    print(traj.dtype)

    if kdata.dtype == "complex64":
        traj=traj.astype("float64")

    if kdata.dtype == "complex128":
        traj=traj.astype("float128")

    if b1 is not None:
        if b1.ndim == 2:
            b1 = np.expand_dims(b1, axis=(1, 2))
        elif b1.ndim == 3:
            b1 = np.expand_dims(b1, axis = (1))

    traj = traj.astype(np.float32)

    kdata = kdata.reshape((nb_channels, -1, npoint))
    images_series_rebuilt_nav = np.zeros((nb_slices, nb_gating_spokes, npoint_image), dtype=np.complex64)
    # all_channels_images_nav = np.zeros((nb_channels,nb_slices,nb_gating_spokes,int(npoint/2)),dtype=np.complex64)

    for i in tqdm(range(nb_channels)):
        fk = finufft.nufft1d1(traj[0, :, 2], kdata[i, :, :], image_size)
        fk = fk.reshape((nb_slices, nb_gating_spokes, npoint_image))

        # all_channels_images_nav[i]=fk

        if b1 is None:
            images_series_rebuilt_nav += np.abs(fk) ** 2
        else:
            images_series_rebuilt_nav += b1[i].conj() * fk

    if b1 is None:
        images_series_rebuilt_nav = np.sqrt(images_series_rebuilt_nav)

    return images_series_rebuilt_nav

def calculate_displacements_allchannels(data_for_nav,nb_segments,shifts = list(range(-5, 5)),lambda_tv=0.001,pad=10,randomize=False):
    print("Processing Nav Data...")
    nb_allspokes=nb_segments
    nb_slices=data_for_nav.shape[1]
    nb_channels=data_for_nav.shape[0]
    nb_gating_spokes=data_for_nav.shape[2]
    npoint_nav=data_for_nav.shape[-1]

    all_timesteps = np.arange(nb_allspokes)
    nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,applied_timesteps=list(nav_timesteps))

    nav_image_size = (int(npoint_nav / 2),)
    print("Estimating Movement...")
    bottom = -shifts[0]
    top = int(npoint_nav / 2+2*pad) -shifts[-1]
    displacements_all_channels = []

    for j in range(nb_channels):
        images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[j], axis=0), nav_traj,
                                                                     nav_image_size, b1=None)
        images_series_rebuilt_nav_ch = np.pad(images_series_rebuilt_nav_ch, pad_width=((0, 0), (0, 0), (pad, pad)), mode="edge")

        image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
        displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, lambda_tv=lambda_tv,randomize=randomize)
        displacements_all_channels.append(displacements)
    displacements_all_channels=np.array(displacements_all_channels)
    displacements_all_channels=displacements_all_channels.reshape(nb_channels,nb_slices,-1)
    max_slices = nb_slices
    As_hat_normalized = np.zeros(displacements_all_channels.shape)
    As_hat_filtered = np.zeros(displacements_all_channels.shape)

    for ch in tqdm(range(nb_channels)):
        for sl in range(max_slices):
            signal = displacements_all_channels[ch, sl, :]
            if np.max(signal) == np.min(signal):
                signal = 0.5 * np.ones_like(signal)
            else:
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            As_hat_normalized[ch, sl, :] = signal
            signal_filtered = savgol_filter(signal, 3, 2)
            signal_filtered = lowess(signal_filtered, np.arange(len(signal_filtered)), frac=0.1)[:, 1]
            As_hat_filtered[ch, sl, :] = signal_filtered

    data_for_pca = As_hat_filtered.reshape(nb_channels, -1)
    
    #pca = PCA(n_components=1)
    #pca.fit(data_for_pca.T)
    #pcs = pca.components_ @ data_for_pca
    displacements = np.mean(data_for_pca,axis=0)
    return displacements


def calculate_displacements_singlechannel(data_for_nav, nb_segments, shifts=list(range(-5, 5)), lambda_tv=0.001,ch=0,pad=10,randomize=False,dct_frequency_filter=None,seasonal_adj=False,interp_bad_correl=False):
    print("Processing Nav Data...")
    nb_allspokes = nb_segments
    nb_slices = data_for_nav.shape[1]
    nb_gating_spokes = data_for_nav.shape[2]
    npoint_nav = data_for_nav.shape[-1]

    all_timesteps = np.arange(nb_allspokes)
    nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                           applied_timesteps=list(nav_timesteps))

    nav_image_size = (int(npoint_nav),)
    print("Estimating Movement...")
    bottom = np.maximum(-shifts[0],int(nav_image_size[0]/4))
    top = np.minimum(int(npoint_nav + 2*pad) - shifts[-1],int(3*nav_image_size[0]/4))
    

    images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[ch], axis=0), nav_traj,
                                                                 nav_image_size, b1=None)
    images_series_rebuilt_nav_ch = np.pad(images_series_rebuilt_nav_ch, pad_width=((0, 0), (0, 0), (pad, pad)),
                                              mode="edge")

    image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
    displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, lambda_tv=lambda_tv,randomize=randomize,dct_frequency_filter=dct_frequency_filter,seasonal_adj=seasonal_adj,interp_bad_correl=interp_bad_correl)


    max_slices = nb_slices
    displacements=displacements.reshape(nb_slices,-1)
    As_hat_normalized = np.zeros(displacements.shape)
    As_hat_filtered = np.zeros(displacements.shape)

    for sl in range(max_slices):
        signal = displacements[sl, :]
        if np.max(signal) == np.min(signal):
            signal = 0.5 * np.ones_like(signal)
        else:
            min=np.min(signal)
            max=np.max(signal)

            signal = (signal - min) / (max - min)
        As_hat_normalized[sl, :] = signal
        signal_filtered = savgol_filter(signal, 3, 2)
        signal_filtered = lowess(signal_filtered, np.arange(len(signal_filtered)), frac=0.1)[:, 1]
        As_hat_filtered[sl, :] = min + (max-min)*signal_filtered

    displacements = As_hat_filtered.flatten()

    return displacements


def estimate_weights_bins(displacements,nb_slices,nb_segments,nb_gating_spokes,ntimesteps,radial_traj,nb_bins=5,retained_categories=None,bins=None,equal_spoke_per_bin=False,us=1,sim_us=1,us_file=None,polyfit=False,soft_weight_for_full_inspi=False,nb_rep_center_part=1,std_disp=None,disp_respi=None):

    if soft_weight_for_full_inspi:
        alpha=0.35
        tau=0.5

    displacement_for_binning = displacements
    max_bin = np.max(displacement_for_binning)
    min_bin = np.min(displacement_for_binning)
    bin_width=(max_bin-min_bin)/nb_bins

    print("Min displacement {}".format(min_bin))
    print("Max displacement {}".format(max_bin))
    displacement_for_binning=displacement_for_binning.flatten()
    if (np.prod(displacement_for_binning.shape)==nb_segments*nb_slices):
        print("Granular displacement")
        
        pt =True
    else:
        pt=False

    if bins is None:
        if not(equal_spoke_per_bin):
            #bins = np.arange(min_bin, max_bin+0.9*bin_width, bin_width)
            min_std = 1000
            for offset in np.arange(0., bin_width, 0.1):
                nb_bins = 5
                max_bin = np.max(displacement_for_binning)
                min_bin = np.min(displacement_for_binning) + offset

                bin_width = (max_bin - min_bin) / nb_bins
                bins = np.arange(min_bin, max_bin + 0.9 * bin_width, bin_width)
                categories = np.digitize(displacement_for_binning, bins)
                df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
                df_groups = df_cat.groupby("cat").count()
                min_curr = df_groups.displacement.iloc[:-1].std()
                if min_curr < min_std:
                    bins_final = bins
                    min_std = min_curr

            bins=bins_final
            if retained_categories is None:
                retained_categories = list(range(0, nb_bins + 1))
        else:
            disp_sorted_index = np.argsort(displacement_for_binning)

            if polyfit:
                disp_sorted=np.sort(displacement_for_binning)
                num_gs=np.arange(len(displacement_for_binning))
                p=np.polyfit(num_gs,disp_sorted,3)
                disp_sorted_fit=np.poly1d(p)(num_gs)
                inverse_sorted_index=[]
                for j in range(len(displacement_for_binning)):
                    inverse_sorted_index.append(np.argwhere(disp_sorted_index==j).squeeze())
                displacement_for_binning=disp_sorted_fit[inverse_sorted_index]

            

            count_disp = len(disp_sorted_index)
            disp_width = int(count_disp / nb_bins)

            displacement_for_bins=copy(displacement_for_binning)

            if std_disp is not None:
                displacement_for_bins=displacement_for_binning[displacement_for_binning>(np.mean(displacement_for_binning)-std_disp*np.std(displacement_for_binning))]
                disp_width = int(len(displacement_for_bins) / nb_bins)

            bins = []
            for j in range(1, nb_bins):
                bins.append(np.sort(displacement_for_bins)[j * disp_width])
            if retained_categories is None:
                retained_categories = list(range(0, nb_bins))
    else:
        nb_bins=len(bins)+1
        if retained_categories is None:
            retained_categories = list(range(0, nb_bins))

        if disp_respi is not None:
            print("Matching displacement distribution on provided displacement")
            ot_emd = ot.da.EMDTransport()
            # ot_emd=ot.da.SinkhornTransport(reg_e=1e-1)
            ot_emd.fit(Xs=displacement_for_binning.reshape(-1, 1), Xt=disp_respi.reshape(-1, 1))
            displacement_for_binning = ot_emd.transform(Xs=displacement_for_binning.reshape(-1, 1))[:,0]

    print(bins)

    categories = np.digitize(displacement_for_binning, bins,right=True)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()


    print(retained_categories)
    print(df_groups)

    groups=[]

    if std_disp is not None:
        for cat in retained_categories:
            groups.append((categories==cat)&(displacement_for_binning>(np.mean(displacement_for_binning)-std_disp*np.std(displacement_for_binning))))
    else:
        for cat in retained_categories:
            groups.append(categories==cat)

    nb_part=nb_slices
    dico_traj_retained = {}
    dico_retained_ts={}

    if nb_rep_center_part>1:
        print("Central partition was repeated : aggregating weights for central part")
        groups_new=[]
        for j, g in tqdm(enumerate(groups)):
            g=g.reshape((nb_slices,nb_gating_spokes))
            nb_slices = nb_slices-nb_rep_center_part+1
            nb_part = nb_part-nb_rep_center_part+1
            g_new=np.zeros((nb_slices,nb_segments),dtype=g.dtype)
            g_new[:int(nb_slices / 2)] = g[:int(nb_slices / 2)]
            g_new[(int(nb_slices / 2) + 1):] = g[(int(nb_slices / 2) + nb_rep_center_part):]
            g_new[(int(nb_slices / 2))]=g[int(nb_slices / 2):(int(nb_slices / 2) + nb_rep_center_part)].any(axis=0)
            g_new=g_new.flatten()
            groups_new.append(g_new)
            print(g_new[(int(nb_slices / 2))])
            print(g[int(nb_slices/2)])
            print("##################################")
        groups=groups_new
        

    
    

    if not(pt):
        spoke_groups = np.argmin(np.abs(
            np.arange(0, nb_segments * nb_part, 1).reshape(-1, 1) - np.arange(0, nb_segments * nb_part,
                                                                              nb_segments / nb_gating_spokes).reshape(1,
                                                                                                                      -1)),
            axis=-1)

        spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
        spoke_groups[:-1, -int(nb_segments / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_segments / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
        spoke_groups = spoke_groups.flatten()

    # if nb_rep_center_part>1:

    #     spoke_groups = spoke_groups.reshape(nb_slices, nb_segments)
    #     nb_slices=nb_slices-nb_rep_center_part+1
    #     spoke_groups_new=np.zeros((nb_slices,nb_segments),dtype=spoke_groups.dtype)
    #     spoke_groups_new[:int(nb_slices / 2)] = spoke_groups[:int(nb_slices / 2)]
    #     spoke_groups_new[(int(nb_slices / 2) + 1):] = spoke_groups[(int(nb_slices / 2) + nb_rep_center_part):]
        

    for j, g in tqdm(enumerate(groups)):
        print("######################  BUILDING FULL VOLUME AND MASK FOR GROUP {} ##########################".format(j))
        retained_nav_spokes_index = np.argwhere(g).flatten()
        #print(retained_nav_spokes_index)

        if not(pt):
            included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
            included_spokes[::int(nb_segments / nb_gating_spokes)] = False
        else:
            included_spokes = np.array([s in retained_nav_spokes_index for s in range(len(displacement_for_binning))])

        if (j==(len(retained_categories)-1))and(soft_weight_for_full_inspi):
            alpha=-2*np.log(tau)/(bins[-1]-bins[-2])
            print(alpha)
            print("Using Soft Weights for full inspiration phase")
            retained_nav_spokes_index_prev_bin = np.argwhere(groups[-2]).flatten()
            included_spokes_prev_bin = np.array([s in retained_nav_spokes_index_prev_bin for s in spoke_groups])
            included_spokes_prev_bin[::int(nb_segments / nb_gating_spokes)] = False
            disp_reshaped=np.array([displacement_for_binning[i] for i in spoke_groups])
            included_soft_weight = (np.exp(-alpha * np.abs(disp_reshaped - bins[-1])) > tau)
            print(np.sum(included_soft_weight))
            included_spokes_prev_bin=(included_soft_weight & included_spokes_prev_bin)
            print(np.sum(included_spokes_prev_bin))
            included_spokes_current_bin=included_spokes
            print(np.sum(included_spokes))
            included_spokes=included_spokes | included_spokes_prev_bin
            print(np.sum(included_spokes))

        if (j==0)and(soft_weight_for_full_inspi):
            alpha=-2*np.log(tau)/(bins[1]-bins[0])
            print(alpha)
            print("Using Soft Weights for full inspiration phase")
            retained_nav_spokes_index_prev_bin = np.argwhere(groups[1]).flatten()
            included_spokes_prev_bin = np.array([s in retained_nav_spokes_index_prev_bin for s in spoke_groups])
            included_spokes_prev_bin[::int(nb_segments / nb_gating_spokes)] = False
            disp_reshaped=np.array([displacement_for_binning[i] for i in spoke_groups])
            included_soft_weight = (np.exp(-alpha * np.abs(disp_reshaped - bins[0])) > tau)
            included_spokes_prev_bin=(included_soft_weight & included_spokes_prev_bin)
            included_spokes_current_bin=included_spokes
            included_spokes=included_spokes | included_spokes_prev_bin

        if us_file is not None:
            print("Using file {} for simulation kz undersampling".format(us_file))
            cond_us=np.load(us_file)
            cond_us = cond_us.flatten()
            included_spokes = cond_us * (1 * (included_spokes))
            included_spokes = (included_spokes > 0)

        elif sim_us>1:
            print("Simulating Undersampling of {} on acquired data".format(sim_us))
            cond_us = np.zeros((nb_slices, nb_segments))
            nspoke_per_part = 8
            cond_us = cond_us.reshape((nb_slices, -1,nspoke_per_part))

            curr_start = 0

            for sl in range(nb_slices):
                cond_us[sl, curr_start::sim_us, :] = 1
                curr_start = curr_start + 1
                curr_start = curr_start % sim_us

            cond_us = cond_us.flatten()
            included_spokes = cond_us*(1*(included_spokes))
            included_spokes=(included_spokes>0)
            np.save("test_us_included_spokes.npy",included_spokes)

        elif us>1:
            print("Adjusting Navigators with Undersampling of {} on acquired data".format(us))
            nb_slices_no_us=int(nb_slices*us)
            included_spokes_new=np.zeros((nb_slices_no_us,nb_segments),dtype=included_spokes.dtype)
            
            nspoke_per_part = 8
            included_spokes_new = included_spokes_new.reshape((nb_slices_no_us, -1,nspoke_per_part))
            included_spokes = included_spokes.reshape((nb_slices, -1,nspoke_per_part))

            curr_start = 0

            for sl in range(nb_slices_no_us):
                included_spokes_new[sl, curr_start::us, :] = included_spokes[int(sl/us),curr_start::us]
                curr_start = curr_start + 1
                curr_start = curr_start % us

            included_spokes=included_spokes_new.flatten()
            np.save("test_us_included_spokes.npy",included_spokes)

            
        np.save("test_us_included_spokes.npy",included_spokes)
        
        print(included_spokes.shape)

        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)

        print(weights.shape)

        if (j == (len(retained_categories) - 1)) and (soft_weight_for_full_inspi):
            disp_reshaped = disp_reshaped.reshape(nb_slices, nb_segments)
            disp_reshaped = np.moveaxis(disp_reshaped, -1, 0)
            included_spokes_current_bin=1*np.expand_dims(included_spokes_current_bin.reshape(nb_slices,nb_segments).T,axis=0)
            included_spokes_prev_bin = 1*np.expand_dims(included_spokes_prev_bin.reshape(nb_slices, nb_segments).T,axis=0)
            weights=weights*included_spokes_current_bin+weights*included_spokes_prev_bin*np.exp(-alpha * np.abs(disp_reshaped - bins[-1]))

        if (j == 0) and (soft_weight_for_full_inspi):
            disp_reshaped = disp_reshaped.reshape(nb_slices, nb_segments)
            disp_reshaped = np.moveaxis(disp_reshaped, -1, 0)
            included_spokes_current_bin=1*np.expand_dims(included_spokes_current_bin.reshape(nb_slices,nb_segments).T,axis=0)
            included_spokes_prev_bin = 1*np.expand_dims(included_spokes_prev_bin.reshape(nb_slices, nb_segments).T,axis=0)
            weights=weights*included_spokes_current_bin+weights*included_spokes_prev_bin*np.exp(-alpha * np.abs(disp_reshaped - bins[0]))


        dico_traj_retained[j] = weights
        dico_retained_ts[j]=retained_timesteps

    return dico_traj_retained,dico_retained_ts,bins


def calculate_sensitivity_map(kdata,trajectory,res=16,image_size=(256,256),hanning_filter=False,density_adj=False):
    traj_all = trajectory.get_traj().astype("float32")
    traj_all=traj_all.reshape(-1,traj_all.shape[-1])
    npoint = kdata.shape[-1]
    center_res = int(npoint / 2)

    nb_channels=kdata.shape[1]
    nb_slices=kdata.shape[0]

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        # density=np.expand_dims(axis=0)
        density = np.expand_dims(density, tuple(range(kdata.ndim - 1)))
        kdata*=density

    kdata_for_sensi = np.zeros_like(kdata)

    if hanning_filter:
        if kdata.ndim==3:#Tried :: syntax but somehow it introduces errors in the allocation
            kdata_for_sensi[:,:, (center_res - int(res / 2)):(center_res + int(res / 2))]=kdata[:,:,
                                                                                       (center_res - int(res / 2)):(
                                                                                                   center_res + int(
                                                                                               res / 2))]*np.expand_dims(np.hanning(2*int(res/2)),axis=(0,1))
        else:
            kdata_for_sensi[:, :,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:, :,:,
                                                                                             (center_res - int(res / 2)):(
                                                                                                     center_res + int(
                                                                                                 res / 2))]*np.expand_dims(np.hanning(2*int(res/2)),axis=(0,1,2))


    else:
        if kdata.ndim==3:#Tried :: syntax but somehow it introduces errors in the allocation
            kdata_for_sensi[:,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:,:,
                                                                                       (center_res - int(res / 2)):(
                                                                                                   center_res + int(
                                                                                               res / 2))]
        else:
            kdata_for_sensi[:, :,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:, :,:,
                                                                                             (center_res - int(res / 2)):(
                                                                                                     center_res + int(
                                                                                                 res / 2))]

    #kdata_all = kdata_for_sensi.reshape(np.prod(kdata.shape[:-2]), -1)
    #print(traj_all.shape)
    #print(kdata_all.shape)

    #coil_sensitivity = finufft.nufft2d1(traj_all[:, 0], traj_all[:, 1], kdata_all, image_size)
    
    coil_sensitivity=np.zeros((nb_slices,nb_channels,)+image_size,dtype=kdata.dtype)
    print(kdata_for_sensi.shape)

    for sl in tqdm(range(nb_slices)):
        coil_sensitivity[sl]=finufft.nufft2d1(traj_all[:, 0], traj_all[:, 1], kdata_for_sensi[sl].reshape(nb_channels,-1), image_size)
    
    #coil_sensitivity=coil_sensitivity.reshape(*kdata.shape[:-2],*image_size)
    print(coil_sensitivity.shape)

    if coil_sensitivity.ndim==3:
        print("Ndim 3)")
        b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
        #b1 = b1 / np.max(np.abs(b1.flatten()))
    else:#first dimension contains slices
        print("Ndim > 3")
        b1=coil_sensitivity.copy()
        for i in range(coil_sensitivity.shape[0]):
            b1[i]=coil_sensitivity[i] / np.linalg.norm(coil_sensitivity[i], axis=0)
            #b1[i]=b1[i] / np.max(np.abs(b1[i].flatten()))
    return b1

def calculate_sensitivity_map_3D(kdata,trajectory,res=16,image_size=(1,256,256),useGPU=False,eps=1e-6,light_memory_usage=False,density_adj=False,hanning_filter=False,res_kz=None):
    traj_all = trajectory.get_traj()
    print(traj_all.shape)
    traj_all = traj_all.reshape(-1, traj_all.shape[-1])
    npoint = kdata.shape[-1]
    nb_channels = kdata.shape[0]
    center_res = int(npoint / 2)
    print("Here 2")
    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata.ndim - 1)))
        kdata*=density

    print("Here 3")
    if not(light_memory_usage):
        kdata_for_sensi = np.zeros(kdata.shape, dtype=np.complex128)
        if not(hanning_filter):
            kdata_for_sensi[:, :, :, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:,
                                                                                            :, :,
                                                                                            (center_res - int(res / 2)):(
                                                                                                    center_res + int(
                                                                                                res / 2))]
        else:
            kdata_for_sensi[:, :, :, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:,
                                                                                                :, :,
                                                                                                (center_res - int(
                                                                                                    res / 2)):(
                                                                                                        center_res + int(
                                                                                                    res / 2))]*np.expand_dims(np.hanning(2 * int(res / 2)), axis=(0, 1, 2))



        del kdata
        kdata_for_sensi = kdata_for_sensi.reshape(nb_channels, -1)

    print("Performing NUFFT on filtered data for sensitivity calculation")
    if not(useGPU):
        if not(light_memory_usage):
            index_non_zero_kdata=np.nonzero(kdata_for_sensi[0])
            kdata_for_sensi=kdata_for_sensi[index_non_zero_kdata]
            traj_all=traj_all[index_non_zero_kdata]
            coil_sensitivity = finufft.nufft3d1(traj_all[:, 2], traj_all[:, 0], traj_all[:, 1], kdata_for_sensi, image_size)
        else:
            coil_sensitivity=np.zeros((nb_channels,)+image_size,dtype=np.complex128)
            kdata_for_sensi = np.zeros(kdata.shape[1:], dtype=np.complex128)
            for i in tqdm(range(nb_channels)):
                if not(hanning_filter):
                    kdata_for_sensi[:, :, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[i,
                                                                                                    :, :,
                                                                                                    (center_res - int(
                                                                                                        res / 2)):(
                                                                                                            center_res + int(
                                                                                                      res / 2))]

                else:
                    if res_kz is not None:
                        center_res_kz = int(kdata.shape[-2] / 2)
                        kdata_for_sensi[:, (center_res_kz - int(res_kz / 2)):(center_res_kz + int(res_kz / 2)+1), (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[i,
                                                                                                         :, (center_res_kz - int(res_kz / 2)):(center_res_kz + int(res_kz / 2)+1),
                                                                                                         (
                                                                                                                     center_res - int(
                                                                                                                 res / 2)):(
                                                                                                                 center_res + int(
                                                                                                             res / 2))] * np.expand_dims(
                            np.hanning(2 * int(res / 2)), axis=(0, 1))* np.expand_dims(
                            np.hanning(2 * int(res_kz / 2)+1), axis=(0,2))


                    else:
                        kdata_for_sensi[:, :, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[i,
                                                                                                         :, :,
                                                                                                         (center_res - int(
                                                                                                             res / 2)):(
                                                                                                                 center_res + int(
                                                                                                             res / 2))]*np.expand_dims(np.hanning(2 * int(res / 2)), axis=(0, 1))


                flattened_kdata_for_sensi=kdata_for_sensi.flatten()
                index_non_zero_kdata = np.nonzero(flattened_kdata_for_sensi)
                flattened_kdata_for_sensi=flattened_kdata_for_sensi[index_non_zero_kdata]
                traj_current = traj_all[index_non_zero_kdata]
                coil_sensitivity[i] = finufft.nufft3d1(traj_current[:, 2], traj_current[:, 0], traj_current[:, 1], flattened_kdata_for_sensi, image_size)


    else:
        N1, N2, N3 = image_size[0], image_size[1], image_size[2]
        dtype = np.float32  # Datatype (real)
        complex_dtype = np.complex64

        fk_gpu = GPUArray((nb_channels, N1, N2, N3), dtype=complex_dtype)

        kx = traj_all[:, 0]
        ky = traj_all[:, 1]
        kz = traj_all[:, 2]

        kx = kx.astype(dtype)
        ky = ky.astype(dtype)
        kz = kz.astype(dtype)

        c_retrieved_gpu = to_gpu(kdata_for_sensi.astype(complex_dtype))
        plan = cufinufft(1, (N1, N2, N3), nb_channels, eps=eps, dtype=dtype)
        plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

        plan.execute(c_retrieved_gpu, fk_gpu)

        coil_sensitivity = np.squeeze(fk_gpu.get())

        fk_gpu.gpudata.free()
        c_retrieved_gpu.gpudata.free()
        plan.__del__()

    del kdata_for_sensi
    print("Normalizing sensi")

    #b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
    #del coil_sensitivity
    #b1 = b1 / np.max(np.abs(b1.flatten()))
    coil_sensitivity /= np.linalg.norm(coil_sensitivity, axis=0)
    #coil_sensitivity /= np.max(np.abs(coil_sensitivity.flatten()))


    return coil_sensitivity

def coil_compression_2Dplus1(kdata_all_channels_all_slices,ntimesteps=175,n_comp=16,invert_dens_adj=False,res=16,cc_res=None,cc_res_kz=None):
    '''
    Input:
    kdata_all_channels_all_slices : raw kdata
    ntimesteps : number of MRF images (not really useful)
    n_comp : retained virtual coils

    Output:
    pca_dict: dictionary which contains the coils PCA components for each slice
    b1_all_slices_2Dplus1_pca: virtual coil compressions (size n_comp x nb_slices x image_size) 
    '''

    

    data_shape=kdata_all_channels_all_slices.shape
    nb_channels=data_shape[0]
    nb_allspokes = data_shape[-3]
    npoint = data_shape[-1]
    nb_slices = data_shape[-2]
    image_size=(nb_slices,int(npoint/2),int(npoint/2))

    radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)

    if invert_dens_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        # density=np.expand_dims(axis=0)
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))
        kdata_all_channels_all_slices/=density

    if cc_res is not None:
        center_res = int(npoint / 2)
        kdata_for_sensi = np.zeros(kdata_all_channels_all_slices.shape, dtype=kdata_all_channels_all_slices.dtype)
        if cc_res_kz is not None:
            center_res_kz = int(nb_slices / 2)
            print(kdata_for_sensi.shape)
            print(np.expand_dims(np.hanning(2*int(cc_res/2)),axis=(0,1,2)).shape)
            print(np.expand_dims(np.hanning(2*int(cc_res_kz/2)+1),axis=(0,1,3)).shape)
            print(kdata_all_channels_all_slices[:, :,(center_res_kz - int(cc_res_kz / 2)):(center_res_kz + int(cc_res_kz / 2)+1),
                                                                                                (center_res - int(cc_res / 2)):(
                                                                                                        center_res + int(
                                                                                                    cc_res / 2))].shape)


            kdata_for_sensi[:, :,(center_res_kz - int(cc_res_kz / 2)):(center_res_kz + int(cc_res_kz / 2)+1), (center_res - int(cc_res / 2)):(center_res + int(cc_res / 2))] = kdata_all_channels_all_slices[:, :,(center_res_kz - int(cc_res_kz / 2)):(center_res_kz + int(cc_res_kz / 2)+1),
                                                                                                (center_res - int(cc_res / 2)):(
                                                                                                        center_res + int(
                                                                                                    cc_res / 2))]*np.expand_dims(np.hanning(2*int(cc_res/2)),axis=(0,1,2))*np.expand_dims(np.hanning(2*int(cc_res_kz/2)+1),axis=(0,1,3))
        else:
            kdata_for_sensi[:, :, :,
            (center_res - int(cc_res / 2)):(center_res + int(cc_res / 2))] = kdata_all_channels_all_slices[:, :,
                                                                             :,
                                                                             (center_res - int(cc_res / 2)):(
                                                                                     center_res + int(
                                                                                 cc_res / 2))] * np.expand_dims(
                np.hanning(2 * int(cc_res / 2)), axis=(0, 1, 2))


        kdata_all_channels_all_slices=kdata_for_sensi



    data_numpy_zkxky=np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices,axes=2),axis=2,workers=24),axes=2).astype("complex64")
    #data_numpy_zkxky=data_numpy_zkxky.reshape(nb_channels,ntimesteps,-1,nb_slices,npoint)
    data_numpy_zkxky = data_numpy_zkxky.reshape(nb_channels, -1, nb_slices, npoint)
    data_numpy_zkxky=np.moveaxis(data_numpy_zkxky,-2,1)

    #window=data_numpy_zkxky.shape[3]


    #data_numpy_zkxky_for_pca_all=np.zeros((n_comp,nb_slices,ntimesteps,window,npoint),dtype=data_numpy_zkxky.dtype)
    data_numpy_zkxky_for_pca_all = np.zeros((n_comp, nb_slices, nb_allspokes, npoint),
                                            dtype=data_numpy_zkxky.dtype)
    pca_dict={}
    for sl in tqdm(range(nb_slices)):
        data_numpy_zkxky_slice=data_numpy_zkxky[:,sl]
        #data_numpy_zkxky_slice=data_numpy_zkxky_slice.reshape(nb_channels,ntimesteps,-1)

        data_numpy_zkxky_for_pca=data_numpy_zkxky_slice.reshape(nb_channels,-1)

        pca=PCAComplex(n_components_=n_comp)

        pca.fit(data_numpy_zkxky_for_pca.T)

        pca_dict[sl]=deepcopy(pca)

        data_numpy_zkxky_for_pca_transformed=pca.transform(data_numpy_zkxky_for_pca.T)
        data_numpy_zkxky_for_pca_transformed=data_numpy_zkxky_for_pca_transformed.T

        data_numpy_zkxky_for_pca_all[:,sl,:,:]=data_numpy_zkxky_for_pca_transformed.reshape(n_comp,-1,npoint)

    if invert_dens_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        # density=np.expand_dims(axis=0)
        density = np.expand_dims(density, tuple(range(data_numpy_zkxky_for_pca_all.ndim - 1)))
        data_numpy_zkxky_for_pca_all*=density

    b1_all_slices_2Dplus1_pca=calculate_sensitivity_map(np.moveaxis(data_numpy_zkxky_for_pca_all,1,0).reshape(nb_slices,n_comp,nb_allspokes,-1),radial_traj_2D,image_size=image_size[1:],hanning_filter=True,res=res)
    b1_all_slices_2Dplus1_pca=np.moveaxis(b1_all_slices_2Dplus1_pca,1,0)

    return pca_dict,b1_all_slices_2Dplus1_pca


def build_volume_2Dplus1_cc(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,weights,selected_spokes=None,nb_rep_center_part=1):
    #weights=1*(weights>0)
    #weights=np.ones_like(weights)

    # data=data.reshape(nb_channels,ntimesteps,8,-1,npoint)
    # data*=weights
    
    data_shape=kdata_all_channels_all_slices.shape
    nb_channels=data_shape[0]
    nb_allspokes = data_shape[-3]
    npoint = data_shape[-1]
    nb_slices = data_shape[-2]

    nb_slices=nb_slices-nb_rep_center_part+1

    n_comp=pca_dict[0].n_components_

    image_size =  (nb_slices,int(npoint/2),int(npoint/2))

    ntimesteps_reco=1

    radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)
    if selected_spokes is not None:
        kdata_all_channels_all_slices=kdata_all_channels_all_slices[:,selected_spokes]
        weights=weights[:,selected_spokes]
        nb_allspokes=kdata_all_channels_all_slices.shape[1]
        print("Selected {} spokes".format(nb_allspokes))
        radial_traj_2D.traj=radial_traj_2D.get_traj()[selected_spokes]
    traj_reco = radial_traj_2D.get_traj_for_reconstruction(ntimesteps_reco).astype("float32")
    traj_reco = traj_reco.reshape(-1, 2)

    print(traj_reco.shape)
    print(kdata_all_channels_all_slices.shape)
    print(weights.shape)
    
    # for ch in tqdm(range(nb_channels)):

    
    if nb_rep_center_part>1:
        data = np.fft.fftshift(
            sp.fft.ifft(np.fft.ifftshift(kdata_aggregate_center_part(kdata_all_channels_all_slices * weights,nb_rep_center_part), axes=2), axis=2, workers=24), axes=2)

    else:
        data = np.fft.fftshift(
            sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices * weights, axes=2), axis=2, workers=24), axes=2)

    print(weights.shape)
    data = data.reshape((nb_channels, nb_allspokes, nb_slices, -1))
    data = np.moveaxis(data, -2, 1)

    # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
    images_series_rebuilt = np.zeros(image_size, dtype=np.complex64)

    # data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))

    coil_compression=True
    if pca_dict[0].components_.shape[0]==pca_dict[0].components_.shape[1]:
        coil_compression=False

    for sl in tqdm(range(nb_slices)):
        data_curr = data[:, sl]
        data_curr = data_curr.reshape(nb_channels, -1)
        
        
        if coil_compression:
            pca = pca_dict[sl]
            print("PCA")
            data_curr_transformed = pca.transform(data_curr.T)
            data_curr_transformed = data_curr_transformed.T
            
            # data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
            data_curr_transformed = data_curr_transformed.reshape(n_comp, -1).astype('complex64')
        else:
            data_curr_transformed=data_curr.astype("complex64")


        for j in tqdm(range(n_comp)):
            kdata_singular = data_curr_transformed[j].flatten()
            print(traj_reco.shape)
            fk = finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size[1:])
            images_series_rebuilt[sl] += b1_all_slices_2Dplus1_pca[j, sl].conj() * fk

    return images_series_rebuilt


def build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights,selected_spokes,nb_rep_center_part=1):
    nb_gr=all_weights.shape[0]
    all_images=[]
    for gr in range(nb_gr):
        curr_image=build_volume_2Dplus1_cc(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights[gr],selected_spokes,nb_rep_center_part)
        all_images.append(curr_image)
    return np.array(all_images)




def build_volume_singular_2Dplus1_cc_registered(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict, weights,phi,L0,deformation_map,gr,useGPU=True,nb_rep_center_part=1,interp=cv2.INTER_LINEAR,select_first_rep=False):

    if useGPU:
        xp=cp
    else:
        xp=np

    data_shape = kdata_all_channels_all_slices.shape
    nb_channels = data_shape[0]
    nb_allspokes = data_shape[-3]
    npoint = data_shape[-1]
    nb_slices = data_shape[-2]

    nb_slices = nb_slices - nb_rep_center_part + 1
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))
    output_shape=(L0,)+image_size



    n_comp = pca_dict[0].n_components_
    ntimesteps=phi.shape[1]

    print("Image size {}".format(image_size))

    radial_traj_2D = Radial(total_nspokes=nb_allspokes, npoint=npoint)
    traj_reco = radial_traj_2D.get_traj_for_reconstruction(ntimesteps).astype("float32")
    print(traj_reco.shape)
    traj_reco = traj_reco.reshape(-1, 2)

    window=int(nb_allspokes/ntimesteps)

    #data = np.zeros((nb_channels, ntimesteps, 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)

    # for ch in tqdm(range(nb_channels)):

    if nb_rep_center_part>1:
        data = np.fft.fftshift(
            sp.fft.ifft(np.fft.ifftshift(kdata_aggregate_center_part(kdata_all_channels_all_slices * weights,nb_rep_center_part,select_first_rep), axes=2), axis=2, workers=24), axes=2)

    else:
        data = np.fft.fftshift(
        sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices * weights, axes=2), axis=2, workers=24), axes=2)

    data = data.reshape((nb_channels, ntimesteps, 8, nb_slices, -1))
    data = np.moveaxis(data, -2, 1)
    print("Kdata 2D+1 shape {}".format(data.shape))

    # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    # data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))

    for sl in tqdm(range(nb_slices)):
        data_curr = xp.asarray(data[:, sl])
        data_curr = data_curr.reshape(nb_channels, -1)
        pca = pca_dict[sl]

        print("PCA")
        data_curr_transformed = pca.transform(data_curr.T)
        data_curr_transformed = data_curr_transformed.T

        # data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
        data_curr_transformed = data_curr_transformed.reshape(n_comp, ntimesteps, -1)

        for j in tqdm(range(n_comp)):
            kdata_singular = xp.zeros((ntimesteps, npoint * window) + (L0,), dtype="complex64")
            for ts in tqdm(range(ntimesteps)):
                kdata_singular[ts, :, :] = xp.matmul(data_curr_transformed[j, ts, :, None],
                                                     (xp.asarray(phi[:L0]).conj().T[ts][None, :]))
            kdata_singular = xp.moveaxis(kdata_singular, -1, 0)

            kdata_singular = kdata_singular.reshape(L0, -1)
            if useGPU:
                kdata_singular=kdata_singular.get()


            fk = finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], kdata_singular, image_size[1:])
            images_series_rebuilt[:, sl] += np.expand_dims(b1_all_slices_2Dplus1_pca[j, sl].conj(), axis=0) * fk

    #np.save("./log/moving_gr{}.npy".format(gr),images_series_rebuilt)
    for l in range(L0):
        
        images_series_rebuilt[l]=apply_deformation_to_complex_volume(images_series_rebuilt[l], deformation_map,interp=interp)
    
    #np.save("./log/registered_gr{}.npy".format(gr),images_series_rebuilt)
    return images_series_rebuilt


def build_volume_singular_2Dplus1_cc_allbins_registered(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights,phi,L0,deformation_map,useGPU=True,nb_rep_center_part=1,interp=cv2.INTER_LINEAR,select_first_rep=False):
    nb_gr=all_weights.shape[0]
    all_images=[]
    for gr in range(nb_gr):
        curr_image=build_volume_singular_2Dplus1_cc_registered(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights[gr],phi,L0,deformation_map[:,gr],gr,useGPU,nb_rep_center_part,interp=interp,select_first_rep=select_first_rep)
        all_images.append(curr_image)
    return np.sum(np.array(all_images),axis=0)


def build_volume_singular_2Dplus1_cc(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict, weights,phi,L0,useGPU=True,nb_rep_center_part=1,select_first_rep=False,selected_spokes=None):

    if useGPU:
        xp=cp
    else:
        xp=np

    data_shape = kdata_all_channels_all_slices.shape
    nb_channels = data_shape[0]
    nb_allspokes = data_shape[-3]
    npoint = data_shape[-1]
    nb_slices = data_shape[-2]
    nb_slices = nb_slices - nb_rep_center_part + 1
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))
    output_shape=(L0,)+image_size

    n_comp = pca_dict[0].n_components_
    ntimesteps=phi.shape[1]


    radial_traj_2D = Radial(total_nspokes=nb_allspokes, npoint=npoint)

    if selected_spokes is not None:
        kdata_all_channels_all_slices=kdata_all_channels_all_slices[:,selected_spokes]
        if not((type(weights)==int)or(weights is None)):
            weights=weights[:,selected_spokes]
        nb_allspokes=kdata_all_channels_all_slices.shape[1]
        print("Selected {} spokes".format(nb_allspokes))
        radial_traj_2D.traj=radial_traj_2D.get_traj()[selected_spokes]

    traj_reco = radial_traj_2D.get_traj_for_reconstruction(ntimesteps).astype("float32")
    traj_reco = traj_reco.reshape(-1, 2)

    window=int(nb_allspokes/ntimesteps)

    #data = np.zeros((nb_channels, ntimesteps, 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)

    # for ch in tqdm(range(nb_channels)):

    if nb_rep_center_part>1:
        data = np.fft.fftshift(
            sp.fft.ifft(np.fft.ifftshift(kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part,select_first_rep)*weights, axes=2), axis=2, workers=24), axes=2)

    else:
        data = np.fft.fftshift(
        sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices * weights, axes=2), axis=2, workers=24), axes=2)

    data = data.reshape((nb_channels, ntimesteps, window, nb_slices, -1))
    data = np.moveaxis(data, -2, 1)

    # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    # data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))

    for sl in tqdm(range(nb_slices)):
        data_curr = xp.asarray(data[:, sl])
        data_curr = data_curr.reshape(nb_channels, -1)
        pca = pca_dict[sl]

        print("PCA")
        data_curr_transformed = pca.transform(data_curr.T)
        data_curr_transformed = data_curr_transformed.T

        # data_pca[:, sl, :, :, :] = data_curr_transformed.reshape(n_comp, ntimesteps,-1, npoint)
        data_curr_transformed = data_curr_transformed.reshape(n_comp, ntimesteps, -1)

        for j in tqdm(range(n_comp)):
            kdata_singular = xp.zeros((ntimesteps, npoint * window) + (L0,), dtype="complex64")
            for ts in tqdm(range(ntimesteps)):
                kdata_singular[ts, :, :] = xp.matmul(data_curr_transformed[j, ts, :, None],
                                                     (xp.asarray(phi[:L0]).conj().T[ts][None, :]))
            kdata_singular = xp.moveaxis(kdata_singular, -1, 0)
            kdata_singular = kdata_singular.reshape(L0, -1)
            if useGPU:
                kdata_singular=kdata_singular.get()
            fk = finufft.nufft2d1(traj_reco[:, 0], traj_reco[:, 1], kdata_singular.squeeze(), image_size[1:])
            if fk.ndim==2:
                images_series_rebuilt[:, sl] += b1_all_slices_2Dplus1_pca[j,sl].conj()* fk
            else:
                images_series_rebuilt[:, sl] += np.expand_dims(b1_all_slices_2Dplus1_pca[j, sl].conj(), axis=0) * fk

    return images_series_rebuilt


def build_volume_singular_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights,phi,L0,useGPU=True,nb_rep_center_part=1,select_first_rep=False):
    nb_gr=all_weights.shape[0]
    all_images=[]
    for gr in range(nb_gr):
        curr_image=build_volume_singular_2Dplus1_cc(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights[gr],phi,L0,useGPU,nb_rep_center_part,select_first_rep=select_first_rep)
        all_images.append(curr_image)
    return np.array(all_images)



def build_volume_singular_3D(kdata_all_channels_all_slices, b1_all_slices,radial_traj,weights,phi,L0,useGPU=True,nb_rep_center_part=1,select_first_rep=False):

    data_shape = kdata_all_channels_all_slices.shape
    nb_channels = data_shape[0]
    nb_allspokes = data_shape[-3]
    npoint = data_shape[-1]
    nb_slices = radial_traj.paramDict["nb_slices"]
    nb_rep=data_shape[-2]
    #nb_slices = nb_slices - nb_rep_center_part + 1
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))
    output_shape=(L0,)+image_size

    print("Output shape {}".format(output_shape))

    ntimesteps=phi.shape[1]

    print(data_shape)

    traj_reco = radial_traj.get_traj_for_reconstruction(1).astype("float32")
    traj_reco = traj_reco.reshape(-1, 3)

    num_k_samples=traj_reco.shape[0]

    print(traj_reco.shape)
    window=int(nb_allspokes/ntimesteps)

    #data = np.zeros((nb_channels, ntimesteps, 8, nb_slices, npoint), dtype=kdata_all_channels_all_slices.dtype)

    # for ch in tqdm(range(nb_channels)):

    # if nb_rep_center_part>1:
    #     data = np.fft.fftshift(
    #         sp.fft.ifft(np.fft.ifftshift(kdata_aggregate_center_part(kdata_all_channels_all_slices * weights,nb_rep_center_part,select_first_rep), axes=2), axis=2, workers=24), axes=2)

    # else:
    #     data = np.fft.fftshift(
    #     sp.fft.ifft(np.fft.ifftshift(kdata_all_channels_all_slices * weights, axes=2), axis=2, workers=24), axes=2)

    # data_pca = np.zeros((n_comp, nb_slices, ntimesteps, 8, npoint), dtype=data.dtype)
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)

    # data_curr_transformed_all=cp.zeros((nb_slices,n_comp,ntimesteps,window*npoint))

    print(np.max(weights))

    if not((type(weights)==int)or(weights is None)):
        weights=weights.reshape(1,ntimesteps,window,nb_rep,1)
    data=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,window,-1,npoint)*weights
    data=data.reshape(nb_channels,ntimesteps,-1)

    print(data.shape)

    for j in tqdm(range(nb_channels)):
        kdata_singular = np.zeros((ntimesteps, nb_rep*npoint * window) + (L0,), dtype="complex64")
        print(kdata_singular.shape)
        #print(phi.shape)
        for ts in tqdm(range(ntimesteps)):
            kdata_singular[ ts, :, :] = data[j, ts, :, None] @ (phi[:L0].conj().T[ts][None, :])
        kdata_singular = np.moveaxis(kdata_singular, -1, 0)
        kdata_singular = kdata_singular.reshape(L0, -1)

        print(kdata_singular.shape)
        print(traj_reco.shape)

        fk = finufft.nufft3d1(traj_reco[:, 2], traj_reco[:, 0], traj_reco[:, 1], kdata_singular.squeeze(), image_size)
        if fk.ndim==3:
            images_series_rebuilt += b1_all_slices[j].conj()* fk
        else:
            images_series_rebuilt += np.expand_dims(b1_all_slices[j].conj(), axis=0) * fk

    return images_series_rebuilt/num_k_samples


def build_volume_singular_3D_allbins(kdata_all_channels_all_slices,b1_all_slices,radial_traj,all_weights,phi,L0,useGPU=True,nb_rep_center_part=1,select_first_rep=False):
    nb_gr=all_weights.shape[0]
    all_images=[]
    for gr in range(nb_gr):
        print("###### Build singular volumes for bin {}####################".format(gr))
        curr_image=build_volume_singular_3D(kdata_all_channels_all_slices,b1_all_slices,radial_traj,all_weights[gr],phi,L0,useGPU,nb_rep_center_part,select_first_rep=select_first_rep)
        all_images.append(curr_image)
    all_images=np.array(all_images)
    print(all_images.shape)
    return all_images


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
        curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
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

        images_series_rebuilt+=np.expand_dims(b1_all_slices[k].conj(), axis=0)* (finufft.nufft2d1(traj[:, 0],traj[:, 1],curr_kdata.reshape(L0*nb_slices,-1),size[1:])).reshape((L0,)+size)


    images_series_rebuilt /= num_k_samples
    return images_series_rebuilt

def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:, :, 1], I[:, :, 0] = np.indices((h, w))  # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P


def apply_deformation_to_volume(volume, deformation_map,interp=cv2.INTER_LINEAR):
    '''
    input :
    volume nz x nx x ny type float
    deformation_map 2 x nz x nx x ny
    output :
    deformed volume nz x nx x ny type float
    '''
    deformed_volume = np.zeros_like(volume)
    if volume.ndim==4:
        L0=volume.shape[0]
        nb_slices = volume.shape[1]
        for sl in range(nb_slices):
            
            mapx = deformation_map[0, sl]
            mapy = deformation_map[1, sl]
            for l in range(L0):
                img = volume[l,sl].astype("float32")
                deformed_volume[l,sl] = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), interp)
    else:
        
        nb_slices = volume.shape[0]
        for sl in range(nb_slices):
            img = volume[sl].astype("float32")
            mapx = deformation_map[0, sl]
            mapy = deformation_map[1, sl]
            deformed_volume[sl] = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), interp)

    return deformed_volume

def interp_deformation(deformation_map,start=0,us=2):
    nb_slices_us = deformation_map.shape[2]
    npoint = deformation_map.shape[3]
    nb_slices = us * nb_slices_us
    nb_slices_interp = nb_slices - nb_slices_us
    nb_gr = deformation_map.shape[1]

    def_map_interp_gr_full = np.zeros((2,nb_gr,nb_slices,npoint,npoint),dtype=deformation_map.dtype)
    def_map_interp_gr_full[:, :, ::us] = deformation_map

    mask = np.zeros(nb_slices, dtype=bool)
    exclude_index = np.array(range(start, nb_slices, us))
    mask[exclude_index] = True

    x = np.arange(npoint)
    y = np.arange(npoint)
    z = np.arange(nb_slices)[::us]
    zg, xg, yg = np.meshgrid(z, x, y, indexing='ij')
    zi, xi, yi = np.meshgrid(np.arange(nb_slices)[~mask], x, y, indexing='ij')
    i_points = np.stack([zi, xi, yi], axis=-1)
    i_points = i_points.reshape(-1, 3)

    for gr in tqdm(range(nb_gr)):
        for d in range(2):
            data = deformation_map[d, gr]
            interp = RegularGridInterpolator((z, x, y), data, bounds_error=False, fill_value=None)
            def_map_interp_gr = interp(i_points)
            def_map_interp_gr = def_map_interp_gr.reshape(nb_slices_interp, npoint, npoint)
            def_map_interp_gr_full[d, gr, ~mask] = def_map_interp_gr

    return def_map_interp_gr_full

def interp_b1(b1,start=0,us=2):
    nb_slices_us = b1.shape[1]
    npoint = b1.shape[2]
    nb_slices = us * nb_slices_us
    nb_slices_interp = nb_slices - nb_slices_us
    nb_channels = b1.shape[0]

    b1_map_interp_full = np.zeros((nb_channels,nb_slices,npoint,npoint),dtype=b1.dtype)
    b1_map_interp_full[:, ::us] = b1

    mask = np.zeros(nb_slices, dtype=bool)
    exclude_index = np.array(range(start, nb_slices, us))
    mask[exclude_index] = True

    x = np.arange(npoint)
    y = np.arange(npoint)
    z = np.arange(nb_slices)[::us]
    zg, xg, yg = np.meshgrid(z, x, y, indexing='ij')
    zi, xi, yi = np.meshgrid(np.arange(nb_slices)[~mask], x, y, indexing='ij')
    i_points = np.stack([zi, xi, yi], axis=-1)
    i_points = i_points.reshape(-1, 3)

    for ch in range(nb_channels):
        data = b1[ch]
        interp = RegularGridInterpolator((z, x, y), data, bounds_error=False, fill_value=None)
        b1_interp = interp(i_points)
        b1_interp = b1_interp.reshape(nb_slices_interp, npoint, npoint)
        b1_map_interp_full[ch, ~mask] = b1_interp

    return b1_map_interp_full

def interp_b1_resize(b1,new_shape):
    nb_channels = b1.shape[0]

    b1_map_interp_full = np.zeros((nb_channels,)+new_shape,dtype=b1.dtype)

    for ch in tqdm(range(nb_channels)):
        b1_map_interp_full[ch]=resize(b1[ch].real,new_shape)+1j*resize(b1[ch].imag,new_shape)

    return b1_map_interp_full

def interp_deformation_resize(deformation_map,new_shape):
    _,nb_gr,nb_slices,npoint,_=deformation_map.shape
    X,Y=np.meshgrid(np.arange(npoint),np.arange(npoint))
    deformation_flow=np.stack([deformation_map[0]-np.expand_dims(X,axis=(0,1)),deformation_map[1]-np.expand_dims(Y,axis=(0,1))],axis=0)
    new_npoint=new_shape[2]

    deformation_flow_rescaled=deformation_flow*new_npoint/npoint

    deformation_flow_resized=np.zeros((2,nb_gr)+new_shape,dtype=deformation_flow.dtype)
    for gr in tqdm(range(nb_gr)):
        for l in range(2):
            deformation_flow_resized[l,gr]=resize(deformation_flow_rescaled[l,gr],new_shape)
           
    X_new,Y_new=np.meshgrid(np.arange(new_npoint),np.arange(new_npoint))

    deformation_map_resized=np.stack([deformation_flow_resized[0]+np.expand_dims(X_new,axis=(0,1)),deformation_flow_resized[1]+np.expand_dims(Y_new,axis=(0,1))],axis=0)
    deformation_map_resized=deformation_map_resized.astype(deformation_map.dtype)
    return deformation_map_resized


def apply_deformation_to_complex_volume(volume, deformation_map,interp=cv2.INTER_LINEAR):
    '''
    input :
    volume nz x nx x ny type complex
    deformation_map 2 x nz x nx x ny
    output :
    deformed volume nz x nx x ny type complex
    '''
    deformed_volume = apply_deformation_to_volume(np.real(volume), deformation_map,interp) + 1j * apply_deformation_to_volume(
        np.imag(volume), deformation_map,interp)
    return deformed_volume


def calculate_inverse_deformation_map(deformation_map):
    '''
    input :
    deformation_map 2 x nz x nx x ny
    output :
    inv_deformation_map 2 x nz x nx x ny
    '''
    inv_deformation_map = np.zeros_like(deformation_map)
    nb_slices = deformation_map.shape[1]
    for sl in range(nb_slices):
        inv_deformation_map[:, sl] = np.moveaxis(invert_map(np.moveaxis(deformation_map[:, sl], 0, -1)), -1, 0)

    return inv_deformation_map

def change_deformation_map_ref(deformation_map_allbins,index_ref):
    '''
    input :
    deformation_map_allbins nb_gr x 2 x nz x nx x ny

    output :
    deformation_map_allbins_new  nb_gr x 2 x nz x nx x ny -> deformation map to transform all bins to the bin index_ref

    '''

    ndim,nb_gr,nb_slices,nx,ny=deformation_map_allbins.shape
    deformation_map_allbins_new=np.zeros_like(deformation_map_allbins)

    mapx_base, mapy_base = np.meshgrid(np.arange(nx), np.arange(ny))
    mapx_base=np.tile(mapx_base,reps=(nb_slices,1,1))
    mapy_base = np.tile(mapy_base, reps=(nb_slices, 1, 1))
    deformation_base=np.stack([mapx_base,mapy_base],axis=0)
    
    for gr in range(nb_gr):
        deformation_map_allbins_new[:,gr]=deformation_map_allbins[:,gr]-deformation_map_allbins[:,index_ref]+deformation_base

    return deformation_map_allbins_new




def register_allbins_to_baseline(all_volumes,file_model,config_train,niter=1):


    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")
    nb_gr=all_volumes.shape[0]
    
    
    
    #pad_amount=config_train["padding"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    n = all_volumes.shape[-1]
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    pad_amount = ((0, 0), (pad, pad), (pad, pad))
    print(pad_amount)

    nb_features=config_train["nb_features"]
    inshape=np.pad(all_volumes[0],pad_amount,mode="constant").shape[1:]
    #print(inshape)
    vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model.load_weights(file_model)
    
    registered_volumes=copy(all_volumes)
    i=0
    while i<niter:
        print("Registration for iter {}".format(i+1))
        for gr in range(nb_gr):
            registered_volumes[gr]=register_motionbin_simple(vxm_model,all_volumes,gr,pad_amount)

        all_volumes=copy(registered_volumes)
        i+=1
    return registered_volumes


def best_channel_selection(data_for_nav,nav_traj,nav_image_size,shifts=list(range(-30,30))):
    nb_channels=data_for_nav.shape[0]
    image_nav_all_channels = []
    displacements_all_ch = []
    bottom = -shifts[0]
    top = nav_image_size[0] - shifts[-1]
    for j in range(nb_channels):
        images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[j], axis=0), nav_traj,
                                                                 nav_image_size, b1=None)
        image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
        # plt.figure()
        # plt.imshow(image_nav_ch.reshape(-1, int(npoint / 2)).T, cmap="gray")
        # plt.title("Image channel {}".format(j))
        image_nav_all_channels.append(image_nav_ch)

        displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, lambda_tv=0.001)
        displacements_all_ch.append(displacements)

    displacements_all_ch = np.array(displacements_all_ch)
    pca = PCAComplex(n_components_=1)
    disp_transf = pca.fit_transform(displacements_all_ch)

    return np.argsort(disp_transf[:, 0]),image_nav_all_channels

def register_motionbin_simple(vxm_model,all_volumes,gr,pad_amount):
    curr_gr=gr
    moving_volume=np.pad(all_volumes[curr_gr],pad_amount,mode="constant")
    nb_slices=all_volumes.shape[1]

    print(all_volumes.shape)

    
    while curr_gr>0:
        input=np.stack([np.pad(all_volumes[curr_gr-1],pad_amount,mode="constant"),moving_volume],axis=0)
        
        x_val_fixed,x_val_moving=format_input_voxelmorph(input,((0,0),(0,0),(0,0)),sl_down=0,sl_top=nb_slices)
        val_input=[x_val_moving[...,None],x_val_fixed[...,None]]
        


        val_pred=vxm_model.predict(val_input)
        moving_volume=val_pred[0][:,:,:,0]
       
        curr_gr=curr_gr-1
    return unpad(moving_volume,pad_amount)




def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]



def format_input_voxelmorph(all_volumes,pad_amount,sl_down=5,sl_top=-5,normalize=True,all_groups_combination=False,exclude_zero_slices=True):
    nb_gr=all_volumes.shape[0]
    nb_slices=all_volumes.shape[1]
    fixed_volume=[]
    moving_volume=[]

    #Filtering out slices with only 0 as it seems to be buggy

    if exclude_zero_slices:
        sl_down_non_zeros = 0
        while not(np.any(all_volumes[:,sl_down_non_zeros])):
            sl_down_non_zeros+=1

        sl_top_non_zeros=nb_slices
        while not(np.any(all_volumes[:,sl_top_non_zeros-1])):
            sl_top_non_zeros-=1


        sl_down=np.maximum(sl_down,sl_down_non_zeros)
        sl_top=np.minimum(sl_top,sl_top_non_zeros)
    
    print(sl_down)
    print(sl_top)


    for gr in range(nb_gr-1):
        fixed_volume.append(all_volumes[gr,sl_down:sl_top])
        moving_volume.append(all_volumes[gr+1,sl_down:sl_top])

    if all_groups_combination:
        shift=2
        while shift<nb_gr:
            for gr in range(nb_gr - shift):
                fixed_volume.append(all_volumes[gr, sl_down:sl_top])
                moving_volume.append(all_volumes[gr + shift, sl_down:sl_top])
            shift+=1


    fixed_volume=np.array(fixed_volume)
    moving_volume=np.array(moving_volume)

    fixed_volume=fixed_volume.reshape(-1,fixed_volume.shape[-2],fixed_volume.shape[-1])
    moving_volume=moving_volume.reshape(-1,moving_volume.shape[-2],moving_volume.shape[-1])

    if normalize:
        fixed_volume/=np.max(fixed_volume,axis=(1,2),keepdims=True)
        moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)


    # fix data
    fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
    moving_volume = np.pad(moving_volume, pad_amount, 'constant')

    #x_train_fixed=copy(fixed_volume)
    #x_train_moving=copy(moving_volume)

    #x_train_fixed/=np.max(x_train_fixed,axis=(1,2),keepdims=True)
    #x_train_moving/=np.max(x_train_moving,axis=(1,2),keepdims=True)

    

    return fixed_volume,moving_volume

def format_input_voxelmorph_3D(all_volumes,pad_amount,sl_down=5,sl_top=-5,normalize=True,all_groups_combination=False):
    nb_gr=all_volumes.shape[0]

    fixed_volume=[]
    moving_volume=[]

    for gr in range(nb_gr-1):
        fixed_volume.append(all_volumes[gr,:])
        moving_volume.append(all_volumes[gr+1,:])

    if all_groups_combination:
        shift=2
        while shift<nb_gr:
            for gr in range(nb_gr - shift):
                fixed_volume.append(all_volumes[gr, :])
                moving_volume.append(all_volumes[gr + shift, :])
            shift+=1


    fixed_volume=np.array(fixed_volume)
    moving_volume=np.array(moving_volume)

    #print(fixed_volume.shape)
    #print(moving_volume.shape)

    #fixed_volume=np.expand_dims(fixed_volume,axis=0)
    #moving_volume=np.expand_dims(moving_volume,axis=0)

    if normalize:
        fixed_volume/=np.max(fixed_volume,axis=(1,2,3),keepdims=True)
        moving_volume/=np.max(moving_volume,axis=(1,2,3),keepdims=True)


    # fix data
    fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
    moving_volume = np.pad(moving_volume, pad_amount, 'constant')

    #x_train_fixed=copy(fixed_volume)
    #x_train_moving=copy(moving_volume)

    #x_train_fixed/=np.max(x_train_fixed,axis=(1,2),keepdims=True)
    #x_train_moving/=np.max(x_train_moving,axis=(1,2),keepdims=True)

    #print(fixed_volume.shape)

    return fixed_volume,moving_volume




def kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part,select_first_rep=False):
    
    nb_channels,nb_allspokes,nb_part,npoint=kdata_all_channels_all_slices.shape
    nb_slices=nb_part-nb_rep_center_part+1
    
    kdata_all_channels_all_slices_new=np.zeros((nb_channels,nb_allspokes,nb_slices,npoint),dtype=kdata_all_channels_all_slices.dtype)
    kdata_all_channels_all_slices_new[:,:,:int(nb_slices/2)]=kdata_all_channels_all_slices[:,:,:int(nb_slices/2)]
    
    if not(select_first_rep):
        kdata_center=kdata_all_channels_all_slices[:,:,int(nb_slices/2):(int(nb_slices/2)+nb_rep_center_part)]
        
        non_zeros_count=np.count_nonzero(kdata_center,axis=2)
        #print(non_zeros_count.shape)
        non_zeros_count[non_zeros_count==0]=1
        kdata_center=np.sum(kdata_center,axis=2)/non_zeros_count

        kdata_all_channels_all_slices_new[:,:,int(nb_slices/2)]=kdata_center
    else:#select first repetition of central partition
        kdata_all_channels_all_slices_new[:,:,int(nb_slices/2)]=kdata_all_channels_all_slices[:,:,int(nb_slices/2)]
    
    kdata_all_channels_all_slices_new[:,:,(int(nb_slices/2)+1):]=kdata_all_channels_all_slices[:,:,(int(nb_slices/2)+nb_rep_center_part):]
    return kdata_all_channels_all_slices_new






def plot_deformation_map(deformation_map,us=4,save_file=None):
    '''
    deformation_map : shape 2*n*n 2D new coordinates map for an image
    '''
    npoint=deformation_map.shape[-1]
    X, Y = np.meshgrid(np.arange(npoint), np.arange(npoint))
    ne.plot.flow([np.moveaxis(np.stack([deformation_map[0] - X, deformation_map[1] - Y], axis=0), 0, -1)[::us, ::us]],
                 width=5, scale=us,plot_block=False)

    if save_file is not None:
        plt.savefig(save_file)


def resample_deformation(deformation_map,resolution):
    _, nb_gr, nb_slices, npoint, _=deformation_map.shape
    X_lowres, Y_lowres = np.meshgrid(np.arange(resolution), np.arange(resolution))
    deformation_flow = np.stack([deformation_map[0] - np.expand_dims(X_lowres, axis=(0, 1)),
                                 deformation_map[1] - np.expand_dims(Y_lowres, axis=(0, 1))], axis=0)
    deformation_flow_resized = np.zeros((2, nb_gr, nb_slices, npoint, npoint), dtype=deformation_flow.dtype)

    for gr in tqdm(range(nb_gr)):
        for sl in tqdm(range(nb_slices)):
            for l in range(2):
                deformation_flow_resized[l, gr, sl] = resize(deformation_flow[l, gr, sl], (npoint, npoint), order=3)

    deformation_flow_resized = deformation_flow_resized * npoint / resolution
    X, Y = np.meshgrid(np.arange(npoint), np.arange(npoint))
    deformation_map_resized = np.stack([deformation_flow_resized[0] + np.expand_dims(X, axis=(0, 1)),
                                        deformation_flow_resized[1] + np.expand_dims(Y, axis=(0, 1))], axis=0)
    deformation_map = deformation_map_resized.astype(deformation_map.dtype)
    return deformation_map




def weights_aggregate_center_part(weights, nb_rep_center_part,select_first_rep=False):
    nb_gr, _, nb_allspokes, nb_part, _ = weights.shape
    nb_slices = nb_part - nb_rep_center_part + 1

    weights_new = np.zeros((nb_gr, 1, nb_allspokes, nb_slices, 1), dtype=weights.dtype)
    print(weights_new.shape)
    weights_new[:, :, :, :int(nb_slices / 2)] = weights[:, :, :, :int(nb_slices / 2)]

    if not(select_first_rep):
        weights_center = weights[:, :, :, int(nb_slices / 2):(int(nb_slices / 2) + nb_rep_center_part)]

        non_zeros_count = np.count_nonzero(weights_center, axis=-2)
        print(non_zeros_count.shape)
        non_zeros_count[non_zeros_count == 0] = 1
        weights_center = np.sum(weights_center, axis=-2) / non_zeros_count

        weights_new[:, :, :, int(nb_slices / 2)] = weights_center
    else:#select first repetition of central partition
        weights_new[:, :, :, int(nb_slices / 2)]=weights[:, :, :, int(nb_slices / 2)]

    weights_new[:, :, :, (int(nb_slices / 2) + 1):] = weights[:, :, :, (int(nb_slices / 2) + nb_rep_center_part):]
    return weights_new



def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    print(dims)
    print(strides)
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6#.reshape(-1, *blck)

def stuff_patches_3D(sh,patches,strd,blck):
    out = np.zeros(sh, patches.dtype)
    sh = np.asanyarray(sh)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[out.strides * strd, out.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(out, strides=strides, shape=dims)
    data6[...]=patches.reshape(data6.shape)
    return out


def proj_LLR(vol, strd, blck, threshold):
    x_patches = cutup(vol, strd, blck)
    patch_shape = x_patches.shape
    x_patches = x_patches.reshape((np.prod(patch_shape[:3]), -1))

    a, s, bh = da.linalg.svd(da.asarray(x_patches))
    bh = np.array(bh)
    s = np.array(s)
    a = np.array(a)

    sig = pywt.threshold(s, threshold)

    print("Retained comp % {}".format(np.sum(sig > 0) / a.shape[0] * 100))
    x_patches_lr = a @ np.diag(sig) @ bh
    u = stuff_patches_3D(vol.shape, x_patches_lr.reshape(patch_shape), strd, blck)
    return u