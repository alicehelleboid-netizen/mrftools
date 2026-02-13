import json
import pathlib
import numpy as np
import itertools
from itertools import product
import epgpy as epg
import pickle

import finufft
from .utils_mrf import groupby
from .dictmodel import Dictionary
from tqdm import tqdm

from mrftools.trajectory import Radial
# from mrftools.dictoptimizers import *
  
def calculate_sensitivity_map(kdata,res=16,hanning_filter=True,density_adj=False):
    '''
    Calculates coil sensitivity maps
    inputs:
    kdata - numpy array containing k-space data nb_slices x nb_channels x nb_segments x npoint
    res - k-space data cutoff (int)
    hanning_filter - bool
    density_adj - bool
    outputs:
    b1 - coil sensitivity map of size nb_slices x nb_channels x npoint/2 x npoint/2
    '''
    nb_allspokes=kdata.shape[-2]
    npoint=kdata.shape[-1]

    kdata = kdata.astype(np.complex64)

    image_size=(int(npoint/2),int(npoint/2))

    trajectory=Radial(total_nspokes=nb_allspokes,npoint=npoint)
    traj_all = trajectory.get_traj().astype("float32")
    traj_all=traj_all.reshape(-1,traj_all.shape[-1])
    npoint = kdata.shape[-1]
    center_res = int(npoint / 2)

    nb_channels=kdata.shape[1]
    nb_slices=kdata.shape[0]

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
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

    
    coil_sensitivity=np.zeros((nb_slices,nb_channels,)+image_size,dtype=kdata.dtype)
    
    for sl in tqdm(range(nb_slices)):
        coil_sensitivity[sl]=finufft.nufft2d1(
            np.ascontiguousarray(traj_all[:, 0]), 
            np.ascontiguousarray(traj_all[:, 1]),
            np.ascontiguousarray(kdata_for_sensi[sl].reshape(nb_channels,-1)), 
            image_size)
    
    #coil_sensitivity=coil_sensitivity.reshape(*kdata.shape[:-2],*image_size)

    if coil_sensitivity.ndim==3:
        print("Ndim 3)")
        b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
        #b1 = b1 / np.max(np.abs(b1.flatten()))
    else:#first dimension contains slices
        b1=coil_sensitivity.copy()
        for i in range(coil_sensitivity.shape[0]):
            b1[i]=coil_sensitivity[i] / np.linalg.norm(coil_sensitivity[i], axis=0)
            #b1[i]=b1[i] / np.max(np.abs(b1[i].flatten()))
    return b1

class T1MRFSS:
    def __init__(self, FA, TI, TE, TR, B1,T_recovery,nrep,rep=None):
        """ build sequence """
        seqlen = len(B1)
        self.TR=TR
        self.inversion = epg.T(180, 0) # perfect inversion
        self.T_recovery=T_recovery
        self.nrep=nrep
        self.rep=rep
        seq=[]
        for r in range(nrep):
            curr_seq = [epg.Offset(TI)]
            for i in range(seqlen):
                echo = [
                    epg.T(FA * B1[i], 90),
                    epg.Wait(TE[i]),
                    epg.ADC,
                    epg.Wait(TR[i] - TE[i]),
                    epg.SPOILER,
                ]
                curr_seq.extend(echo)
            recovery=[epg.Wait(T_recovery)]
            curr_seq.extend(recovery)
            self.len_rep = len(curr_seq)
            seq.extend(curr_seq)
        self._seq = seq

    def __call__(self, T1, T2, g, att,**kwargs):
        """ simulate sequence """
        seq=[]
        rep=self.rep
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[self.inversion, epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        
        result=np.asarray(epg.simulate(seq, **kwargs))
        if rep is not None:
            result = result.reshape((self.nrep, -1) + result.shape[1:])[rep]
        return result

def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new

def create_new_seq(FA_list,TE_list,min_TR_delay,TI,FA_factor=5):
    seq_config_new={}
    seq_config_new["FA"]=FA_factor
    seq_config_new["TI"]=TI
    seq_config_new["TE"] = list(np.array(TE_list[1:]) * 10 ** 3)
    seq_config_new["TR"] = list((np.array(TE_list[1:])+min_TR_delay) * 10 ** 3)
    seq_config_new["B1"] = list(np.array(FA_list[1:]) * 180 / np.pi / 5)
    
    
    

    return seq_config_new

def generate_epg_dico_T1MRFSS_from_sequence(sequence_config,filedictconf,recovery,rep=2,overwrite=True,sim_mode="mean",start=None,window=None, dest=None,prefix_dico="dico"):
    if type(filedictconf)==str:
        with open(filedictconf) as f:
            dict_config = json.load(f)
    
    elif type(filedictconf)==dict:
        dict_config=filedictconf



    # generate signals
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    att = dict_config["B1_att"]
    df = dict_config["delta_freqs"]
    df = [- value / 1000 for value in df]  # temp
    # df = np.linspace(-0.1, 0.1, 101)

    TR_total = np.sum(sequence_config["TR"])

    sequence_config["T_recovery"] = recovery*1000
    sequence_config["nrep"] = rep

    TR_delay=np.round(sequence_config["TR"][0]-sequence_config["TE"][0],2)

    seq = T1MRFSS(**sequence_config)


    fat_amp = np.array(dict_config["fat_amp"])
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options
    if window is None:
        window = dict_config["window_size"]


    if start is None:
        dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(recovery))
    else:
        dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(recovery),start)

    if dest is not None:
        dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
    # print("Generating dictionary {}".format(dictfile))

    # water
    print("Generate water signals.")
    water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])
    water = water.reshape((rep, -1) + water.shape[1:])[-1]

    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)

        water = water[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    # fat
    print("Generate fat signals.")
    # eval = "dot(signal, amps)"
    # args = {"amps": fat_amp}
    # merge df and fat_cs df to dict
    fatdf = [[cs + f for cs in fat_cs] for f in df]
    fat = seq(T1=[fT1], T2=fT2, att=[[att]], g=[[[fatdf]]])#, eval=eval, args=args)
    fat=fat @ fat_amp
    fat = fat.reshape((rep, -1) + fat.shape[1:])[-1]

    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)
        fat = fat[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    water = np.array(water)
    fat = np.array(fat)
    # join water and fat
    print("Build dictionary.")
    keys = list(itertools.product(wT1, fT1, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    # print("Save dictionary.")
    mrfdict = Dictionary(keys, values)
    # mrfdict.save(dictfile, overwrite=overwrite)
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":recovery,"initial_repetitions":rep,"window":window,"sim_mode":sim_mode}
    return mrfdict,hdr,dictfile



def load_sequence_file(fileseq,recovery,min_TR_delay):

    if type(fileseq)==str:
        with open(fileseq, "r") as file:
            seq_config = json.load(file)
    elif type(fileseq)==dict:
        seq_config = fileseq

    TI = seq_config["TI"] * 10 ** -3
    TE = list(np.array(seq_config["TE"]) * 10 ** -3)
    TR = list(np.array(TE)+min_TR_delay)
    FA = seq_config["FA"]
    B1 = seq_config["B1"]
    # B1[:600]=[1.5]*600
    B1 = list(1 * np.array(B1))


    TR[-1] = TR[-1] + recovery

    TR_list = [TI] + TR
    TE_list = [0] + TE
    FA_list = [np.pi] + list(np.array(B1) * FA * np.pi / 180)
    return TR_list,FA_list,TE_list

# def generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=8.32):

#     _,FA_list,TE_list=load_sequence_file(sequence_file,reco,min_TR_delay/1000)
#     seq_config=create_new_seq(FA_list,TE_list,min_TR_delay,TI)

#     dictfile,hdr=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf,FA_list,TE_list,reco,min_TR_delay/1000,TI=TI)
#     dictfile_light,hdr_light=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf_light,FA_list,TE_list,reco,min_TR_delay/1000,TI=TI)

#     dico_full_with_hdr={"hdr":hdr,
#                         "hdr_light":hdr_light,
#                         "dictfile":dictfile,
#                         "dictfile_light":dictfile_light}
    
#     dico_full_name=str.split(dictfile,".dict")[0]+".pkl"
#     with open(dico_full_name,"wb") as file:
#         pickle.dumps(dico_full_with_hdr)

#     return

def calc_A_B_gen(FA_, TR_, T_1,B1):
    b_s = np.exp(-np.array(TR_).reshape(-1, 1) / T_1)
    b_s = np.expand_dims(b_s,axis=tuple(range(2,B1.ndim)))
    FA_init=FA_[0]
    FA_=np.array(FA_)
    FA_=np.expand_dims(FA_,axis=tuple(range(1,B1.ndim)))
    FA_=FA_*B1
    FA_[0]=FA_init
    a_s = np.cos(FA_)
    k_s = (a_s * b_s)[:len(TR_)]
    b_s_one_rep = b_s[:len(TR_)]
    A = np.prod(k_s, axis=0)
    cumprod_ks = np.cumprod(k_s[::-1], axis=0)[:-1]
    ones = np.ones((1,) + cumprod_ks.shape[1:])
    # B = np.sum(np.cumprod(np.array([1]+list(k_s))[:-1][::-1])[::-1]*(1-b_s_one_rep))
    B = np.sum(np.concatenate([ones, cumprod_ks])[::-1] * (1 - b_s_one_rep), axis=0)
    l = B / (1 - A)
    return A, l

def simulate_gen(u_0, TR_list, FA_list, nb_rep, T_1,B1):
    b_s = np.exp(-np.array(TR_list * nb_rep).reshape(-1, 1) / T_1)
    b_s = np.expand_dims(b_s, axis=tuple(range(2, B1.ndim)))
    FA_list=np.array(FA_list * nb_rep)
    FA_init = FA_list[0]
    FA_list = np.expand_dims(FA_list, axis=tuple(range(1, B1.ndim)))
    FA_list = FA_list * B1
    FA_list[0] = FA_init
    a_s = np.cos(FA_list)
    N = len(a_s)
    u_s = [u_0]
    u = u_0

    for j in range(N):
        u = b_s[j] * a_s[j] * u + (1 - b_s[j])
        u_s.append(u)
    u_s = np.array(u_s)
    return u_s

def simulate_gen_eq(TR_list, FA_list, T_1,B1):
    A, l = calc_A_B_gen(FA_list, TR_list, T_1,B1)
    u_i = simulate_gen(l, TR_list, FA_list, 1, T_1,B1)

    return u_i

def simulate_gen_eq_signal(TR_list, FA_list, TE_list, FF, df, T_1w, T_1f,B1, T_2w=40 / 1000, T_2f=80 / 1000,
                           amp=np.array([1]), shift=np.array([-418]), sigma=None, list_deriv=None,noise_size=None,noise_type="Absolute",group_size=None,return_fat_water=False):
    T_1w = np.array(T_1w)
    T_1f = np.array(T_1f)
    df = np.array(df)
    FF = np.array(FF)
    B1 = np.array(B1)

    if (np.array(T_1w).shape == ()):
        T_1w = np.array([T_1w])
    if (np.array(T_1f).shape == ()):
        T_1f = np.array([T_1f])
    if (np.array(df).shape == ()):
        df = np.array([df])
    if (np.array(FF).shape == ()):
        FF = np.array([FF])
    if (np.array(B1).shape == ()):
        B1 = np.array([B1])

    if not (T_1w.shape == (1,)):
        T_1w = np.squeeze(T_1w)
    if not (T_1f.shape == (1,)):
        T_1f = np.squeeze(T_1f)
    if not (df.shape == (1,)):
        df = np.squeeze(df)
    if not (FF.shape == (1,)):
        FF = np.squeeze(FF)

    if not (B1.shape == (1,)):
        B1 = np.squeeze(B1)

    keys = list(product(list(T_1w), list(T_1f), list(B1), list(df)))

    #keys=np.array(keys).reshape(len(T_1w),len(T_1f),1,len(df),4)

    T_1w = np.expand_dims(T_1w, axis=0)
    T_1f = np.expand_dims(T_1f, axis=0)
    B1=np.expand_dims(B1, axis=(0, 1))
    df = np.expand_dims(df, axis=(0, 1,2))
    FF = np.expand_dims(FF, axis=(0, 1, 2,3))


    s_iw = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1w, T_2w,B1)[1:]
    s_iw = np.expand_dims(s_iw, axis=(2, -1))
    s_if = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1f, T_2f,B1, amp, shift)[1:]
    s_if = np.expand_dims(s_if, axis=(1, -1))
    s_iw, s_if = np.broadcast_arrays(s_iw, s_if)

    if group_size is not None:
        s_iw=np.array([np.mean(gp, axis=0) for gp in groupby(s_iw, group_size)])
        s_if = np.array([np.mean(gp, axis=0) for gp in groupby(s_if, group_size)])


    s_i = FF * s_if + (1 - FF) * s_iw



    if sigma is not None:
        if noise_size is None:
            e_i = np.random.normal(size=s_i.shape) + 1j * np.random.normal(size=s_i.shape)
            # print(e_i.shape)
            # e_i*=np.abs(np.mean(s_i))/np.abs(e_i)/snr
            # e_i*=np.abs(s_i)/np.abs(e_i)/sigma
            if noise_type=="Absolute":
                e_i *= sigma
            elif noise_type=="Relative":
                e_i*=sigma*np.abs(s_i)
            else:
                raise ValueError("Unknown noise_type")

            s_i += e_i
        else:
            e_i = np.random.normal(size=s_i.shape+(noise_size,)) + 1j * np.random.normal(size=s_i.shape+(noise_size,))
            s_i = np.expand_dims(s_i, axis=-1)
            if noise_type == "Absolute":
                e_i *= sigma
            elif noise_type == "Relative":
                e_i *= sigma * np.abs(s_i)
            else:
                raise ValueError("Unknown noise_type")

            s_i=s_i+e_i

    if list_deriv is None:
        if return_fat_water:
            return s_i,s_iw,s_if,keys
        else:
            return s_i

    else:
        dico_calc_deriv = {}
        if "ff" in list_deriv:
            ds_ff = s_if - s_iw
            ds_ff = ds_ff[1:]
            reps = tuple(np.ones(s_i.ndim - 1).astype(int)) + (FF.shape[-1],)
            # print(reps)
            ds_ff = np.tile(ds_ff, reps)
            ds_ff = np.moveaxis(ds_ff, 0, -1)
            # print(ds_ff.shape)
            dico_calc_deriv["ff"] = ds_ff

        if "wT1" in list_deriv:
            dT1 = 10 ** -3
            s_iw_dT1 = simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1w + dT1, T_2w,B1)[1:]
            s_iw_dT1 = np.expand_dims(s_iw_dT1, axis=(2, -1))
            ds_T1 = (1 - FF) * (s_iw_dT1 - s_iw) / dT1
            ds_T1 = ds_T1[1:]
            ds_T1 = np.moveaxis(ds_T1, 0, -1)
            dico_calc_deriv["wT1"] = ds_T1

        return s_i, dico_calc_deriv


def simulate_gen_eq_transverse(TR_list, FA_list, TE_list, df, T_1, T_2,B1, amp=np.array([1]), shift=np.array([0])):
    u_i = simulate_gen_eq(TR_list, FA_list, T_1,B1)
    ax_expand = tuple(range(1, df.ndim))
    FA_init = FA_list[0]
    TEs = np.expand_dims(np.array(TE_list), axis=ax_expand)
    FAs = np.expand_dims(np.array(FA_list), axis=ax_expand)
    ax_expand_B1=tuple(range(B1.ndim, df.ndim))
    B1=np.expand_dims(B1,axis=ax_expand_B1)

    FAs = FAs * B1
    FAs[0] = FA_init

    chemical_shift = (np.exp(np.array(TE_list).reshape(-1, 1) * 2j * np.pi * shift.reshape(1, -1))) @ amp
    chemical_shift = np.expand_dims(chemical_shift, axis=ax_expand)
    E_2 = np.exp(TEs * (2j * np.pi * df - 1 / T_2)) * (chemical_shift)
    u = np.expand_dims(u_i[:-1], axis=-1)
    s_i = u * np.sin(np.array(FAs)) * E_2
    return s_i
