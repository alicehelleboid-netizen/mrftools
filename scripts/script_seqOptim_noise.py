

from mrftools.utils_simu import *
from mrftools.dictoptimizers import SimpleDictSearch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#generate_epg_dico_T1MRFSS_from_sequence_file("mrf_sequence_adjusted.json","mrf_dictconf_SimReco2.json",4,sim_mode="mid_point",start=0,window=int(1400/50))

#TR_list,FA_list,TE_list=load_sequence_file("mrf_sequence_adjusted.json",3,1.87/1000)
#
# generate_epg_dico_T1MRFSS("mrf_sequence_adjusted_1_87.json","mrf_dictconf_Dico2_Invivo.json",FA_list,TE_list,4,1.87/1000)
#

#
#
# #https://cds.ismrm.org/protected/16MPresentations/videos/0429/index.html
#


# with open("./dico/mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_0_55T_fT1.json") as f:
#     dict_config = json.load(f)

# fat_amp = np.array(dict_config["fat_amp"])
# fat_shift = -np.array(dict_config["fat_cshift"])

DFs=[[-120, -60, 0, 60, 120], [-46,-23, 0, 23, 46], [-240, -120 ,0, 120, 240]] 
FFs=[[0.,0.002, 0.004, 0.007, 0.009, 0.1], [0.3, 0.35, 0.4, 0.45, 0.5], [0.,0.1,0.2,0.3,0.4,0.5,0.95]]
SNR_list=[10,40,70,100]

# fileseq_list=[
#     r"/home/ahelleboid/Low_field_MRF/mrftools/dico/mrf_sequence_adjusted_0_55T_TEv1.json",
# ]

# recoveries=[1.89]

# ff_std = np.zeros((len(FFs), len(DFs), len(SNR_list)))
# wT1_std = np.zeros_like(ff_std)
# ff_erreur = np.zeros_like(ff_std)
# wT1_erreur = np.zeros_like(ff_std)


# for l,FF in enumerate(FFs):
#     print('#####',l)
#     FFs_list=FF

#     for k,DF in enumerate(DFs):

#         T1_w=1
#         dTs=np.arange(-500,1000,100)*10**-3
#         #dTs=np.array([500,500,1000])
#         # DFs=[-30,0,30] # à modif le B0
#         #DFs=[-60,-30,0,30,60]
#         # FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.95] # à modif les FF pour différents range faire ++ subplot
#         B1=[0.5,0.7,1]
#         DFs_list=DF
#         #FFs=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7]


#         for i in range (len(SNR_list)):
#             # SNR
#             SNR=SNR_list[i]
#             sigma=1/SNR
#             noise_size=100

#             group_size=8
#             noise_type="Relative"

#             wT1_errors=[]
#             ff_errors=[]

#             df_results=pd.DataFrame(index=[f+"_"+str(recoveries[j]) for j,f in enumerate(fileseq_list)],columns=["Error rel wT1","Error abs FF","std wT1","std FF","TR"])
#             min_TR_delay=1.87/1000
#             L=None

#             labels=["MRF T1-FF 1400 Spokes"]
#             for j,fileseq_1 in enumerate(fileseq_list):
#                 recovery=recoveries[j]
#                 ind = fileseq_1 + "_" + str(recovery)

#                 TR_list_1,FA_list_1,TE_list_1=load_sequence_file(fileseq_1,recovery,min_TR_delay)

#                 s,s_w,s_f,keys=simulate_gen_eq_signal(TR_list_1,FA_list_1,TE_list_1,FFs_list,DFs_list,T1_w+dTs,300/1000,B1,T_2w=40/1000,T_2f=80/1000,amp=fat_amp,shift=fat_shift,sigma=sigma,noise_size=noise_size,group_size=group_size,return_fat_water=True,noise_type=noise_type)#,amp=np.array([1]),shift=np.array([-418]),sigma=None):

#                 s_w=s_w.reshape(s_w.shape[0],-1).T
#                 s_f=s_f.reshape(s_f.shape[0],-1).T

#                 s=s.reshape(s.shape[0],-1)

#                 nb_signals=s.shape[-1]
#                 mask=None
#                 pca = True
#                 threshold_pca_bc = 20


#                 split=nb_signals+1
#                 split=10
#                 dict_optim_bc_cf =SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
#                                                 threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False, useGPU_simulation=False,
#                                                 gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,return_matched_signals=True)


#                 all_maps,matched_signals=dict_optim_bc_cf.search_patterns_test((s_w,s_f,keys),s)

#                 if L is None:
#                     L = np.random.choice(range(matched_signals.shape[-1]))

#                 # plt.figure()
#                 # plt.plot(matched_signals[:,l],label=labels[j])
#                 # plt.plot(s[:,l])

#                 keys_all=list(product(keys,FFs_list))
#                 keys_all=[(*rest, a) for rest,a in keys_all]
#                 keys_all=np.array(keys_all)

#                 key="wT1"
#                 map=all_maps[0][0][key].reshape(-1,noise_size)
#                 keys_all_current=np.array(list(keys_all[:,0])*noise_size).reshape(noise_size,-1).T
#                 error=np.abs(map-keys_all_current)
#                 error_wT1=np.mean(np.mean(error,axis=-1)/keys_all[:,0])
#                 std_wT1=np.mean(np.std(error,axis=-1)/keys_all[:,0])
#                 # print('errorwT1, std_wT1', error_wT1, std_wT1)


#                 key="ff"
#                 map=all_maps[0][0][key].reshape(-1,noise_size)
#                 keys_all_current=np.array(list(keys_all[:,-1])*noise_size).reshape(noise_size,-1).T
#                 error=np.abs(map-keys_all_current)
#                 error_ff=np.mean(np.mean(error,axis=-1))
#                 std_ff = np.mean(np.std(error,axis=-1))
#                 df_results.loc[ind]=[error_wT1,error_ff,std_wT1,std_ff,np.sum(TR_list_1)]
#                 # print('error-ff, std_ff', error_ff, std_ff)

#                 ff_std[l][k][i]=std_ff
#                 wT1_std[l][k][i]=std_wT1
#                 ff_erreur[l][k][i]=error_ff
#                 wT1_erreur[l][k][i]=error_wT1


SNR_values=[10,40,70,100]
# np.save('ff_std_save.npy',ff_std)
# np.save('wT1_std_save.npy',wT1_std)
# np.save('ff_erreur_save.npy',ff_erreur)
# np.save('wT1_erreur_save.npy',wT1_erreur)

# plt.figure(figsize=(12, 8))

# # 1er subplot
# plt.subplot(5, 4, 1)
# plt.plot(SNR_values, ff_std, marker='o')
# plt.title("ff_std")
# plt.xlabel("SNR")
# plt.ylabel("ff_std")

# # 2ème subplot
# plt.subplot(5, 4, 2)
# plt.plot(SNR_values, wT1_std, marker='o')
# plt.title("wT1_std")
# plt.xlabel("SNR")
# plt.ylabel("wT1_std")

# # 3ème subplot
# plt.subplot(5, 4, 3)
# plt.plot(SNR_values, ff_erreur, marker='o')
# plt.title("ff_erreur")
# plt.xlabel("SNR")
# plt.ylabel("ff_erreur")

# # 4ème subplot
# plt.subplot(5, 4, 4)
# plt.plot(SNR_values, wT1_erreur, marker='o')
# plt.title("wT1_erreur")
# plt.xlabel("SNR")
# plt.ylabel("wT1_erreur")

# plt.tight_layout()
# plt.savefig("all_plots.png")  
# plt.show()

ff_std=np.load('/home/ahelleboid/Low_field_MRF/mrftools/ff_std_save.npy')
wT1_std=np.load('/home/ahelleboid/Low_field_MRF/mrftools/wT1_std_save.npy')
ff_erreur=np.load('/home/ahelleboid/Low_field_MRF/mrftools/ff_erreur_save.npy')
wT1_erreur=np.load('/home/ahelleboid/Low_field_MRF/mrftools/wT1_erreur_save.npy')
print( ff_erreur.shape)

fig, axes = plt.subplots(nrows=2, ncols=len(FFs), figsize=(5*len(FFs),10), sharex=True)

# Trouver les bornes communes pour toutes les erreurs
all_ff_err = np.array(ff_erreur).flatten()
all_wT1_err = np.array(wT1_erreur).flatten()
ymin_ff, ymax_ff = all_ff_err.min(), all_ff_err.max()
ymin_wT1, ymax_wT1 = all_wT1_err.min(), all_wT1_err.max()

for l, FF in enumerate(FFs):

    # ff_erreur
    ax = axes[0,l]
    for k, DF in enumerate(DFs):
        ax.plot(SNR_values, ff_erreur[l][k], marker='o', label=f'DF={DF}')
    ax.set_title(f'ff_erreur – FF={FF}')
    ax.set_ylabel("ff_erreur")
    ax.set_ylim(ymin_ff, ymax_ff)  # même échelle partout
    ax.legend()

    # wT1_erreur
    ax = axes[1,l]
    for k, DF in enumerate(DFs):
        ax.plot(SNR_values, wT1_erreur[l][k], marker='o', label=f'DF={DF}')
    ax.set_title(f'wT1_erreur – FF={FF}')
    ax.set_ylabel("wT1_erreur")
    ax.set_ylim(ymin_wT1, ymax_wT1)  # même échelle partout

for ax in axes[-1,:]:
    ax.set_xlabel("SNR")

plt.tight_layout()
plt.savefig("all_erreurs_FF_DF.png", dpi=200)
plt.show()








































# from utils_simu import *
# from dictoptimizers import SimpleDictSearch
# import json
# from image_series import *
# from trajectory import *
# import pickle
# import warnings
# warnings.filterwarnings("ignore")

# with open("./mrf_dictconf_SimReco2.json") as f:
#     dict_config = json.load(f)

# fat_amp = np.array(dict_config["fat_amp"])
# fat_shift = -np.array(dict_config["fat_cshift"])

# nspoke=8
# spokes_count=760
# ntimesteps=int(spokes_count/nspoke)
# nb_channels=1
# compression_factor=2

# nspoke=int(spokes_count/ntimesteps)
# num=1
# file="./data/KneePhantom/Phantom{}/paramMap_Control.mat".format(num)


# m_=MapFromFile("Knee2D_Optim",file=file,rounding=True)
# m_.buildParamMap()
# m_.change_resolution(compression_factor)
# npoint_image=m_.image_size[-1]
# # plt.close("all")
# # plt.figure()
# # plt.imshow(makevol(m_.paramMap["wT1"],m_.mask>0))
# # plt.savefig("test_compression_map.jpg")

# npoint = npoint_image*2
# image_size = (npoint_image, npoint_image)

# radial_traj = Radial(total_nspokes=spokes_count, npoint=npoint)

# fileseq_basis="./mrf_sequence_adjusted.json"

# with open(fileseq_basis, "r") as file:
#     seq_config_base = json.load(file)

# def cost_function_simul_breaks_random_FA_KneePhantom(params):
#     #global result
#     global spokes_count

#     global min_TR_delay
#     global DFs
#     global FFs
#     global T1s
#     global B1s
#     global bound_min_FA
#     global num_breaks_TE
#     global num_params_FA

#     #global bound_max_FA

#     global lambda_FA
#     global lambda_T1
#     global lambda_time
#     global lambda_FF
#     global inversion

#     global seq_config_base

#     global useGPU
#     global m_
#     global radial_traj
#     global ntimesteps

#     start_time=datetime.now()
#     group_size = 8

#     bound_max_FA=params[-1]
#     params_for_curve=params[:-1]
#     TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

#     m_.build_ref_images_bloch(TR_,FA_,TE_)
#     m_.images_series=m_.images_series.astype("complex64")

#     s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
#                                                amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
#                                                return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

#     s_w = s_w.reshape(s_w.shape[0], -1).T
#     s_f = s_f.reshape(s_f.shape[0], -1).T

#     data = m_.generate_kdata(radial_traj, useGPU=useGPU)
#     data = np.array(data)
#     data = data.reshape(spokes_count, -1, npoint)
#     volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=useGPU,ntimesteps=ntimesteps)

#     mask = m_.mask
#     pca = True
#     threshold_pca_bc = 10

#     split = 100
#     dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
#                                         threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
#                                         useGPU_simulation=False,
#                                         gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
#                                         return_matched_signals=True)

#     all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), volumes_all)

#     key = "wT1"
#     map = all_maps[0][0][key][m_.paramMap["ff"]<0.7]*1000
#     map_gt=m_.paramMap[key][m_.paramMap["ff"]<0.7]
#     error = np.abs(map - map_gt)
#     error_wT1 = np.mean(error/ map_gt)

#     print("wT1 Cost : {}".format(error_wT1))

#     std_wT1 = np.std(error/ map_gt)
#     print("wT1 Std : {}".format(std_wT1))

#     key = "ff"
#     map = all_maps[0][0][key]
#     map_gt = m_.paramMap[key]
#     error = np.abs(map - map_gt)
#     error_ff = np.mean(error)

#     print("FF Cost : {}".format(error_ff))

#     key = "df"
#     map = all_maps[0][0][key]/1000
#     map_gt = m_.paramMap[key]
#     error = np.abs(map - map_gt)
#     error_df = np.mean(error)

#     print("DF Cost : {}".format(error_df))

#     key = "attB1"
#     map = all_maps[0][0][key]
#     map_gt = m_.paramMap[key]
#     error = np.abs(map - map_gt)
#     error_b1 = np.mean(error)

#     print("B1 Cost : {}".format(error_b1))

#     # num_breaks_TE=len(TE_breaks)
#     FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
#     # print("FA Cost : {}".format(FA_cost))

#     time_cost = np.sum(TR_)
#     print("Time Cost : {}".format(time_cost))
#     #test_time=time_cost-TR_[-1]
#     #print("Time Test : {}".format(test_time))

#     result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_DF * error_df + lambda_B1 * error_b1 +lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1

#     end_time=datetime.now()
#     print(end_time-start_time)

#     return result



# def cost_function_simul_breaks_random_FA_KneePhantom_variablespokecounts(params):
#     #global result
#     #global spokes_count

#     global min_TR_delay
#     global DFs
#     global FFs
#     global T1s
#     global B1s
#     global bound_min_FA
#     global num_breaks_TE
#     global num_params_FA

#     #global bound_max_FA

#     global lambda_FA
#     global lambda_T1
#     global lambda_time
#     global lambda_FF
#     global inversion

#     global seq_config_base

#     global useGPU
#     global m_
#     #global radial_traj

#     start_time=datetime.now()
#     group_size = 8

#     spokes_count=520+8*int(params[-1]*(1400-520)/8)
#     ntimesteps = int(spokes_count / nspoke)

#     print("spokes_count: {}".format(spokes_count))
#     bound_max_FA=params[-2]
#     params_for_curve=params[:-2]
#     TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)




#     m_.build_ref_images_bloch(TR_,FA_,TE_)
#     m_.images_series=m_.images_series.astype("complex64")

#     s, s_w, s_f, keys = simulate_gen_eq_signal(TR_, FA_, TE_, FFs, DFs, T1s, 300 / 1000,B1s, T_2w=40 / 1000, T_2f=80 / 1000,
#                                                amp=fat_amp, shift=fat_shift, sigma=None, group_size=group_size,
#                                                return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

#     s_w = s_w.reshape(s_w.shape[0], -1).T
#     s_f = s_f.reshape(s_f.shape[0], -1).T

#     radial_traj = Radial(total_nspokes=spokes_count, npoint=npoint)

#     data = m_.generate_kdata(radial_traj, useGPU=useGPU)
#     data = np.array(data)
#     data = data.reshape(spokes_count, -1, npoint)
#     volumes_all = simulate_radial_undersampled_images(data, radial_traj, image_size, density_adj=True, useGPU=useGPU,ntimesteps=ntimesteps)

#     mask = m_.mask
#     pca = True
#     threshold_pca_bc = 10

#     split = 1000
#     dict_optim_bc_cf = SimpleDictSearch(mask=mask, niter=0, seq=None, trajectory=None, split=split, pca=pca,
#                                         threshold_pca=threshold_pca_bc, log=False, useGPU_dictsearch=False,
#                                         useGPU_simulation=False,
#                                         gen_mode="other", movement_correction=False, cond=None, ntimesteps=None,
#                                         return_matched_signals=True)

#     all_maps, matched_signals = dict_optim_bc_cf.search_patterns_test((s_w, s_f, keys), volumes_all)

#     key = "wT1"
#     map = all_maps[0][0][key][m_.paramMap["ff"]<0.7]*1000
#     map_gt=m_.paramMap[key][m_.paramMap["ff"]<0.7]
#     error = np.abs(map - map_gt)
#     error_wT1 = np.mean(error/ map_gt)

#     print("wT1 Cost : {}".format(error_wT1))

#     std_wT1 = np.std(error/ map_gt)
#     print("wT1 Std : {}".format(std_wT1))

#     key = "ff"
#     map = all_maps[0][0][key]
#     map_gt = m_.paramMap[key]
#     error = np.abs(map - map_gt)
#     error_ff = np.mean(error)

#     print("FF Cost : {}".format(error_ff))

#     key = "df"
#     map = all_maps[0][0][key]/1000
#     map_gt = m_.paramMap[key]
#     error = np.abs(map - map_gt)
#     error_df = np.mean(error)

#     print("DF Cost : {}".format(error_df))

#     key = "attB1"
#     map = all_maps[0][0][key]
#     map_gt = m_.paramMap[key]
#     error = np.abs(map - map_gt)
#     error_b1 = np.mean(error)

#     print("B1 Cost : {}".format(error_b1))

#     # num_breaks_TE=len(TE_breaks)
#     FA_cost = np.mean((np.abs(np.diff(FA_[1:]))))
#     # print("FA Cost : {}".format(FA_cost))

#     time_cost = np.sum(TR_)
#     print("Time Cost : {}".format(time_cost))
#     #test_time=time_cost-TR_[-1]
#     #print("Time Test : {}".format(test_time))

#     result=lambda_T1 * error_wT1 + lambda_FF * error_ff + lambda_DF * error_df + lambda_B1 * error_b1 +lambda_FA * FA_cost + lambda_time * time_cost+lambda_stdT1 * std_wT1

#     plt.close("all")
#     plt.figure()
#     plt.plot(TE_[1:])
#     plt.savefig("TE_log.jpg")

#     plt.figure()
#     plt.plot(FA_[1:])
#     plt.savefig("FA_log.jpg")

#     plt.figure()
#     plt.plot(TR_[1:-1])
#     plt.savefig("TR_log.jpg")

#     end_time=datetime.now()
#     print(end_time-start_time)

#     return result

# class LogCost(object):
#     def __init__(self,f):
#         self.cost_values=[]
#         self.it=1
#         self.f=f

#     def __call__(self,x,**kwargs):
#         global min_TR_delay
#         global num_breaks_TE
#         global num_params_FA
#         global bound_min_FA
#         global inversion

#         print("############# ITERATION {}###################".format(self.it))
#         self.cost_values.append(self.f(x))
#         self.it +=1

#         with open("x_random_FA_US_H4_variablespokecount_log.pkl", "wb") as file:
#             pickle.dump(x, file)

#         spokes_count = 520 + 8 * int(x[-1] * (1400 - 520) / 8)


#         print("spokes_count: {}".format(spokes_count))
#         bound_max_FA = x[-2]
#         params_for_curve = x[:-2]
#         TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay, spokes_count,
#                                                                     num_breaks_TE, num_params_FA, bound_min_FA,
#                                                                     bound_max_FA, inversion)

#         plt.close("all")
#         plt.figure()
#         plt.plot(TE_[1:])
#         plt.savefig("TE_log_it{}.jpg".format(self.it))

#         plt.figure()
#         plt.plot(FA_[1:])
#         plt.savefig("FA_log_it{}.jpg".format(self.it))

#         plt.figure()
#         plt.plot(TR_[1:-1])
#         plt.savefig("TR_log_it{}.jpg".format(self.it))

#     def reset(self):
#         self.cost_values = []
#         self.it=1

# #sigma2=0.02**2

# #spokes_count=760


# min_TR_delay=1.87*10**-3
# DFs=[-45,-30,-15,0,15,30,45]
# FFs=[0.]
# B1s=[0.5,0.6,0.7,0.8,0.9,1]
# T1s=np.array([1000,1100,1200,1300,1400,1500,1600,1800,2000])/1000
# T1s=np.array([1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1600,1700,1800,2000])/1000
# #T1s=np.array([1000,1050,1100,1150,1200,1250,1300])/1000



# bound_min_TE=2.2/1000
# bound_min_FA=5*np.pi/180
# num_breaks_TE=3
# #bound_max_FA=70*np.pi/180

# H=4
# num_params_FA=2*H
# bounds=[(100,400)]*(num_breaks_TE)+[(0.0022,0.006)]*(num_breaks_TE+1)+[(0,1)]*num_params_FA+[(0,4)]+[(15*np.pi/180,70*np.pi/180)]+[(0,1)]

# from scipy.optimize import LinearConstraint
# A_TEbreaks=np.zeros((1,len(bounds)))
# A_TEbreaks[0,:num_breaks_TE]=1
# con1=LinearConstraint(A_TEbreaks,lb=-np.inf,ub=spokes_count-100)
# constraints=(con1)

# from scipy.optimize import differential_evolution
# lambda_FA=0.
# lambda_time=0.01
# lambda_FF=2
# lambda_T1=1
# lambda_DF=4
# lambda_B1=1
# lambda_stdT1=0
# inversion=True
# useGPU=False

# log_cost=LogCost(cost_function_simul_breaks_random_FA_KneePhantom_variablespokecounts)

# res=differential_evolution(cost_function_simul_breaks_random_FA_KneePhantom_variablespokecounts,bounds=bounds,callback=log_cost,constraints=constraints,maxiter=1000)#,constraints=constraints)




# from utils_simu import *
# from dictoptimizers import SimpleDictSearch
# import json
# from image_series import *
# from trajectory import *
# import pickle
# import warnings

# ts="20240507_112330"

# file_params="./optims/res_simul_random_FA_US_{}.pkl".format(ts)
# file_config="./optims/random_FA_opt_config_{}.json".format(ts)
# file_cost="./optims/log_cost_{}.npy".format(ts)

# log_cost=np.load(file_cost)
# plt.figure()
# plt.plot(log_cost)
# plt.savefig("./optims/log_cost_{}.jpg".format(ts))

# with open(file_config, "rb") as file:
#     config=json.load(file)

# with open(file_params, "rb") as file:
#     res=pickle.load(file)

# x=res.x

# bound_min_FA=config["bound_min_FA"]
# bound_max_FA = x[-2]
# params_for_curve = x[:-2]
# num_breaks_TE=config["num_breaks_TE"]
# H=config["H"]
# num_params_FA=2*H
# min_TR_delay=2.22*10**-3
# spokes_count = 520 + 8 * int(x[-1] * (1400 - 520) / 8)
# inversion=config["inversion"]
# lambda_time=config["lambda_time"]


# sequence_file="./mrf_sequence_random_FA_variablesp_thighs_time_{}_allparams.json".format("_".join(str.split(str(lambda_time),".")))
# #file_params="x_random_FA_US_H4_variablespokecount_log_time0_01_allparams.pkl"




# TR_, FA_, TE_ = convert_params_to_sequence_breaks_proportional_random_FA(params_for_curve, min_TR_delay, spokes_count,
#                                                                     num_breaks_TE, num_params_FA, bound_min_FA,
#                                                                     bound_max_FA, inversion)

# plt.close("all")
# plt.figure()
# plt.plot(TE_[1:])
# plt.savefig("./optims/TE_{}.jpg".format(ts))

# plt.figure()
# plt.plot(FA_[1:])
# plt.savefig("./optims/FA_{}.jpg".format(ts))


# reco=x[-3]



# new_sequence_file=str.split(sequence_file,".json")[0]+"_{}.json".format(np.round(min_TR_delay*1000,2))

# dictconf="mrf_dictconf_Dico2_Invivo.json"
# dictconf_light="mrf_dictconf_Dico2_Invivo_light_for_matching.json"
# generate_epg_dico_T1MRFSS(new_sequence_file,dictconf,FA_,TE_,np.round(reco,2),min_TR_delay,rep=3)
# generate_epg_dico_T1MRFSS(new_sequence_file,dictconf_light,FA_,TE_,np.round(reco,2),min_TR_delay,rep=3)

# with open(new_sequence_file) as f:
#     sequence_config = json.load(f)

# TE=sequence_config["TE"]
# TE=np.array(TE)

# #TE=np.maximum(TE,2.2)
# TE=np.round(TE,2)
# #TE=np.maximum(np.round(TE,2),2.2)

# TE_file="TE_temp_{}.text".format(ts)

# with open(TE_file,"w") as file:
#     for te in TE:
#         file.write(str(te)+"\n")

# #pd.DataFrame(np.maximum(np.round(TE,2),2.2)).to_clipboard()

# FA=sequence_config["B1"]

# FA=np.array(FA)
# FA=np.round(FA,3)
# #pd.DataFrame(np.round(FA,3)).to_clipboard(excel=False,index=False)

# FA_file="FA_temp_{}.text".format(ts)

# with open(FA_file,"w") as file:
#     for fa in FA:
#         file.write(str(fa)+"\n")
# print(np.min(np.array(TE)))
# print(np.max(np.array(TE)))
# print(np.min(np.array(FA)))
# print(np.max(np.array(FA)))


# # it=0
# # plt.close("all")
# # plt.figure()
# # plt.plot(TE_[1:])
# # plt.savefig("TE_log_it{}.jpg".format(it))
# #
# # plt.figure()
# # plt.plot(FA_[1:])
# # plt.savefig("FA_log_it{}.jpg".format(it))
# #
# # plt.figure()
# # plt.plot(TR_[1:-1])
# # plt.savefig("TR_log_it{}.jpg".format(it))











# import pickle
# with open("res_simul_random_FA_US_all_params.pkl","wb") as file:
#     pickle.dump(res, file)

# plt.figure()
# plt.plot(log_cost.cost_values)

# import pickle
# with open("res_simul_random_FA_US.pkl","rb") as file:
#     res=pickle.load(file)

# cost_function_simul_breaks_random_FA_KneePhantom(res.x)

# params=res.x
# bound_max_FA=params[-1]
# params_for_curve=params[:-1]
# TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

# plt.figure()
# plt.plot(FA_[1:])


# plt.figure()
# plt.plot(TE_[1:])
# res.x
# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp1400_optimized_DE_Simu_FF_random_US_v2.json","./mrf_dictconf_SimReco2.json",FA_,TE_,3.73,1.87/1000,fileseq_basis="./mrf_sequence_adjusted.json")











# import pickle
# from utils_simu import *
# with open("./optims/res_simul_random_20221028_125515.pkl","rb") as file:
#     res=pickle.load(file)


# with open("./optims/random_opt_config_20221028_125515.json","rb") as file:
#     config=json.load(file)

# #cost_function_simul_breaks_random(res.x)

# inversion=config["inversion"]
# min_TR_delay=config["min_TR_delay"]
# spokes_count=config["spokes_count"]
# bound_min_TE=config["bound_min_TE"]
# bound_max_TE=config["bound_max_TE"]
# bound_min_FA=config["bound_min_FA"]

# params=res.x
# bound_max_FA=params[-1]
# params_for_curve=params[:-1]
# TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,bound_min_TE,bound_max_TE,bound_min_FA,bound_max_FA,inversion)

# plt.figure()
# plt.plot(FA_[1:])


# plt.figure()
# plt.plot(TE_[1:])
# res.x
# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_v3.json","./mrf_dictconf_SimReco2_lightDFB1.json",FA_,TE_,4,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")





# import pickle
# from utils_simu import *
# with open("./optims/res_simul_random_FA_US_20230123_180436.pkl","rb") as file:
#     res=pickle.load(file)


# with open("./optims/random_FA_opt_config_20230123_180436.json","rb") as file:
#     config=json.load(file)

# #cost_function_simul_breaks_random(res.x)

# inversion=config["inversion"]
# min_TR_delay=config["min_TR_delay"]
# spokes_count=config["spokes_count"]
# bound_min_TE=config["bound_min_TE"]
# bound_max_TE=config["bound_max_TE"]
# bound_min_FA=config["bound_min_FA"]
# num_breaks_TE=config["num_breaks_TE"]
# num_params_FA=config["num_params_FA"]




# params=res.x
# bound_max_FA=params[-1]
# params_for_curve=params[:-1]
# TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)

# plt.figure()
# plt.plot(FA_[1:])


# plt.figure()
# plt.plot(TE_[1:])
# res.x
# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_random_FA_v2_1_87.json","./mrf_dictconf_SimReco2_light_matching.json",FA_,TE_,3.53,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")






# import pickle

# from utils_simu import *
# from dictoptimizers import SimpleDictSearch
# import json
# from image_series import *
# from trajectory import *
# from utils_simu import *
# with open("x_random_FA_US_H4_allparams_log.pkl","rb") as file:
#     x=pickle.load(file)

# # with open("random_FA_US_seqOptim_config.json","r") as file:
# #     config=json.load(file)

# #cost_function_simul_breaks_random(res.x)

# inversion=True

# spokes_count=760

# min_TR_delay=2.22*10**-3
# bound_min_TE=2.2/1000
# bound_min_FA=5*np.pi/180
# num_breaks_TE=3
# include_flat=False
# nb_flat_spokes = 760

# H=4
# num_params_FA=2*H


# params=x
# bound_max_FA=params[-1]
# params_for_curve=params[:-1]
# TR_, FA_, TE_ = convert_params_to_sequence_breaks_random_FA(params_for_curve, min_TR_delay,spokes_count,num_breaks_TE,num_params_FA,bound_min_FA,bound_max_FA,inversion)


# if include_flat:

#     FA_ = FA_ + [6 * np.pi / 180] * nb_flat_spokes
#     # print("FA_ {}".format(FA_[-1]))
#     TE_ = list(TE_) + [3.45 / 1000] * nb_flat_spokes
#     # print("TE {}".format(TE_[-1]))
#     TR_[1:] = np.array(TE_[1:]) + min_TR_delay
#     TR_[-1] = TR_[-1] + params_for_curve[-1]
#     TR_ = list(TR_)
#     # print("TR {}".format(TR_[-1]))

# plt.close("all")
# plt.figure()
# plt.plot(np.array(FA_)[1:]*180/np.pi,color="r")


# plt.figure()
# plt.plot(np.array(TE_)[1:]*1000)
# recovery=np.round(params_for_curve[-1],2)



# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_Body_plus_flat_v2_2_22.json","mrf_dictconf_Dico2_Invivo.json",FA_,TE_,recovery,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")
# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_Body_plus_flat_v2_2_22.json","mrf_dictconf_Dico2_Invivo_light_for_matching.json",FA_,TE_,recovery,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")

# FA_new=list(FA_)+[6*np.pi/180]*(1400-760)
# TE_new=list(TE_)+[3.45/1000]*(1400-760)

# plt.close("all")
# plt.figure()
# plt.plot(np.array(FA_new)[1:]*180/np.pi)


# plt.figure()
# plt.plot(np.array(TE_new)[1:]*1000)


# write_seq_file(r"./mrf_sequence_adjusted_optimized_Body_plus_flat_v1_2_22.json",FA_new,TE_new,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")

# recovery=0
# recovery=3

# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_Body_plus_flat_v1_2_22.json","mrf_dictconf_Dico2_Invivo.json",FA_new,TE_new,recovery,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")
# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_Body_plus_flat_v1_2_22.json","mrf_dictconf_Dico2_Invivo_light_for_matching.json",FA_new,TE_new,recovery,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")


# import pickle
# from utils_simu import *
# with open("./optims/res_simul_breaks_common_20221031_143619.pkl","rb") as file:
#     res=pickle.load(file)


# with open("./optims/breaks_common_opt_config_20221031_143619.json","rb") as file:
#     config=json.load(file)

# #cost_function_simul_breaks_random(res.x)

# inversion=config["inversion"]
# min_TR_delay=config["min_TR_delay"]
# spokes_count=config["spokes_count"]
# num_breaks_TE=config["num_breaks_TE"]
# num_breaks_FA=config["num_breaks_FA"]

# params=res.x
# bound_max_FA=params[-1]
# params_for_curve=params[:-1]
# TR_,FA_,TE_=convert_params_to_sequence_breaks_common(res.x,min_TR_delay,num_breaks_TE,num_breaks_FA,spokes_count)
# # #
# plt.figure()
# plt.plot(FA_[1:])


# plt.figure()
# plt.plot(TE_[1:])
# res.x
# generate_epg_dico_T1MRFSS(r"./mrf_sequence_adjusted_optimized_M0_T1_local_optim_correl_crlb_filter_sp760_optimized_DE_Simu_FF_v4.json","./mrf_dictconf_SimReco2_lightDFB1.json",FA_,TE_,3.8,min_TR_delay,fileseq_basis="./mrf_sequence_adjusted.json")



