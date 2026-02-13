try:
    import matplotlib
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
except:
    pass


from mrftools.utils_simu import simulate_gen_eq_signal
from scipy import ndimage


import pandas as pd
from mrftools.utils_mrf_simu import create_random_map,voronoi_volumes,normalize_image_series,build_mask_from_volume,generate_kdata,build_mask_single_image,buildROImask,correct_mvt_kdata,create_map
# from mutools.optim.dictsearch import dictsearch
from mrftools.dictmodel import Dictionary
import itertools
# from mrfsim import groupby,makevol,load_data,loadmat
import finufft
from tqdm import tqdm
import matplotlib.animation as animation
from mrftools.trajectory import *

try:
    import pycuda.autoinit
    from pycuda.gpuarray import GPUArray, to_gpu
    from cufinufft import cufinufft

except:
    pass

import gc
try:
    import h5py
except:
    pass

DEFAULT_wT2 = 40
DEFAULT_fT1 = 300
DEFAULT_fT2 = 80

DEFAULT_ROUNDING_wT1=0
DEFAULT_ROUNDING_wT2=0
DEFAULT_ROUNDING_fT1=0
DEFAULT_ROUNDING_fT2=0
DEFAULT_ROUNDING_df=3 #df in kHz but chemical shifts generally order of magnitude in Hz
DEFAULT_ROUNDING_attB1=2
DEFAULT_ROUNDING_ff=2

DEFAULT_IMAGE_SIZE =(256,256)

UNITS ={
    "attB1":"a.u",
    "df":"kHz",
    "wT1":"ms",
    "fT1":"ms",
    "wT2":"ms",
    "fT2":"ms",
    "ff":"a.u"
}

PARAMS_WINDOWS={
    "attB1":[0,2],
    "df":[-0.5,0.5],
    "wT1":[0,2000],
    "fT1":[0,1000],
    "wT2":[0,500],
    "fT2":[0,500],
    "ff":[0,1]


}

def dump_function(func):
    """
    Decorator to print function call details.

    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):

        print(f"{func.__module__}.{func.__qualname__} ")
        return func(*args, **kwargs)

def wrapper_rounding(func):
    def wrapper(self, *args, **kwargs):
        print("Building Param Map")
        func(self,*args,**kwargs)
        if self.paramDict["rounding"]:
            print("Warning : Values in the initial map are being rounded")
            for paramName in self.paramMap.keys():
                self.roundParam(paramName,self.paramDict["rounding_"+paramName])

        if "dict_overrides" in self.paramDict:
            print("Warning : Overriding map values for params {}" .format(self.paramDict["dict_overrides"].keys()))
            for paramName in self.paramDict["dict_overrides"].keys():
                self.paramMap[paramName]=np.ones(self.paramMap[paramName].shape)*self.paramDict["dict_overrides"][paramName]

    return wrapper


class ImageSeries(object):

    def __init__(self,name,dict_config={},**kwargs):
        self.name=name
        self.dict_config=dict_config
        self.paramDict = kwargs
        if "image_size" not in self.paramDict:
            self.paramDict["image_size"]=DEFAULT_IMAGE_SIZE

        if "nb_rep" not in self.paramDict:
            self.paramDict["nb_rep"]=1

        # if "sim_mode" not in self.paramDict:
        #     self.paramDict["sim_mode"]="mean" #mid_point

        if "gen_mode" not in self.paramDict:
            self.paramDict["gen_mode"]=None #loop

        self.image_size=self.paramDict["image_size"]
        self.images_series=None
        self.cached_images_series=None

        self.mask =np.ones(self.image_size)
        self.paramMap=None

        self.list_movements=[]

        if "rounding" not in self.paramDict:
            self.paramDict["rounding"]=False
        else:
            if self.paramDict["rounding"]:
                if "rounding_wT1" not in self.paramDict:
                    self.paramDict["rounding_wT1"] = DEFAULT_ROUNDING_wT1
                if "rounding_wT2" not in self.paramDict:
                    self.paramDict["rounding_wT2"] = DEFAULT_ROUNDING_wT2
                if "rounding_fT1" not in self.paramDict:
                    self.paramDict["rounding_fT1"] = DEFAULT_ROUNDING_fT1
                if "rounding_fT2" not in self.paramDict:
                    self.paramDict["rounding_fT2"] = DEFAULT_ROUNDING_fT2
                if "rounding_ff" not in self.paramDict:
                    self.paramDict["rounding_ff"] = DEFAULT_ROUNDING_ff
                if "rounding_attB1" not in self.paramDict:
                    self.paramDict["rounding_attB1"] = DEFAULT_ROUNDING_attB1
                if "rounding_df" not in self.paramDict:
                    self.paramDict["rounding_df"] = DEFAULT_ROUNDING_df


        self.fat_amp=[0.0586, 0.0109, 0.0618, 0.1412, 0.66, 0.0673]
        fat_cs = [-101.1, 208.3, 281.0, 305.7, 395.6, 446.2]
        self.fat_cs = [- value / 1000 for value in fat_cs]  # temp


    def build_ref_images(self,seq,norm=None,phase=None):
        print("Building Ref Images")
        if self.paramMap is None:
            return ValueError("buildparamMap should be called prior to image simulation")

        list_keys = ["wT1","wT2","fT1","fT2","attB1","df","ff"]
        for k in list_keys:
            if k not in self.paramMap:
                raise ValueError("key {} should be in the paramMap".format(k))

        map_all_on_mask = np.stack(list(self.paramMap.values())[:-1], axis=-1)
        map_ff_on_mask = self.paramMap["ff"]

        params_all = np.reshape(map_all_on_mask, (-1, 6))
        params_unique = np.unique(params_all, axis=0)

        wT1_in_map = np.unique(params_unique[:, 0])
        wT2_in_map = np.unique(params_unique[:, 1])
        fT1_in_map = np.unique(params_unique[:, 2])
        fT2_in_map = np.unique(params_unique[:, 3])
        attB1_in_map = np.unique(params_unique[:, 4])
        df_in_map = np.unique(params_unique[:, 5])

        # Simulating the image sequence

        # water
        print("Simulating Water Signal")

        if self.paramDict["gen_mode"]=="loop":
            water_list=[]
            print("Simulation in loop mode")
            for param in tqdm(params_unique):
                current_water = seq(T1=param[0], T2=param[1], att=param[4], g=param[5])
                water_list.append(current_water)
            water = np.squeeze(np.array(water_list))
            del water_list
            water=water.T

        else:
            water = seq(T1=wT1_in_map, T2=wT2_in_map, att=[[attB1_in_map]], g=[[[df_in_map]]])


        # if self.paramDict["sim_mode"]=="mean":
        #     water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
        # elif self.paramDict["sim_mode"]=="mid_point":
        #     water = water[(int(window / 2) - 1):-1:window]
        # else:
        #     raise ValueError("Unknow sim_mode")


        # fat
        print("Simulating Fat Signal")
        eval = "dot(signal, amps)"
        args = {"amps": self.fat_amp}


        if self.paramDict["gen_mode"] == "loop":
            fat_list = []
            print("Simulation in loop mode")
            for param in tqdm(params_unique):
                current_fat = seq(T1=param[2], T2=param[3], att=param[4], g=[cs + param[5] for cs in self.fat_cs],eval=eval,args=args)
                fat_list.append(current_fat)
            fat = np.squeeze(np.array(fat_list))
            del fat_list
            fat = fat.T
            

        else:
            # merge df and fat_cs df to dict
            fatdf_in_map = [[cs + f for cs in self.fat_cs] for f in df_in_map]
            print(eval)
            print(args)
            fat = seq(T1=[fT1_in_map], T2=fT2_in_map, att=[[attB1_in_map]], g=[[[fatdf_in_map]]], eval=eval, args=args)
            fat = np.dot(fat, self.fat_amp)

        # if self.paramDict["sim_mode"]=="mean":
        #     fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
        # elif self.paramDict["sim_mode"]=="mid_point":
        #     fat = fat[(int(window / 2) - 1):-1:window]
        # else:
        #     raise ValueError("Unknow sim_mode")

        # building the time axis
        self.build_timeline(seq.TR)

        # join water and fat
        print("Build dictionary.")
        if self.paramDict["gen_mode"] == "loop":
            keys=[tuple(param) for param in params_unique]
            values=np.stack((water, fat),axis=-1)
            values = np.moveaxis(values, 0, 1)
        else :
            keys = list(itertools.product(wT1_in_map, wT2_in_map, fT1_in_map, fT2_in_map, attB1_in_map, df_in_map))
            values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
            values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)
        # mrfdict = dictsearch.Dictionary(keys, values)
        mrfdict = Dictionary(keys=keys, values=values)

        images_series = np.zeros(self.image_size + (values.shape[-2],), dtype=np.complex64)
        #water_series = images_series.copy()
        #fat_series = images_series.copy()

        map_all_with_ff = np.append(map_all_on_mask, map_ff_on_mask.reshape(-1, 1), axis=-1)
        #unique_values, index_unique = np.unique(map_all_with_ff, return_inverse=True,axis=0)

        #images_in_mask_unique = np.array(
        #    [mrfdict[tuple(pixel_params[:-1])][:, 0] * (1 - pixel_params[-1]) + mrfdict[tuple(
        #        pixel_params[:-1])][:, 1] * (pixel_params[-1]) for pixel_params in unique_values])
        print("Building image series")
        images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])

        if norm is not None :
            images_in_mask *= np.expand_dims(norm,axis=1)

        if phase is not None:
            images_in_mask *= np.expand_dims(np.exp(1j*phase),axis=1)

        #water_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0]  for (i, pixel_params) in enumerate(map_all_on_mask)])
        #fat_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 1]  for (i, pixel_params) in enumerate(map_all_on_mask)])
        print("Image series built")

        images_series[self.mask > 0, :] = images_in_mask
        #water_series[self.mask > 0, :] = water_in_mask
        #fat_series[self.mask > 0, :] = fat_in_mask

        images_series = np.moveaxis(images_series, -1, 0)
        #water_series = np.moveaxis(water_series, -1, 0)
        #fat_series = np.moveaxis(fat_series, -1, 0)


        #images_series=normalize_image_series(images_series)
        self.images_series=images_series

        #self.water_series = water_series
        #self.fat_series=fat_series

        self.cached_images_series=images_series

    def build_ref_images_bloch(self,TR_, FA_, TE_,norm=None,phase=None):
        print("Building Ref Images")
        if self.paramMap is None:
            return ValueError("buildparamMap should be called prior to image simulation")

        list_keys = ["wT1","wT2","fT1","fT2","attB1","df","ff"]
        for k in list_keys:
            if k not in self.paramMap:
                raise ValueError("key {} should be in the paramMap".format(k))

        map_all_on_mask = np.stack(list(self.paramMap.values())[:-1], axis=-1)
        map_ff_on_mask = self.paramMap["ff"]

        params_all = np.reshape(map_all_on_mask, (-1, 6))
        params_unique = np.unique(params_all, axis=0)

        wT1_in_map = np.unique(params_unique[:, 0])
        wT2_in_map = np.unique(params_unique[:, 1])
        fT1_in_map = np.unique(params_unique[:, 2])
        fT2_in_map = np.unique(params_unique[:, 3])
        attB1_in_map = np.unique(params_unique[:, 4])
        df_in_map = np.unique(params_unique[:, 5])

        # Simulating the image sequence

        # water
        print("Simulating Signals")

        s, water, fat, keys=simulate_gen_eq_signal(TR_, FA_, TE_, [0.1], df_in_map*1000, wT1_in_map/1000, fT1_in_map[0] / 1000, attB1_in_map, T_2w=wT2_in_map[0] / 1000, T_2f=fT2_in_map[0] / 1000,
                               amp=np.array(self.fat_amp), shift=np.array(self.fat_cs)*1000, sigma=None, group_size=1,
                               return_fat_water=True)  # ,amp=np.array([1]),shift=np.array([-418]),sigma=None):

        # if self.paramDict["sim_mode"]=="mean":
        #     fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
        # elif self.paramDict["sim_mode"]=="mid_point":
        #     fat = fat[(int(window / 2) - 1):-1:window]
        # else:
        #     raise ValueError("Unknow sim_mode")

        # building the time axis
        self.build_timeline(np.array(TR_[1:])*1000)


        # join water and fat
        print("Build dictionary.")

        water=water.reshape(water.shape[0],-1)
        fat = fat.reshape(fat.shape[0], -1)
        if self.paramDict["gen_mode"] == "loop":
            values=np.stack((water, fat),axis=-1)
            values = np.moveaxis(values, 0, 1)
        else :
            values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
            values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)
        mrfdict = dictsearch.Dictionary(keys, values)

        images_series = np.zeros(self.image_size + (values.shape[-2],), dtype=np.complex_)
        #water_series = images_series.copy()
        #fat_series = images_series.copy()

        map_all_with_ff = np.append(map_all_on_mask, map_ff_on_mask.reshape(-1, 1), axis=-1)
        #unique_values, index_unique = np.unique(map_all_with_ff, return_inverse=True,axis=0)

        #images_in_mask_unique = np.array(
        #    [mrfdict[tuple(pixel_params[:-1])][:, 0] * (1 - pixel_params[-1]) + mrfdict[tuple(
        #        pixel_params[:-1])][:, 1] * (pixel_params[-1]) for pixel_params in unique_values])
        print("Building image series")
        map_all_on_mask = map_all_on_mask[:, [0, 2, 4, 5]]
        map_all_on_mask[:,0]/=1000
        map_all_on_mask[:, 1] /= 1000
        map_all_on_mask[:, 3] *= 1000
        images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])

        if norm is not None :
            images_in_mask *= np.expand_dims(norm,axis=1)

        if phase is not None:
            images_in_mask *= np.expand_dims(np.exp(1j*phase),axis=1)

        #water_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0]  for (i, pixel_params) in enumerate(map_all_on_mask)])
        #fat_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 1]  for (i, pixel_params) in enumerate(map_all_on_mask)])
        print("Image series built")

        images_series[self.mask > 0, :] = images_in_mask
        #water_series[self.mask > 0, :] = water_in_mask
        #fat_series[self.mask > 0, :] = fat_in_mask

        images_series = np.moveaxis(images_series, -1, 0)
        #water_series = np.moveaxis(water_series, -1, 0)
        #fat_series = np.moveaxis(fat_series, -1, 0)

        #images_series=images_series.astype("complex64")
        #images_series=normalize_image_series(images_series)
        self.images_series=images_series

        #self.water_series = water_series
        #self.fat_series=fat_series

        self.cached_images_series=images_series


    def build_timeline(self,TR_list):

        # t = np.cumsum([np.sum(dt, axis=0) for dt in groupby(np.array(TR_list), window)]).reshape(1,-1)

        t = np.cumsum(TR_list).reshape(1, -1)

        if "nb_total_slices" in self.paramDict:
            nb_rep = self.paramDict["nb_rep"]
            final_time=t[0,-1]
            nb_timesteps = t.shape[1]
            t=np.resize(t,(nb_rep,nb_timesteps))
            rest = np.tile([self.paramDict["resting_time"]+final_time],nb_rep)
            rest[0] = 0
            rest = np.cumsum(rest).reshape(-1,1)
            t = t+rest

        self.t=t

    def add_movements(self,list_movements):
        self.list_movements=[*self.list_movements,*list_movements]


    def generate_kdata(self,trajectory,useGPU=False,eps=1e-4,movement_correction=False,perc=80,nthreads=1,fftw=0):
        print("Generating kdata")
        #nspoke = trajectory.paramDict["nspoke"]
        #npoint = trajectory.paramDict["npoint"]
        

        traj = trajectory.get_traj()
        print(self.images_series.dtype)

        #traj = traj.reshape((self.images_series.shape[0], -1, traj.shape[-1]))
        if self.images_series.dtype=="complex64":
            traj = traj.astype("float32")

        if not (traj.shape[-1] == len(self.image_size)):
            raise ValueError("Trajectory dimension does not match Image Space dimension")

        size = self.image_size
        if trajectory.applied_timesteps is None:
            images_series = self.images_series
        else:
            images_series=self.images_series[trajectory.applied_timesteps]
        # images_series =normalize_image_series(self.images_series)

        if traj.shape[-1] == 2:  # 2D
            if self.list_movements == []:

                if not(useGPU):
                    
                    kdata = [
                                finufft.nufft2d2(t[:, 0], t[:, 1], p,nthreads=nthreads,fftw=fftw)
                                for t, p in tqdm(zip(traj, images_series))
                            ]

                else:
                    dtype = np.float32  # Datatype (real)
                    complex_dtype = np.complex64
                    N1, N2 = size[0], size[1]
                    M = traj.shape[1]
                    # Initialize the plan and set the points.
                    kdata = []

                    for i in tqdm(list(range(self.images_series.shape[0]))):

                        # print("Allocating input")
                        # start = datetime.now()

                        fk = images_series[i,:,:]
                        kx = traj[i, :, 0]
                        ky = traj[i, :, 1]

                        kx = kx.astype(dtype)
                        ky = ky.astype(dtype)
                        fk = fk.astype(complex_dtype)



                        # end=datetime.now()
                        # print(end-start)
                        # print("Allocating Output")
                        # start=datetime.now()

                        fk_gpu = to_gpu(fk)
                        c_gpu = GPUArray((M), dtype=complex_dtype)

                        # end=datetime.now()
                        # print(end-start)
                        #
                        # print("Executing FFT")
                        # start=datetime.now()

                        plan = cufinufft(2, (N1, N2), 1, eps=eps, dtype=dtype)
                        plan.set_pts(to_gpu(kx), to_gpu(ky))
                        plan.execute(c_gpu, fk_gpu)

                        c = np.squeeze(c_gpu.get())

                        fk_gpu.gpudata.free()
                        c_gpu.gpudata.free()

                        kdata.append(c)

                        del c
                        del kx
                        del ky
                        del fk
                        del fk_gpu
                        del c_gpu
                        plan.__del__()

                        # gc.collect()



                    gc.collect()

            else:
                df = pd.DataFrame(index=range(self.images_series.shape[0]), columns=["Timesteps", "Images"])
                df["Timesteps"] = self.t.reshape(-1, 1)
                df["Images"] = list(self.images_series)

                for movement in self.list_movements:
                    df = movement.apply(df)

                images_series = df.Images


                if not(useGPU):

                    # kdata = [
                    #     finufft.nufft2d2(t[:, 0], t[:, 1], p)
                    #     for t, p in zip(traj, images_series)
                    # ]


                    kdata = [
                                finufft.nufft2d2(
                                    np.ascontiguousarray(t[:, 0], dtype=np.float64),
                                    np.ascontiguousarray(t[:, 1], dtype=np.float64),
                                    np.ascontiguousarray(p, dtype=np.complex64)
                                )
                                for t, p in zip(traj, images_series)
                            ]
                    
                    


                else:
                    # Allocate memory for the nonuniform coefficients on the GPU.
                    dtype = np.float32  # Datatype (real)
                    complex_dtype = np.complex64
                    N1, N2 = size[0], size[1]
                    M = traj.shape[1]
                    c_gpu = GPUArray((M), dtype=complex_dtype)
                    # Initialize the plan and set the points.
                    kdata = []
                    for i in tqdm(list(range(self.images_series.shape[0]))):
                        fk = images_series.iloc[i]
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

        elif traj.shape[-1] == 3:  # 3D
            if not(useGPU):
                if self.list_movements == []:
                    # kdata = [
                    #     finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], p)
                    #     for t, p in zip(traj, images_series)
                    # ]
                    # kdata = []
                    # for i in tqdm(range(len(traj))):
                    #     kdata.append(finufft.nufft3d2(traj[i,:,2],traj[i, :, 0], traj[i, :, 1], images_series[i]))


                    kdata = []
                    for i in tqdm(range(len(traj))):
                        x = np.ascontiguousarray(traj[i, :, 2], dtype=np.float64)
                        y = np.ascontiguousarray(traj[i, :, 0], dtype=np.float64)
                        z = np.ascontiguousarray(traj[i, :, 1], dtype=np.float64)
                        p = np.ascontiguousarray(images_series[i], dtype=np.complex128)

                        kdata.append(finufft.nufft3d2(x, y, z, p))

                else:
                    kdata = []
                    for i, x in (enumerate(tqdm(zip(traj, images_series)))):

                        nb_rep = self.paramDict["nb_rep"]
                        current_data = pd.DataFrame(x[0], columns=["KX", "KY", "KZ"])
                        current_data['rep_number'] = current_data.groupby("KZ").ngroup()
                        current_image = x[1]
                        df = pd.DataFrame(index=range(self.paramDict["nb_rep"]), columns=["Timesteps", "Images"])
                        df["Timesteps"] = self.t[:, i].reshape(-1, 1)
                        images_for_df = list(
                            np.tile(current_image,(nb_rep,1,1,1)))
                        df["Images"] = images_for_df

                        for movements in self.list_movements:
                            df = movements.apply(df, useGPU)

                        if not(trajectory.reconstruct_each_partition):
                            values = np.array(list(current_data.groupby("KZ").apply(
                                lambda grp: finufft.nufft3d2(grp.KZ, grp.KX, grp.KY,
                                                             df.Images[np.unique(grp.rep_number)[0]])).values)).flatten()
                            kdata.append(values)
                        else:#outputs kdata for each repetition of the sequence separately - typically for navigators
                            values=[finufft.nufft3d2(current_data.KZ, current_data.KX, current_data.KY,
                                                             image) for image in df.Images]
                            kdata.append(values)

            else:

                if self.list_movements == []:
                    kdata = []

                    N1, N2, N3 = size[0], size[1], size[2]
                    M = traj.shape[1]
                    dtype = np.float32  # Datatype (real)
                    complex_dtype = np.complex64

                    for i, x in (enumerate(tqdm(zip(traj, images_series)))):


                        fk = x[1]
                        t = x[0]

                        kx = t[:,0]
                        ky = t[:,1]
                        kz = t[:,2]

                        # Initialize the plan and set the points.

                        kx = kx.astype(dtype)
                        ky = ky.astype(dtype)
                        kz = kz.astype(dtype)
                        fk = fk.astype(complex_dtype)


                        c_gpu = GPUArray((kx.shape[0]), dtype=complex_dtype)
                            # fk_gpu = GPUArray(fk.shape, dtype=complex_dtype)
                            # fk_gpu.fill(fk)
                        # kx_gpu = to_gpu(kx)
                        # ky_gpu = to_gpu(ky)
                        # kz_gpu = to_gpu(kz)

                        plan = cufinufft(2, (N1, N2, N3), 1, eps=eps, dtype=dtype)
                        plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))

                        plan.execute(c_gpu, to_gpu(fk))
                        c = np.squeeze(c_gpu.get())

                        #fk_gpu.gpudata.free()
                        c_gpu.gpudata.free()

                        plan.__del__()
                        kdata.append(c)

                        del c
                        del kx
                        del ky
                        del kz
                        del fk
                        del t
                        gc.collect()

                else:
                    kdata = []

                    N1, N2, N3 = size[0], size[1], size[2]
                    M = traj.shape[1]
                    dtype = np.float32  # Datatype (real)
                    complex_dtype = np.complex64


                    for i, x in (enumerate(tqdm(zip(traj, images_series)))):


                        nb_rep = self.paramDict["nb_rep"]
                        current_data = pd.DataFrame(x[0], columns=["KX", "KY", "KZ"])
                        current_image = x[1]
                        df = pd.DataFrame(index=range(self.paramDict["nb_rep"]), columns=["Timesteps", "Images"])
                        df["Timesteps"] = self.t[:, i].reshape(-1, 1)
                        images_for_df = list(
                            np.tile(current_image, (nb_rep, 1, 1, 1)))
                        df["Images"] = images_for_df

                        for movements in self.list_movements:
                            df = movements.apply(df,useGPU)

                        groups = current_data.groupby("KZ")

                        kdata_current=[]

                        if not (trajectory.reconstruct_each_partition):
                            for j,g in enumerate(groups):
                                name = g[0]
                                group = g[1]
                                fk = df.Images[j]
                                kx = np.array(group["KX"])
                                ky = np.array(group["KY"])
                                kz = np.array(group["KZ"])

                                # Initialize the plan and set the points.

                                kx = kx.astype(dtype)
                                ky = ky.astype(dtype)
                                kz = kz.astype(dtype)
                                fk = fk.astype(complex_dtype)

                                c_gpu = GPUArray((kx.shape[0]), dtype=complex_dtype)
                                #fk_gpu = GPUArray(fk.shape, dtype=complex_dtype)
                                #fk_gpu.fill(fk)
                                fk_gpu = to_gpu(fk)


                                plan = cufinufft(2, (N1, N2,N3), 1, eps=eps, dtype=dtype)
                                plan.set_pts(to_gpu(kz),to_gpu(kx), to_gpu(ky))
                                plan.execute(c_gpu, fk_gpu)
                                c = np.squeeze(c_gpu.get())
                                kdata_current.append(c)
                                fk_gpu.gpudata.free()
                                c_gpu.gpudata.free()
                                plan.__del__()

                            kdata_current=np.array(kdata_current).flatten()

                        else:# For navigators
                            kx = np.array(current_data.KX)
                            ky = np.array(current_data.KY)
                            kz = np.array(current_data.KZ)

                            kx = kx.astype(dtype)
                            ky = ky.astype(dtype)
                            kz = kz.astype(dtype)

                            for image in df.Images:
                                fk = image
                                fk = fk.astype(complex_dtype)
                                c_gpu = GPUArray((trajectory.paramDict["npoint"]), dtype=complex_dtype)
                                # fk_gpu = GPUArray(fk.shape, dtype=complex_dtype)
                                # fk_gpu.fill(fk)
                                fk_gpu = to_gpu(fk)

                                plan = cufinufft(2, (N1, N2, N3), 1, eps=eps, dtype=dtype)
                                plan.set_pts(to_gpu(kz), to_gpu(kx), to_gpu(ky))
                                plan.execute(c_gpu, fk_gpu)
                                c = np.squeeze(c_gpu.get())
                                kdata_current.append(c)
                                fk_gpu.gpudata.free()
                                c_gpu.gpudata.free()
                                plan.__del__()

                        kdata.append(kdata_current)

        #dtheta = np.pi / nspoke
        #kdata = np.array(kdata) / (npoint * self.paramDict["nb_rep"]) * dtheta

        # kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)
        if movement_correction:
            transf = self.list_movements[0].paramDict["transformation"]
            t = self.t
            kdata_corrected,traj_corrected,retained_timesteps = correct_mvt_kdata(kdata,traj,t,transf,perc)

            return kdata_corrected,traj_corrected,retained_timesteps

        return kdata

    def generate_kdata_multi(self,trajectory,b1_all_slices,useGPU=False,eps=1e-4):
        b1_prev = np.ones(b1_all_slices[0].shape, dtype=b1_all_slices[0].dtype)
        b1_all = np.concatenate([np.expand_dims(b1_prev, axis=0), b1_all_slices], axis=0)

        kdata = []

        # images = copy(m.images_series)

        for i in tqdm(range(1, b1_all.shape[0])):
            self.images_series *= np.expand_dims(b1_all[i] / b1_all[i - 1], axis=0)
            kdata.append(np.array(self.generate_kdata(trajectory, useGPU=useGPU,eps=eps)))

        self.images_series /= np.expand_dims(b1_all[-1], axis=0)
        # del images

        kdata = np.array(kdata)

        return kdata

    def buildROImask(self):
        return buildROImask(self.paramMap)


    # def generate_kdata(self,traj):
    #     kdata = generate_kdata(self.images_series,traj)
    #     return kdata

    def rotate_images(self,angles_t):
        # angles_t function returning rotation angle as a function of t
        angles = [angles_t(t) for t in self.t]
        self.images_series= np.array([ndimage.rotate(self.images_series[i,:,:], angles[i], reshape=False) for i in range(self.images_series.shape[0])])


    def simulate_moving_images(self):
        if "nb_total_slices" not in self.paramDict:#2D slice
            df = pd.DataFrame(index=range(self.images_series.shape[0]), columns=["Timesteps", "Images"])
            df["Timesteps"] = self.t.reshape(-1, 1)
            df["Images"] = list(self.images_series)

            for movement in self.list_movements:
                df = movement.apply(df)
            return df

        else:#3D volume
            timesteps_1rep = self.t.shape[1]
            all_timesteps = self.t.reshape(-1,1)
            df = pd.DataFrame(index=range(all_timesteps.shape[0]), columns=["Timesteps", "Images"])
            df["Timesteps"] = all_timesteps
            for i in range(self.paramDict["nb_rep"]):
                current_df = df.iloc[i*(timesteps_1rep):(i+1)*(timesteps_1rep),:]
                current_df["Images"]=list(self.images_series)
                for movement in self.list_movements:
                    current_df = movement.apply(current_df)
                df.iloc[i*(timesteps_1rep):(i+1)*(timesteps_1rep),:]=current_df



    def change_resolution(self,compression_factor=2):
        print("WARNING : Compression is irreversible")
        kept_indices=int(compression_factor)
        if len(self.image_size)==2:

            if self.images_series is not None:
                self.images_series = self.images_series[:,::kept_indices,::kept_indices]
                self.cached_images_series=self.images_series
            new_mask=self.mask[::kept_indices,::kept_indices]

            for param in self.paramMap.keys():
                values_on_mask=self.paramMap[param]
                values=makevol(values_on_mask,self.mask>0)
                values=values[::kept_indices,::kept_indices]
                new_values_on_mask=values[new_mask>0]
                self.paramMap[param]=new_values_on_mask

        else:
            if self.images_series is not None:

                self.images_series = self.images_series[:, :,::kept_indices, ::kept_indices]
                self.cached_images_series = self.images_series
            new_mask = self.mask[:,::kept_indices, ::kept_indices]

            for param in self.paramMap.keys():
                values_on_mask = self.paramMap[param]
                values = makevol(values_on_mask, self.mask > 0)
                values = values[:,::kept_indices, ::kept_indices]
                new_values_on_mask = values[new_mask > 0]
                self.paramMap[param] = new_values_on_mask

        self.mask=new_mask
        self.image_size = new_mask.shape


    def buildParamMap(self,mask=None):
        raise ValueError("should be implemented in child")

    def plotParamMap(self,key=None,figsize=(5,5),fontsize=5,save=False,sl=None):
        if len(self.image_size)==2:
            if key is None:
                keys=list(self.paramMap.keys())
                fig,axes=plt.subplots(1,len(keys),figsize=(len(keys)*figsize[0],figsize[1]))
                for i,k in enumerate(keys):
                    im=axes[i].imshow(makevol(self.paramMap[k],(self.mask>0)))
                    axes[i].set_title("{} Map {}".format(k,self.name))
                    axes[i].tick_params(axis='x', labelsize=fontsize)
                    axes[i].tick_params(axis='y', labelsize=fontsize)
                    cbar=fig.colorbar(im, ax=axes[i],fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=fontsize)
                if save:
                    plt.savefig("./figures/ParamMap_{}_all".format(self.name, key))
            else:
                fig,ax=plt.subplots(figsize=figsize)

                im=ax.imshow(makevol(self.paramMap[key],(self.mask>0)))
                ax.set_title("{} Map {}".format(key,self.name))
                ax.tick_params(axis='x', labelsize=fontsize)
                ax.tick_params(axis='y', labelsize=fontsize)
                cbar=fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=fontsize)

                if save:
                    plt.savefig("./figures/ParamMap_{}_{}".format(self.name,key))
        elif len(self.image_size)==3:
            if sl is None:
                print("WARNING : plotting the paramMap for slice 0 as sl number was not provided")
                sl=0

            if key is None:
                keys = list(self.paramMap.keys())
                fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * figsize[0], figsize[1]))
                for i, k in enumerate(keys):
                    im = axes[i].imshow(makevol(self.paramMap[k], (self.mask > 0))[sl,:,:])
                    axes[i].set_title("{} Map {} Slice {}".format(k, self.name,sl))
                    axes[i].tick_params(axis='x', labelsize=fontsize)
                    axes[i].tick_params(axis='y', labelsize=fontsize)
                    cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=fontsize)
                if save:
                    plt.savefig("./figures/ParamMap_{}_sl_{}_all".format(self.name, key,sl))
            else:
                fig, ax = plt.subplots(figsize=figsize)

                im = ax.imshow(makevol(self.paramMap[key], (self.mask > 0))[sl,:,:])
                ax.set_title("{} Map {}".format(key, self.name))
                ax.tick_params(axis='x', labelsize=fontsize)
                ax.tick_params(axis='y', labelsize=fontsize)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=fontsize)

                if save:
                    plt.savefig("./figures/ParamMap_{}_{}_sl_{}".format(self.name, key,sl))




        plt.show()

    def roundParam(self,paramName,decimals=0):
        self.paramMap[paramName]=np.round(self.paramMap[paramName],decimals=decimals)

    def reset_image_series(self):
        self.images_series=self.cached_images_series

    def compare_patterns(self,pixel_number):

        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.plot(np.real(self.cached_images_series[:,pixel_number[0],pixel_number[1]]),label="original image - real part")
        ax1.plot(np.real(self.images_series[:, pixel_number[0],pixel_number[1]]), label="transformed image - real part")
        ax1.legend()
        ax2.plot(np.imag(self.cached_images_series[:, pixel_number[0], pixel_number[1]]),
                 label="original image - imaginary part")
        ax2.plot(np.imag(self.images_series[:, pixel_number[0], pixel_number[1]]),
                 label="transformed - imaginary part")
        ax2.legend()

        ax3.plot(np.abs(self.cached_images_series[:, pixel_number[0], pixel_number[1]]),
                 label="original image - norm")
        ax3.plot(np.abs(self.images_series[:, pixel_number[0], pixel_number[1]]),
                 label="transformed - norm")
        ax3.legend()

        plt.show()


class RandomMap(ImageSeries):

    def __init__(self,name,dict_config,**kwargs):
        super().__init__(name,dict_config,**kwargs)

        self.region_size=self.paramDict["region_size"]

        if "mask_reduction_factor" not in self.paramDict:
            self.paramDict["mask_reduction_factor"] =0.0# mask_reduction_factors*total_pixels will be cropped on all edges of the image

        mask_red = self.paramDict["mask_reduction_factor"]
        mask = np.zeros(self.image_size)
        mask[int(self.image_size[0] *mask_red):int(self.image_size[0]*(1-mask_red)), int(self.image_size[1] *mask_red):int(self.image_size[1]*(1-mask_red))] = 1.0
        self.mask=mask

    @wrapper_rounding
    def buildParamMap(self,mask=None):
        #print("Building Param Map")
        if mask is None:
            mask=self.mask
        else:
            self.mask=mask

        wT1 = self.dict_config["water_T1"]
        fT1 = self.dict_config["fat_T1"]
        wT2 = self.dict_config["water_T2"]
        fT2 = self.dict_config["fat_T2"]
        att = self.dict_config["B1_att"]
        df = self.dict_config["delta_freqs"]
        df = [- value / 1000 for value in df]
        ff = self.dict_config["ff"]

        map_wT1 = create_random_map(wT1, self.region_size, self.image_size, mask)
        map_wT2 = create_random_map([wT2], self.region_size, self.image_size, mask)
        map_fT1 = create_random_map(fT1, self.region_size, self.image_size, mask)
        map_fT2 = create_random_map([fT2], self.region_size, self.image_size, mask)
        map_attB1 = create_random_map(att, self.region_size, self.image_size, mask)
        map_df = create_random_map(df, self.region_size, self.image_size, mask)
        map_ff = create_random_map(ff, self.region_size, self.image_size, mask)
        map_all = np.stack((map_wT1, map_wT2, map_fT1, map_fT2, map_attB1, map_df, map_ff), axis=-1)
        map_all_on_mask = map_all[mask > 0]


        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]

        }

    def buildROImask(self):
        mask_red = self.paramDict["mask_reduction_factor"]
        sliced_image_size = (self.image_size[0], self.image_size[0])
        num_regions_shape = (int(sliced_image_size[0] * (1 - 2 * mask_red) / self.region_size),
                             int(sliced_image_size[1] * ((1 - 2 * mask_red)) / self.region_size))
        count_regions_per_slice = np.prod(num_regions_shape)

        current_roi_num=1
        rois = np.arange(current_roi_num, current_roi_num + count_regions_per_slice).reshape(num_regions_shape)
        rois = np.repeat(np.repeat(rois, self.region_size, axis=1), self.region_size, axis=0).flatten()

        return rois



class MapFromFile(ImageSeries):

    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)



        if "file" not in self.paramDict:
            raise ValueError("file key value argument containing param map file path should be given for MapFromFile")

        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1

        if "file_type" not in self.paramDict:
            self.paramDict["file_type"]="GroundTruth" # else "Result"

    @wrapper_rounding
    def buildParamMap(self,mask=None,dico_bumps=None):

        if mask is not None:
            raise ValueError("mask automatically built from wT1 map for file load for now")

        if self.paramDict["file_type"]=="GroundTruth":
            matobj = loadmat(self.paramDict["file"])["paramMap"]
            map_wT1 = matobj["T1"][0][0]
            map_df = matobj["Df"][0, 0]
            map_attB1 = matobj["B1"][0, 0]
            map_ff = matobj["FF"][0, 0]
        elif self.paramDict["file_type"]=="Result":
            try:
                matobj = loadmat(self.paramDict["file"])["MRFmaps"]
            except:
                print("Warning : Had to read Matlab MRF Map with H5 reader")
                matobj_inter = h5py.File(self.paramDict["file"],"r").get("MRFmaps")
                matobj={}
                for k in matobj_inter.keys():
                    matobj[k]=[[np.flip(np.rot90(np.array(matobj_inter.get(k)),axes=(2,1)),axis=2)]]
            try:
                map_wT1 = matobj["T1water"][0][0]
                map_df = matobj["Df"][0][0]
                map_attB1 = matobj["B1"][0][0]
                map_ff = matobj["FF"][0][0]
            except:
                map_wT1 = matobj["T1water_map"][0][0]
                map_df = matobj["Df_map"][0][0]
                map_attB1 = matobj["FA_map"][0][0]
                map_ff = matobj["FF_map"][0][0]

            if (map_wT1.ndim==3)and(map_wT1.shape[-1]<map_wT1.shape[0]):
                map_wT1 = np.moveaxis(map_wT1,-1,0)
                map_df = np.moveaxis(map_df, -1, 0)
                map_attB1 = np.moveaxis(map_attB1, -1, 0)
                map_ff = np.moveaxis(map_ff, -1, 0)

        else:
            raise ValueError("file_type can only be GroundTruth or Result")



        self.image_size = map_wT1.shape

        mask = np.zeros(self.image_size)
        mask[map_wT1>0]=1.0
        self.mask=mask

        map_wT2 = mask * self.paramDict["default_wT2"]
        map_fT2 = mask * self.paramDict["default_fT2"]

        if self.paramDict["file_type"]=="GroundTruth":
            map_fT1 = mask*self.paramDict["default_fT1"]
        elif self.paramDict["file_type"] == "Result":
            try:
                map_fT1 = matobj["T1fat"][0][0]
            except:
                map_fT1 = matobj["T1fat_map"][0][0]

            if (map_fT1.ndim==3) and(map_fT1.shape[-1]<map_fT1.shape[0]):
                map_fT1 = np.moveaxis(map_fT1, -1, 0)
        else:
            raise ValueError("file_type can only be GroundTruth or Result")



        map_all = np.stack((map_wT1, map_wT2, map_fT1, map_fT2, map_attB1, map_df, map_ff), axis=-1)
        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": -map_all_on_mask[:, 5]/1000,
            "ff": map_all_on_mask[:, 6]
        }

        if dico_bumps is not None:
            for k in dico_bumps.keys():
                self.paramMap[k]=np.maximum(np.minimum(self.paramMap[k]+dico_bumps[k][self.mask>0],PARAMS_WINDOWS[k][1]),PARAMS_WINDOWS[k][0])
                #print(self.paramMap[k])


class MapFromDict(ImageSeries):

    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)



        if "paramMap" not in self.paramDict:
            raise ValueError("paramMap key value argument containing param map file path should be given for MapFromFile")

        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1

    @wrapper_rounding
    def buildParamMap(self):



        paramMap = self.paramDict["paramMap"]

        map_wT1=paramMap["wT1"]
        self.image_size=map_wT1.shape

        if "mask" in self.paramDict:
            mask=self.paramDict["mask"]

        else:
            mask = np.zeros(self.image_size)
            mask[map_wT1>0]=1.0

        self.mask=mask

        map_wT2 = mask*self.paramDict["default_wT2"]
        map_fT1 = mask*self.paramDict["default_fT1"]
        map_fT2 = mask*self.paramDict["default_fT2"]

        map_all = np.stack((paramMap["wT1"], map_wT2, map_fT1, map_fT2, paramMap["attB1"], paramMap["df"], paramMap["ff"]), axis=-1)
        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]
        }




class MapFromMatching(ImageSeries):
    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)
        if "search_function" not in self.paramDict:
            raise ValueError("You should define a search_function for building a map from matching with existing patterns")
        if "kdata" not in self.paramDict :
            raise ValueError(
                "You should define a kdata for building a map from matching with existing patterns")
        if "traj" not in self.paramDict :
            raise ValueError(
                "You should define a traj for building a map from matching with existing patterns")

    def buildParamMap(self,mask=None):
        pass


class ImageSeries3D(ImageSeries):

    def __init__(self, name,dict_config={}, **kwargs):
        super().__init__(name, dict_config, **kwargs)
        if "nb_slices" not in self.paramDict:
            raise ValueError("nb_slices should be provided for MapFromFile3D")
        if "gap_slice" not in self.paramDict:
            self.paramDict["gap_slice"]=1
        if "undersampling_factor" not in self.paramDict:
            self.paramDict["undersampling_factor"]=1
        if "resting_time" not in self.paramDict:
            self.paramDict["resting_time"]=0.0
        if "nb_empty_slices" not in self.paramDict:#Empty slices on both sides of the stack
            self.paramDict["nb_empty_slices"]=0

        self.paramDict["nb_total_slices"]=2*self.paramDict["nb_empty_slices"]+self.paramDict["nb_slices"]
        self.paramDict["nb_rep"]=int(self.paramDict["nb_total_slices"]/self.paramDict["undersampling_factor"])

    # def simulate_radial_undersampled_images(self, trajectory, density_adj=True):
    #     # traj_3D is a matrix that represents the k_space trajectory, of size timesteps * number of points * 3 (x-y-z coord)
    #     size = self.image_size
    #     images_series = self.images_series
    #     # images_series =normalize_image_series(self.images_series)
    #
    #     traj_3D = trajectory.get_traj()
    #     nspoke=trajectory.paramDict["nspoke"]
    #     npoint=trajectory.paramDict["npoint"]
    #
    #
    #     kdata = [
    #         finufft.nufft3d2(t[:, 2], t[:, 0], t[:, 1], p)
    #         for t, p in zip(traj_3D, images_series)
    #     ]
    #
    #     dtheta = 1 / nspoke
    #     kdata = np.array(kdata) / npoint * dtheta
    #
    #     # kdata /= np.sum(np.abs(kdata) ** 2) ** 0.5 / len(kdata)
    #
    #     if density_adj:
    #         if npoint is None:
    #             raise ValueError("Should supply number of point on spoke for density compensation")
    #         density = np.abs(np.linspace(-1, 1, npoint))
    #         kdata = [(np.reshape(k, (-1, npoint)) * density).flatten() for k in kdata]
    #
    #     # kdata = (normalize_image_series(np.array(kdata)))
    #
    #     images_series_rebuilt = [
    #         finufft.nufft3d1(t[:, 2], t[:, 0], t[:, 1], s, size)
    #         for t, s in zip(traj_3D, kdata)
    #     ]
    #
    #     # images_series_rebuilt =normalize_image_series(np.array(images_series_rebuilt))
    #
    #     return np.array(images_series_rebuilt)

    def animParamMap(self, key, figsize=(5, 5), fontsize=5, interval=200):

        nb_frames = self.mask.shape[0]
        all_images = makevol(self.paramMap[key],self.mask>0)

        fig, ax = plt.subplots()
        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in
        # each frame
        ims = []
        for i in range(nb_frames):
            image = all_images[i]
            im = ax.imshow(image, animated=True)
            if i == 0:
                ax.imshow(image)  # show an initial one first
            ims.append([im])

        return animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                         repeat_delay=10 * interval)





class MapFromFile3D(ImageSeries3D):

    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)



        if "file" not in self.paramDict:
            raise ValueError("file key value argument containing param map file path should be given for MapFromFile")



        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1

    @wrapper_rounding
    def buildParamMap(self, mask=None):

        if mask is not None:
            raise ValueError("mask automatically built from wT1 map for file load for now")

        matobj = loadmat(self.paramDict["file"])["paramMap"]
        map_wT1 = matobj["T1"][0, 0]
        map_df = matobj["Df"][0, 0]
        map_attB1 = matobj["B1"][0, 0]
        map_ff = matobj["FF"][0, 0]

        mask_slice = np.zeros(map_wT1.shape)
        mask_slice[map_wT1 > 0] = 1.0

        self.image_size = (self.paramDict["nb_slices"] + 2 * self.paramDict["nb_empty_slices"],) + map_wT1.shape

        map_wT1 = np.resize(map_wT1, self.image_size)
        map_df = np.resize(map_df, self.image_size)
        map_attB1 = np.resize(map_attB1, self.image_size)
        map_ff = np.resize(map_ff, self.image_size)

        mask = np.resize(mask_slice, self.image_size)
        mask[:self.paramDict["nb_empty_slices"], :, :] = 0
        mask[-self.paramDict["nb_empty_slices"]:, :, :] = 0
        self.mask = mask

        map_wT2 = mask * self.paramDict["default_wT2"]
        map_fT1 = mask * self.paramDict["default_fT1"]
        map_fT2 = mask * self.paramDict["default_fT2"]
        map_wT1 = mask * map_wT1
        map_df = mask * map_df
        map_attB1 = mask * map_attB1
        map_ff = mask * map_ff

        map_all = np.stack((map_wT1, map_wT2, map_fT1, map_fT2, map_attB1, map_df, map_ff), axis=-1)
        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": -map_all_on_mask[:, 5] / 1000,
            "ff": map_all_on_mask[:, 6]
        }


class MapFromDict3D(ImageSeries3D):

    def __init__(self, name, **kwargs):

        if "paramMap" not in kwargs:
            raise ValueError("paramMap key value argument containing param map file path should be given for MapFromFile")

        nb_slices = kwargs["paramMap"]["wT1"].shape[0]

        super().__init__(name, nb_slices=nb_slices, **kwargs)




        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1

    @wrapper_rounding
    def buildParamMap(self):



        paramMap = self.paramDict["paramMap"]

        map_wT1=paramMap["wT1"]
        self.image_size=map_wT1.shape

        if "mask" in self.paramDict:
            mask=self.paramDict["mask"]

        else:
            mask = np.zeros(self.image_size)
            mask[map_wT1>0]=1.0

        self.mask=mask

        map_wT2 = mask*self.paramDict["default_wT2"]
        map_fT1 = mask*self.paramDict["default_fT1"]
        map_fT2 = mask*self.paramDict["default_fT2"]

        map_all = np.stack((paramMap["wT1"], map_wT2, map_fT1, map_fT2, paramMap["attB1"], paramMap["df"], paramMap["ff"]), axis=-1)
        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]
        }



class RandomMap3D(ImageSeries3D):

    def __init__(self, name,dict_config, **kwargs):
        super().__init__(name, dict_config, **kwargs)

        self.region_size = self.paramDict["region_size"]

        if "mask_reduction_factor" not in self.paramDict:
            self.paramDict[
                "mask_reduction_factor"] = 0.0  # mask_reduction_factors*total_pixels will be cropped on all edges of the image

        if "repeat_slice" not in self.paramDict:
            self.paramDict["repeat_slice"]=1

        mask_red = self.paramDict["mask_reduction_factor"]
        self.image_size=(self.paramDict["nb_total_slices"],self.image_size[0],self.image_size[1]) #for random map the image size is an input provided by the user, need to extend in z dimension
        mask = np.zeros(self.image_size)
        mask[:,int(self.image_size[1] * mask_red):int(self.image_size[1] * (1 - mask_red)),
        int(self.image_size[2] * mask_red):int(self.image_size[2] * (1 - mask_red))] = 1.0
        if not(self.paramDict["nb_empty_slices"]==0):
            mask[:self.paramDict["nb_empty_slices"], :, :] = 0
            mask[-self.paramDict["nb_empty_slices"]:, :, :] = 0
        self.mask = mask

    @wrapper_rounding
    def buildParamMap(self, mask=None):

        # print("Building Param Map")
        if mask is None:
            mask = self.mask
        else:
            self.mask = mask

        wT1 = self.dict_config["water_T1"]
        fT1 = self.dict_config["fat_T1"]
        wT2 = self.dict_config["water_T2"]
        fT2 = self.dict_config["fat_T2"]
        att = self.dict_config["B1_att"]
        df = self.dict_config["delta_freqs"]
        df = [- value / 1000 for value in df]
        ff = self.dict_config["ff"]

        nb_slices=self.paramDict["nb_slices"]
        if not(self.paramDict["nb_empty_slices"]==0):
            mask_without_empty_slices=self.mask[self.paramDict["nb_empty_slices"]:-self.paramDict["nb_empty_slices"],:,:]
        else:
            mask_without_empty_slices=self.mask

        sliced_image_size=(self.image_size[1],self.image_size[2])

        map_all=np.zeros(self.image_size+(7,))

        repeat_slice = self.paramDict["repeat_slice"]
        params_slices_count = int(nb_slices/repeat_slice)

        for j in range(params_slices_count):
            sliced_mask = mask_without_empty_slices[j,:,:]
            map_wT1 = create_random_map(wT1, self.region_size, sliced_image_size, sliced_mask)
            map_wT2 = create_random_map([wT2], self.region_size, sliced_image_size, sliced_mask)
            map_fT1 = create_random_map(fT1, self.region_size, sliced_image_size, sliced_mask)
            map_fT2 = create_random_map([fT2], self.region_size, sliced_image_size, sliced_mask)
            map_attB1 = create_random_map(att, self.region_size, sliced_image_size, sliced_mask)
            map_df = create_random_map(df, self.region_size, sliced_image_size, sliced_mask)
            map_ff = create_random_map(ff, self.region_size, sliced_image_size, sliced_mask)

            j_current = j*repeat_slice+self.paramDict["nb_empty_slices"]
            j_next = np.minimum((j+1)*repeat_slice,nb_slices)+self.paramDict["nb_empty_slices"]

            nb_repeat_current = j_next-j_current

            slices_value = np.stack((map_wT1, map_wT2, map_fT1, map_fT2, map_attB1, map_df, map_ff), axis=-1)

            map_all[j_current:j_next,:,:,:] = np.resize(slices_value,(nb_repeat_current,)+slices_value.shape)

        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]

        }

    def buildROImask(self):

        mask=self.mask

        nb_slices = self.paramDict["nb_slices"]

        mask_red=self.paramDict["mask_reduction_factor"]
        sliced_image_size = (self.image_size[1], self.image_size[2])
        repeat_slice = self.paramDict["repeat_slice"]
        params_slices_count = int(nb_slices / repeat_slice)
        num_regions_shape=(int(sliced_image_size[0]*(1-2*mask_red) / self.region_size), int(sliced_image_size[1]*((1-2*mask_red)) / self.region_size))
        count_regions_per_slice=np.prod(num_regions_shape)

        roi_all_slices = np.zeros(self.image_size)

        current_roi_num=1
        for j in range(params_slices_count):
            rois=np.arange(current_roi_num,current_roi_num+count_regions_per_slice).reshape(num_regions_shape)
            rois=np.repeat(np.repeat(rois, self.region_size, axis=1), self.region_size, axis=0).flatten()
            #map = create_map(rois, self.region_size, sliced_mask)

            j_current = j * repeat_slice + self.paramDict["nb_empty_slices"]
            j_next = np.minimum((j + 1) * repeat_slice, nb_slices) + self.paramDict["nb_empty_slices"]
            rois_volume = makevol(rois, mask[j_current] > 0)

            nb_repeat_current = j_next - j_current
            slices_value = rois_volume

            roi_all_slices[j_current:j_next, :, :] = np.resize(slices_value, (nb_repeat_current,) + slices_value.shape)

            current_roi_num+=count_regions_per_slice

        return roi_all_slices[mask > 0]



# def task_nufft(ts):
#     global traj
#     global images_series
#     print(ts)
#     result=finufft.nufft2d2(traj[ts,:, 0], traj[ts,:, 1], images_series[ts])
#     #print("Saving")
#     np.save("./nufft_optim/nufft_{}.npy".format(ts),result)
