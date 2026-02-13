import numpy as np
import math
import finufft
from tqdm import tqdm
from .utils_mrf import groupby

class Trajectory(object):

    def __init__(self,applied_timesteps=None,**kwargs):
        self.paramDict=kwargs
        self.traj = None
        self.traj_for_reconstruction=None
        self.applied_timesteps=applied_timesteps
        self.reconstruct_each_partition = False #For 3D - whether all reps are used for generating the kspace data or only the current partition


    def get_traj(self):
        #Returns and stores the trajectory array of ntimesteps * total number of points * ndim
        raise ValueError("get_traj should be implemented in child")

    def get_traj_for_reconstruction(self,timesteps=175):
        if self.traj_for_reconstruction is not None:
            print("Warning : Outputting the stored reconstruction traj - timesteps input has no impact - please reset with self.traj_for_reconstruction=None")
            return self.traj_for_reconstruction

        else:
            traj = self.get_traj()
            return traj.reshape(timesteps,-1,traj.shape[-1])


    def adjust_traj_for_window(self,window):
        traj=self.get_traj()
        traj_shape=traj.shape
        traj=np.array(groupby(traj,window))
        traj=traj.reshape((-1,)+traj_shape[1:])
        self.traj=traj

class Radial(Trajectory):

    def __init__(self,total_nspokes=1400,npoint=512,**kwargs):
        super().__init__(**kwargs)

        self.paramDict["total_nspokes"]=total_nspokes #total nspokes per rep
        self.paramDict["npoint"] = npoint
        self.paramDict["nb_rep"]=1

    def get_traj(self):
        if self.traj is None:
            npoint = self.paramDict["npoint"]
            total_nspokes = self.paramDict["total_nspokes"]
            all_spokes = radial_golden_angle_traj(total_nspokes, npoint)
            #traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))
            traj = all_spokes
            traj=np.stack([traj.real,traj.imag],axis=-1)
            self.traj=traj

        return self.traj


class Radial3D(Trajectory):

    def __init__(self,total_nspokes=1400,nspoke_per_z_encoding=8,npoint=512,undersampling_factor=1,incoherent=False,is_random=False,mode="old",offset=0,golden_angle=True,nb_rep_center_part=1,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = total_nspokes
        self.paramDict["nspoke"] = nspoke_per_z_encoding
        self.paramDict["npoint"] = npoint
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"]=math.ceil(self.paramDict["nb_slices"]/self.paramDict["undersampling_factor"])
        print(self.paramDict["nb_rep"])
        self.paramDict["random"]=is_random
        self.paramDict["incoherent"]=incoherent
        self.paramDict["mode"] = mode
        if self.paramDict["mode"]=="Kushball":
            self.paramDict["incoherent"]=True
        
        self.paramDict["offset"] = offset
        self.paramDict["golden_angle"]=golden_angle
        self.paramDict["nb_rep_center_part"] = nb_rep_center_part


    def get_traj(self):
        if self.traj is None:
            nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]
            mode = self.paramDict["mode"]
            offset=self.paramDict["offset"]

            total_nspokes = self.paramDict["total_nspokes"]
            nb_slices=self.paramDict["nb_slices"]
            undersampling_factor=self.paramDict["undersampling_factor"]

            nb_rep_center_part=self.paramDict["nb_rep_center_part"]
            print(self.paramDict["mode"])

            if self.paramDict["golden_angle"]:
                if self.paramDict["mode"]=="Kushball":
                    self.traj=self.traj=spherical_golden_angle_means_traj_3D(total_nspokes, npoint, nb_slices,undersampling_factor)

                
                else:
                    if self.paramDict["incoherent"]:
                        self.traj=radial_golden_angle_traj_3D_incoherent(total_nspokes, npoint, nspoke, nb_slices, undersampling_factor,mode,offset)
                    else:
                        self.traj = radial_golden_angle_traj_3D(total_nspokes, npoint, nspoke, nb_slices,
                                                                undersampling_factor,nb_rep_center_part)

            else:
                self.traj=distrib_angle_traj_3D(total_nspokes, npoint, nspoke, nb_slices,
                                                                undersampling_factor)


        return self.traj



class Navigator3D(Trajectory):

    def __init__(self,direction=[0.0,0.0,1.0],npoint=512,nb_slices=1,undersampling_factor=1,nb_gating_spokes=50,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = nb_gating_spokes
        self.paramDict["npoint"] = npoint
        self.paramDict["direction"] = direction
        self.paramDict["nb_slices"] = nb_slices
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"] = int(self.paramDict["nb_slices"] / self.paramDict["undersampling_factor"])
        self.reconstruct_each_partition=True

    def get_traj(self):
        if self.traj is None:
            npoint = self.paramDict["npoint"]
            direction=self.paramDict["direction"]
            #nb_rep=self.paramDict["nb_rep"]
            total_nspoke=self.paramDict["total_nspokes"]
            k_max=np.pi

            base_spoke=(-k_max+np.arange(npoint)*2*k_max/(npoint-1)).reshape(-1,1)*np.array(direction).reshape(1,-1)
            self.traj=np.repeat(np.expand_dims(base_spoke,axis=0),axis=0,repeats=total_nspoke)


        return self.traj



def radial_golden_angle_traj(total_nspoke,npoint,k_max=np.pi):
    golden_angle=111.246*np.pi/180
    base_spoke = (-k_max+k_max/(npoint)+np.arange(npoint)*2*k_max/(npoint))
    all_rotations = np.exp(1j * np.arange(total_nspoke) * golden_angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes

def distrib_angle_traj(total_nspoke,npoint,k_max=np.pi):
    angle=2*np.pi/total_nspoke
    base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    all_rotations = np.exp(1j * np.arange(total_nspoke) * angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes


def radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,nb_rep_center_part=1):
    timesteps = int(total_nspoke / nspoke)

    nb_rep = math.ceil((nb_slices ) / undersampling_factor)+nb_rep_center_part-1
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)

    k_z = np.zeros((timesteps, nb_rep))
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    

    k_z[0, :] = all_slices[::undersampling_factor]


    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    if nb_rep_center_part>1:
        center_part=all_slices[int(nb_slices/2)]
        k_z_new= np.zeros((timesteps, nb_rep))
        for j in range( k_z.shape[0]):
            num_center_part=np.argwhere(k_z[j]==center_part)[0][0]
            k_z_new[j,:num_center_part]=k_z[j,:num_center_part]
            k_z_new[j,(num_center_part+nb_rep_center_part):]=k_z[j,(num_center_part+1):]
        print(k_z_new[0,:])
        k_z=k_z_new


    k_z=np.repeat(k_z, nspoke, axis=0)

    k_z = np.expand_dims(k_z, axis=-1)


    traj = np.expand_dims(all_spokes, axis=-2)

    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])


def distrib_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = distrib_angle_traj(total_nspoke, npoint)

    k_z = np.zeros((timesteps, nb_rep))
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]

    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    traj = np.expand_dims(all_spokes, axis=-2)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def spherical_golden_angle_means_traj_3D(total_nspoke, npoint, npart, undersampling_factor=4,k_max=np.pi):
    phi1=0.46557123
    phi2=0.6823278

    theta=2*np.pi*np.mod(np.arange(total_nspoke*npart)*phi2,1)
    phi=np.arccos(np.mod(np.arange(total_nspoke*npart)*phi1,1))

    rotation=np.stack([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)],axis=1).reshape(-1,1,3)
    base_spoke = (-k_max+k_max/(npoint)+np.arange(npoint)*2*k_max/(npoint))

    base_spoke=base_spoke.reshape(-1,1)
    spokes=np.matmul(base_spoke,rotation).reshape(npart,total_nspoke,npoint,-1)
    spokes=np.moveaxis(spokes,0,1)
    return spokes.reshape(total_nspoke,-1,3)



def radial_golden_angle_traj_3D_incoherent(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=1,mode="old",offset=0):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = math.ceil(nb_slices / undersampling_factor)

    golden_angle = 111.246 * np.pi / 180
    all_slices = np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)

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


    if undersampling_factor>1:
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




def simulate_nav_images_multi(kdata, trajectory, image_size=(400,), b1=None):
    traj = trajectory.get_traj()
    nb_channels = kdata.shape[0]
    npoint = kdata.shape[-1]
    nb_slices = kdata.shape[1]
    nb_gating_spokes = kdata.shape[2]
    npoint_image=image_size[0]

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

    for i in tqdm(range(nb_channels)):
        fk = finufft.nufft1d1(traj[0, :, 2], kdata[i, :, :], image_size)
        fk = fk.reshape((nb_slices, nb_gating_spokes, npoint_image))


        if b1 is None:
            images_series_rebuilt_nav += np.abs(fk) ** 2
        else:
            images_series_rebuilt_nav += b1[i].conj() * fk

    if b1 is None:
        images_series_rebuilt_nav = np.sqrt(images_series_rebuilt_nav)

    return images_series_rebuilt_nav
