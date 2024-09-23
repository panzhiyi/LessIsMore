from curses import keyname
import numpy as np
import torch
import os
import os.path as osp
import ssl
import sys
import urllib
import h5py
from typing import Optional
from IPython.core.debugger import set_trace


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
    # # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]


# download
def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder. 
    Borrowed from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/download.py
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
        np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, voxel_idx, count = np.unique(key_sort, return_counts=True, return_inverse=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[
                               0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, voxel_idx, count
    
def voxelize_LaS(coord, mask, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    mask = mask.astype(np.int16)
    idx_mask_sort = np.argsort(-mask)
    mask_ = mask[idx_mask_sort]
    key_ = key[idx_mask_sort]
    idx_sort = np.argsort(key_, kind='mergesort')
    key_sort = key_[idx_sort]
    mask_sort = mask_[idx_sort]
    idx_sort = idx_mask_sort[idx_sort]
    _, voxel_idx, count = np.unique(key_sort, return_counts=True, return_inverse=True)
    if mode == 0:  # train mode
        num_label = np.cumsum(mask_sort)[np.cumsum(count) - 1] - np.insert(np.cumsum(mask_sort)[np.cumsum(count) - 1],
                                                                           0, 0)[0:-1]
        num_label[num_label == 0] = count[num_label == 0]
        idx_select = np.cumsum(np.insert(count, 0, 0)[
                               0:-1]) + np.random.randint(0, num_label.max(), num_label.size) % num_label
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, voxel_idx, count

def crop_pc(coord, feat, label, split='train',
            voxel_size=0.04, voxel_max=None,
            downsample=True, variable=True, shuffle=True, mask=None, label_aware=False):
    if voxel_size and downsample:
        # Is this shifting a must? I borrow it from Stratified Transformer and Point Transformer. 
        coord -= coord.min(0)
        if mask is None or not label_aware:
            uniq_idx = voxelize(coord, voxel_size)
        else:
            uniq_idx = voxelize_LaS(coord, mask, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx] if feat is not None else None, label[uniq_idx] if label is not None else None
    if voxel_max is not None:
        crop_idx = None
        N = len(label)  # the number of points
        if N >= voxel_max:
            init_idx = np.random.randint(N) if 'train' in split else N // 2
            crop_idx = np.argsort(
                np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        elif not variable:
            # fill more points for non-variable case (batched data)
            cur_num_points = N
            query_inds = np.arange(cur_num_points)
            padding_choice = np.random.choice(
                cur_num_points, voxel_max - cur_num_points)
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])
        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx
        if shuffle:
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]
        coord, feat, label = coord[crop_idx], feat[crop_idx] if feat is not None else None, label[crop_idx] if label is not None else None
    coord -= coord.min(0) 
    return coord.astype(np.float32), feat.astype(np.float32) if feat is not None else None , label if label is not None else None

def pick_pc(coord, feat, num_picked=1000, theta_space=0.75, theta_color=0.04, voxel_max=None):
    similar_space = coord * coord.transpose()
    similar_color = feat * np.transpose()
    similar_space = similar_space < theta_space
    similar_color = similar_color < theta_color
    mask = similar_space ^ similar_color
    counts = np.sum(mask_pick, axis=0)
    idx = np.random.randint(0,voxel_max,voxel_max*num_picked).reshape((num_picked,voxel_max))%counts
    mask_idx = np.where(mask==True)
    idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1]) + idx
    handpicked = mask_idx[idx_select.reshape(-1)].reshape(num_picked,voxel_max)
    return handpicked

def get_features_by_keys(data, keys='pos,x'):
    key_list = keys.split(',')
    if len(key_list) == 1:
        return data[keys].transpose(1,2).contiguous()
    else:
        return torch.cat([data[key] for key in keys.split(',')], -1).transpose(1,2).contiguous()


def get_class_weights(num_per_class, normalize=False):
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)

    if normalize:
        ce_label_weight = (ce_label_weight *
                           len(ce_label_weight)) / ce_label_weight.sum()
    return torch.from_numpy(ce_label_weight.astype(np.float32))

def rgb2lab_matrix(colors):
    n=len(colors)
    colors=np.array(colors)
    colors=colors.astype('float')
    RGBs=colors/255.0 
    Xs=RGBs[:,0]*0.4124+RGBs[:,1]*0.3576+RGBs[:,2]*0.1805
    Ys=RGBs[:,0]*0.2126+RGBs[:,1]*0.7152+RGBs[:,2]*0.0722
    Zs=RGBs[:,0]*0.0193+RGBs[:,1]*0.1192+RGBs[:,2]*0.9505
    XYZs=np.vstack((Xs,Ys,Zs)).transpose()  
    
    XYZs[:,0]=XYZs[:,0]/(95.045/100.0)
    XYZs[:,1]=XYZs[:,1]/(100.0/100.0)
    XYZs[:,2]=XYZs[:,2]/(108.875/100.0)
    L=np.zeros((n,3),dtype='float')
    for i in range(0,3):
        v=XYZs[:,i]
        vv=np.where(v>0.008856,v**(1.0/3),v*7.787+16.0/116)
        L[:,i]=np.where(v>0.008856,116.0*vv-16.0,v*903.3)
        XYZs[:,i]=vv

    As=500.0*(XYZs[:,0]-XYZs[:,1])/128.0
    Bs=200.0*(XYZs[:,1]-XYZs[:,2])/128.0
    Ls=L[:,1]/100.0
    LABs=np.vstack((Ls,As,Bs)).transpose()
    LABs=LABs.astype('float')
    return LABs

def lab2rgb_matrix(colors):
    n=len(colors)
    colors=np.array(colors)
    Ls=colors[:,0]
    As=colors[:,1]
    Bs=colors[:,2]
    T1=0.008856
    T2=0.206893
    d=T2
    fys=((Ls+16)/116.0)**3.0
    fxs=fys+As/500.0
    fzs=fys-Bs/200.0
    Xs=np.zeros((n),dtype='float')
    Ys=np.zeros((n),dtype='float')
    Zs=np.zeros((n),dtype='float')

    fys=np.where(fys>T1,fys,Ls/903.3)
    Ys=fys
    fys=np.where(fys>T1,fys**(1.0/3),fys*7.787+16.0/116)

    fxs=fys+As/500.0
    Xs=np.where(fxs>T2,fxs**3.0,(fxs-16.0/116)/7.787)

    fzs=fys-Bs/200.0
    Zs=np.where(fzs>T2,fzs**3.0,(fzs-16.0/116)/7.787) 

    Xs*=0.95045
    Zs*=1.08875
    Rs = 3.240479 * Xs + (-1.537150) * Ys + (-0.498535) * Zs
    Gs = (-0.969256) * Xs + 1.875992 * Ys + 0.041556 * Zs
    Bs = 0.055648 * Xs + (-0.204043) * Ys + 1.057311 * Zs
    RGBs=np.vstack((Rs,Gs,Bs)).transpose() 
    RGBs=np.maximum(RGBs*255,0.0)
    RGBs=np.minimum(RGBs,255.0)
    RGBs=RGBs.astype('int')
    return RGBs

def rgb2hsv(rgb):
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdims=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1]).float()
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).float(), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

def hsv2rgb(hsv):
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb