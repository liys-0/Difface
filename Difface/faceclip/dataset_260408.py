import numpy as np
import warnings
import os, h5py
import torch
from torch.utils.data import Dataset
from glob import glob
import random
from utils import sample
from multiprocessing import Manager
from copy import deepcopy

import csv
from typing import Dict, List, Tuple, Optional


def map_categories_to_012(X):
    """
    Map the 1-48 ATGC categories into 0, 1, 2 representing SNP dosage or alleles.
    """
    X_mapped = np.zeros_like(X, dtype=np.int32)
    # TODO: Replace this dictionary with your EXACT biological mapping rule!
    # For now, this is a placeholder that maps the 1-48 IDs into 0, 1, 2 safely.
    for i in range(1, 49):
        X_mapped[X == i] = i % 3  # Maps to 0, 1, or 2

    return X_mapped

def load_category_csv_to_ram(csv_path: str, dtype=np.int32):
    """
    Load category_ids.csv into RAM.

    Expected CSV format:
      row 0: header -> ['', snp1, snp2, ...]
      row 1: LOG10P  -> ['LOG10P', p1, p2, ...]   (optional)
      row 2+: subject rows -> [subject_id, cat1, cat2, ...]

    Returns:
      ids: list[str] length N
      X: np.ndarray (N, S)
      id2row: dict[str, int]
      snp_cols: list[str]
      log10p: np.ndarray (S,) or None
    """
    ids: List[str] = []
    rows: List[np.ndarray] = []
    log10p = None

    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        snp_cols = header[1:]

        for row in r:
            if len(row) == 0:
                continue

            row_name = row[0].strip()

            if row_name.upper() == "LOG10P":
                log10p = np.asarray(row[1:], dtype=np.float32)
                continue

            pid = row_name
            vals = np.asarray(row[1:], dtype=dtype)
            ids.append(pid)
            rows.append(vals)

    if len(rows) == 0:
        raise ValueError(f"No subject rows found in SNP csv: {csv_path}")

    X = np.stack(rows, axis=0)
    X = map_categories_to_012(X) # <-- Apply 0-2 conversion here
    id2row: Dict[str, int] = {pid: i for i, pid in enumerate(ids)}

    if log10p is not None and len(log10p) != X.shape[1]:
        raise ValueError(
            f"Length mismatch: LOG10P has {len(log10p)} values, "
            f"but SNP matrix has {X.shape[1]} columns."
        )

    return ids, X, id2row, snp_cols, log10p


def get_snp_by_id(id_name: str) -> np.ndarray:
    return X[id2row[id_name]]


def get_keypoints(bnds, keypoint_type):
    bnds = np.array(bnds)
    keypoint_types = {
        'corner_9': np.array([16, 20, 28, 24, 45, 41, 37, 46, 52]),
        'corner_11': np.array([16, 20, 28, 24, 45, 41, 37, 46, 49, 52, 54]),
    }
    if keypoint_type == 'full':
        return bnds
    if keypoint_type == 'center_5':
        bnd5 = np.zeros((5, 3))
        bnd5[0] = (bnds[36] + bnds[39]) / 2.
        bnd5[1] = (bnds[42] + bnds[45]) / 2.
        bnd5[2] = bnds[30]
        bnd5[3] = bnds[48]
        bnd5[4] = bnds[54]
        return bnd5
    if keypoint_type == 'center_10':
        bnd10 = np.zeros((10, 3))
        bnd10[0] = (bnds[36] + bnds[39]) / 2.
        bnd10[1] = bnds[27]
        bnd10[2] = (bnds[42] + bnds[45]) / 2.
        bnd10[3] = bnds[3]
        bnd10[4] = bnds[30]
        bnd10[5] = bnds[13]
        bnd10[6] = bnds[48]
        bnd10[7] = bnds[51]
        bnd10[8] = bnds[54]
        return bnd10
    else:
        return bnds[keypoint_types[keypoint_type]]


class NormalDataset(Dataset):
    def __init__(
        self,
        root_path,
        ids,
        exps,
        sample_num,
        sample_func,
        keypoint_type='full',
        id2idx=None,
        exp2idx=None,
        cache_size=10000,
        snp_csv_path="/ZZ_to_RongLiang_20251203/processed_260204/subject_snp_ATGC_category_ids_mapped.csv",
        snp_log10p_threshold=3.3
    ):
        self.root_path = root_path
        self.keypoint_type = keypoint_type

        ids_snp, self.snp_all, self.snp_mapping, self.snp_cols, self.snp_log10p = load_category_csv_to_ram(
            snp_csv_path
        )

        self.snp_log10p_threshold = snp_log10p_threshold

        if self.snp_log10p_threshold is not None:
            if self.snp_log10p is None:
                raise ValueError(
                    "snp_log10p_threshold is set, but no LOG10P row was found in the SNP csv."
                )

            self.snp_keep_mask = self.snp_log10p > float(self.snp_log10p_threshold)
            self.num_snps_before_filter = int(self.snp_keep_mask.shape[0])
            self.num_snps_after_filter = int(self.snp_keep_mask.sum())

            if self.num_snps_after_filter == 0:
                raise ValueError(
                    f"No SNPs remain after applying LOG10P threshold {self.snp_log10p_threshold}."
                )

            print(f"SNP LOG10P threshold = {self.snp_log10p_threshold}")
            print(f"SNPs before filter: {self.num_snps_before_filter}")
            print(f"SNPs after filter:  {self.num_snps_after_filter}")
        else:
            self.snp_keep_mask = None
            self.num_snps_before_filter = int(self.snp_all.shape[1])
            self.num_snps_after_filter = int(self.snp_all.shape[1])

        ids_filtered = [x for x in ids if x in ids_snp]
        ids = ids_filtered

        self.num_all_ids = len(ids)

        self.pcls = []
        for id_name in ids:
            for exp_type in exps:
                surf_pcl_path = os.path.join(root_path, '{}_icp_preprocessed_surf_pcl.npy'.format(id_name))
                if os.path.exists(surf_pcl_path):
                    self.pcls.append(surf_pcl_path)

        self.size = len(self.pcls)
        ids.sort()
        self.id_num = len(ids)
        self.exp_num = len(exps)
        self.ids, self.exps = ids, exps

        if exp2idx is None:
            self.exp2idx = {}
            for i, exp_type in enumerate(exps):
                self.exp2idx[int(exp_type)] = i
        else:
            self.exp2idx = exp2idx

        if id2idx is None:
            self.id2idx = {}
            for i, id_name in enumerate(ids):
                self.id2idx[int(id_name[3:])] = i
        else:
            self.id2idx = id2idx

        self.data = Manager().dict()
        self.cache_size = cache_size
        self.sample_num = sample_num
        self.sample_func = sample_func
        self._get_nu_bnd()

        print('FaceScape dataset initialized.\n')

    def _get_nu_bnd(self):
        bnds = {}
        for id_name in self.ids:
            bnd_array = []
            nu_bnd_file = os.path.join(self.root_path, '{}_icp_preprocessed.bnd'.format(id_name))
            with open(nu_bnd_file, 'r') as bf:
                for line in bf:
                    key_point = line.split()
                    bnd_array.append([float(key_point[0]), float(key_point[1]), float(key_point[2])])
            if bnd_array is None:
                continue
            bnd_array = get_keypoints(bnd_array, self.keypoint_type)
            bnds[str(int(id_name[3:7]))] = bnd_array
        self.nu_bnds = bnds

    def get_template_kpts(self):
        template_kpts = np.loadtxt('dataset/Facescape/FacescapeNormal.bnd')
        return torch.tensor(get_keypoints(template_kpts, self.keypoint_type)).float()

    def _load_data(self, index):
        surf_pcl_file = self.pcls[index]
        assert os.path.exists(surf_pcl_file), '{} not exist!'.format(surf_pcl_file)

        bnds = []
        with open(surf_pcl_file.replace('_surf_pcl.npy', '.bnd'), 'r') as bf:
            for line in bf:
                key_point = line.split()
                bnds.append([float(key_point[0]), float(key_point[1]), float(key_point[2])])
        bnds = get_keypoints(bnds, self.keypoint_type)

        surf_points = torch.tensor(np.load(surf_pcl_file).astype(np.float32)).float()
        surf_normals = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_surf_nor.')).astype(np.float32)).float()
        free_points = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_free_pcl.')).astype(np.float32)).float()
        free_points_grad = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_free_grd.')).astype(np.float32)).float()
        free_points_sdfs = torch.tensor(np.load(surf_pcl_file.replace('_surf_pcl.', '_free_sdf.')).astype(np.float32)).float()

        points = torch.cat([surf_points, free_points], dim=0)
        sdfs = torch.cat([torch.zeros(len(surf_points)), free_points_sdfs])
        normals = torch.cat([surf_normals, free_points_grad], dim=0)
        p_sdf_grad = torch.cat([points, sdfs.unsqueeze(1), normals], dim=1)

        sample_data = {
            'p_sdf_grad': p_sdf_grad,
            'bnd': torch.tensor(bnds).float(),
            'file': surf_pcl_file
        }

        return sample_data

    def __len__(self):
        return self.size

    def _get_item(self, index):
        sample_data = self._load_data(index)
        sdf_file = sample_data['file']

        exp_type = int(1)
        exp_idx = self.exp2idx[exp_type]

        id_name = sdf_file.split('/')[-1]
        id_name_snp = id_name[:7]
        id_name = int(id_name[3:7])
        id_idx = self.id2idx[id_name]

        p_sdf_grad = sample_data['p_sdf_grad']
        key_pts = sample_data['bnd']

        samples = self.sample_func(p_sdf_grad, self.sample_num)

        snp = self.snp_all[self.snp_mapping[id_name_snp]]

        if self.snp_keep_mask is not None:
            snp = snp[self.snp_keep_mask]

        if self.snp_log10p is not None:
            if self.snp_keep_mask is not None:
                kept_log10p = self.snp_log10p[self.snp_keep_mask]
            else:
                kept_log10p = self.snp_log10p
            kept_log10p_tensor = torch.from_numpy(kept_log10p).float()
        else:
            kept_log10p_tensor = torch.empty(0, dtype=torch.float32)

        data_dict = {
            'xyz': samples[:, :3],
            'gt_sdf': samples[:, 3],
            'grad': samples[:, 4:7],
            'exp': exp_idx,
            'id': id_idx,
            'key_pts': key_pts,
            'key_pts_nu': torch.tensor(self.nu_bnds[str(id_name)]).float(),
            'snp': torch.from_numpy(snp).to(dtype=torch.long),
            'num_all_ids': self.num_all_ids,
            'id_name': torch.tensor(id_name),
            'snp_num_before_filter': self.num_snps_before_filter,
            'snp_num_after_filter': self.num_snps_after_filter,
            'kept_log10p': kept_log10p_tensor,
        }

        return data_dict, sdf_file

    def __getitem__(self, index):
        return self._get_item(index)