import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import quintic_spline, fitting_traj_by_qs

class HighwayTrajDataProcess(Dataset):
    def __init__(self, path, t_h=30, t_f=50, d_s=2, enc_size=64, targ_enc_size=112, grid_size=(25, 5), fit_plan_traj=False, fit_plan_further_ds=1):
        # Validate path existence
        if not os.path.exists(path):
            raise RuntimeError(f"{path} not exists!!")

        # Load data from '.mat' file
        if path.endswith('.mat'):
            f = h5py.File(path, 'r')
            f_tracks = f['tracks']
            track_cols, track_rows = f_tracks.shape
            self.Data = np.transpose(f['traj'])
            self.Tracks = [
                [np.transpose(f[f_tracks[j][i]][:]) for j in range(track_cols)] for i in range(track_rows)
            ]
        else:
            raise RuntimeError("Path should end with '.mat' for file or '/' for folder")

        # Determine mask_num_type based on torch version
        self.mask_num_type = torch.bool if int(torch.__version__[0]) >= 1 and int(torch.__version__[2]) >= 2 else torch.uint8

        # Initialize class attributes
        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s
        self.enc_size = enc_size
        self.targ_enc_size = targ_enc_size
        self.hist_len = self.t_h // self.d_s + 1
        self.fut_len = self.t_f // self.d_s
        self.plan_len = self.t_f // self.d_s

        self.fit_plan_traj = fit_plan_traj
        self.further_ds_plan = fit_plan_further_ds

        self.cell_length = 8
        self.cell_width = 7
        self.grid_size = grid_size
        self.grid_cells = grid_size[0] * grid_size[1]
        self.grid_length = self.cell_length * grid_size[0]
        self.grid_width = self.cell_width * grid_size[1]

    def __len__(self):
        return len(self.Data)

    def itsDsId(self, idx):
        """Get Dataset Id for the given index."""
        return self.Data[idx, 0].astype(int)

    # ... (other methods)

    def __getitem__(self, idx):
        dsId = self.itsDsId(idx)
        centVehId = self.itsPlanVehId(idx)
        t = self.itsTime(idx)
        centGrid = self.itsCentGrid(idx)
        planGridLocs = []
        targsHists = []
        targsFuts = []
        targsLonEnc = []
        targsLatEnc = []
        nbsHists = []
        planFuts = []
        targsVehs = np.zeros(self.grid_cells)

        for id, target in enumerate(centGrid):
            if target:
                targetColumn = self.getTracksCol(dsId, target, t)
                grid = self.getGridByCol(dsId, target, targetColumn)
                targsVehs[id] = target
                targsHists.append(self.getHistory(dsId, target, target, t))
                targsFuts.append(self.getFutureByCol(dsId, target, targetColumn))
                latMan, lonMan = self.getManeuverByCol(dsId, target, targetColumn)
                lat_enc = np.zeros([3])
                lon_enc = np.zeros([2])
                lat_enc[latMan - 1] = 1
                lon_enc[lonMan - 1] = 1
                targsLatEnc.append(lat_enc)
                targsLonEnc.append(lon_enc)
                nbsHists.append([self.getHistory(dsId, i, target, t, wholePeriod=True) for i in grid])
                planGridLocs.append(np.where(grid == centVehId)[0][0])
                planFuts.append(self.getPlanFuture(dsId, centVehId, target, t))

        return planFuts, nbsHists, targsHists, targsFuts, targsLonEnc, targsLatEnc, centGrid, planGridLocs, idx

    def collate_fn(self, samples):
        # ... (unchanged)
        return nbsHist_batch, nbsMask_batch, planFut_batch, planMask_batch, \
               targsHist_batch, targsEncMask_batch, targsFut_batch, targsFutMask_batch, lat_enc_batch, lon_enc_batch, idxs
