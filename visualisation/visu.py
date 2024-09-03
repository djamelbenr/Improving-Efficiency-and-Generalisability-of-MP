import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.animation import FuncAnimation
from model import TrajPred
from data_process import itsAllGroundTruthTrajs, getSize

# Constants for visual representation
GREEN = '#0F9D58'
YELLOW = '#F4B400'
RED = '#DB4437'
BLUE = '#4285F4'
FEET2METER = 0.3048

ZORDER = {
    'lane': 0,
    'nbrVeh': 3,
    'targVeh': 4,
    'planVeh': 4,
    'histTraj': 1,
    'trueTraj': 2,
    'planTraj': 3,
    'predTraj': 3,
    'collision': 6
}

LEGEND_FONTSIZE = 'xx-large'
HIST_TIME = 16
FUT_TIME = 25


class Visualizer:
    def __init__(self, hide_elements=False):
        self.hide_elements = hide_elements
        self.num_pred = 0

    def init(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc,
             targsFut, targsFutMask, centerLoc, dsId, data_source, allNeededSizes, 
             gt_plan_traj, gt_targs_traj, gt_nbrs_traj):
        
        self.nbsHist = nbsHist
        self.nbsMask = nbsMask
        self.planFut = planFut
        self.planMask = planMask
        self.targsHist = targsHist
        self.targsEncMask = targsEncMask
        self.lat_enc = lat_enc
        self.lon_enc = lon_enc
        self.targsFut = targsFut
        self.targsFutMask = targsFutMask
        self.centerLoc = centerLoc
        self.dsId = dsId
        self.dataSrc = data_source
        self.allNeededSizes = allNeededSizes
        self.gt_plan_traj = gt_plan_traj
        self.gt_targs_traj = gt_targs_traj
        self.gt_nbrs_traj = gt_nbrs_traj
        self.num_pred = len(gt_targs_traj[0])

    def get_top_x(self, lat_pred, lon_pred, topx): 
        jointP = np.einsum('ij,ik->ijk', lat_pred, lon_pred).reshape(lat_pred.shape[0], -1)
        jointP /= np.sum(jointP, axis=1, keepdims=True)
        return np.argsort(jointP, axis=1)[:, -topx:], jointP

    def set_marker_size(self, dataSrc):
        self.markersize = 8 if dataSrc == 'ngsim' else 8

    def set_nbs_idx(self, nbsMask):
        self.nbsIdx = np.sum(nbsMask[:, :, :, 0].reshape(nbsMask.shape[0], -1), axis=1)
        return self.nbsIdx

    def hide_element(self, ax):
        for spine in ax.spines.values():
            spine.set_visible(False)

    def colorline(self, x, y, ax, z=None, cmap=plt.get_cmap('rainbow'), norm=plt.Normalize(0.0, 1.0), linewidth=4, alpha=0.9, zorder=3):
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))
        z = np.asarray(z)
        segments = self.make_segments(x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, zorder=zorder, linewidth=linewidth, alpha=alpha)
        ax.add_collection(lc)
        return lc

    def make_segments(self, x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        return np.concatenate([points[:-1], points[1:]], axis=1)

    def plot_hist_traj(self, ax):
        # Plot neighbour vehicles' history
        for i, traj in enumerate(self.gt_nbrs_traj[0]):
            hist_x = traj[:, 1] * FEET2METER
            hist_y = traj[:, 0] * FEET2METER
            width, height = self.allNeededSizes[2][i] * FEET2METER
            ax.plot(hist_x, hist_y, color='grey', linestyle='solid', linewidth=3, zorder=ZORDER['histTraj'])
            ax.add_patch(patches.Rectangle((hist_x[-1] - width / 2, hist_y[-1] - height / 2), width, height,
                                           facecolor='none', edgecolor='k', linewidth=3, zorder=ZORDER['nbrVeh']))
        # Plot target vehicles' history
        for i, traj in enumerate(self.gt_targs_traj[0]):
            hist_x = traj[:, 1] * FEET2METER
            hist_y = traj[:, 0] * FEET2METER
            width, height = self.allNeededSizes[1][i] * FEET2METER
            ax.plot(hist_x, hist_y, color='grey', linestyle='solid', linewidth=3, zorder=ZORDER['histTraj'])
            ax.add_patch(patches.Rectangle((hist_x[-1] - width / 2, hist_y[-1] - height / 2), width, height,
                                           facecolor=BLUE, edgecolor='k', zorder=ZORDER['targVeh']))
        # Plot planned trajectory
        hist_x = self.gt_plan_traj[0][:, 1] * FEET2METER
        hist_y = self.gt_plan_traj[0][:, 0] * FEET2METER
        width, height = self.allNeededSizes[0][0] * FEET2METER
        ax.add_patch(patches.Rectangle((hist_x[-1] - width / 2, hist_y[-1] - height / 2), width, height,
                                       facecolor=RED, edgecolor='k', zorder=ZORDER['planVeh']))

    def plot_pred_traj(self, ax, pred_traj, lat_pred, lon_pred, top_x=1):
        idx, jointP = self.get_top_x(lat_pred, lon_pred, top_x)
        for i in range(self.num_pred):
            for j in range(top_x):
                k = idx[i, j]
                self.colorline(
                    np.concatenate((pred_traj[k, self.gt_targs_traj[1][i].shape[0]::-1, i, 1], self.gt_targs_traj[0][i][-1:, 1])) * FEET2METER,
                    np.concatenate((pred_traj[k, self.gt_targs_traj[1][i].shape[0]::-1, i, 0], self.gt_targs_traj[0][i][-1:, 0])) * FEET2METER,
                    ax, zorder=ZORDER['predTraj'])

    def plot_with_plan(self, genPlanFut, comp_pred_infos, comp_lat_preds, comp_lon_preds):
        num_gen = genPlanFut.shape[0]
        fig, axs = plt.subplots(num_gen, 1, figsize=(20, 3*num_gen), constrained_layout=True)
        if num_gen == 1:
            axs = [axs]
        
        for ax in axs:
            ax.set_aspect(aspect=2)
            if self.hide_elements:
                self.hide_element(ax)

        comp_pred_trajs_list = []
        gen_plan_trajs_list = []
        comp_lat_pred_list = []
        comp_lon_pred_list = []
        
        for j in range(num_gen):
            comp_pred_info, comp_pred_traj, centerLoc, comp_lat_pred, comp_lon_pred, gen_plan_traj = \
                self.preprocess(comp_pred_infos[j], self.centerLoc, comp_lat_preds[j], comp_lon_preds[j], genPlanFut[j])
            comp_pred_trajs_list.append(comp_pred_traj)
            gen_plan_trajs_list.append(gen_plan_traj)
            comp_lat_pred_list.append(comp_lat_pred)
            comp_lon_pred_list.append(comp_lon_pred)

        for ax in axs:
            self.plot_hist_traj(ax)

        for i, ax in enumerate(axs):
            # Plan trajectory (red line)
            plan_trajs = gen_plan_trajs_list[i]
            self.colorline(
                np.concatenate((plan_trajs[:, 0, 1], self.gt_plan_traj[0][-1:, 1])) * FEET2METER,
                np.concatenate((plan_trajs[:, 0, 0], self.gt_plan_traj[0][-1:, 0])) * FEET2METER,
                ax, zorder=ZORDER['planTraj'])

            # Predicted trajectories
            self.plot_pred_traj(ax, comp_pred_trajs_list[i], comp_lat_pred_list[i], comp_lon_pred_list[i])

            # Plot lanes
            min_x = min(
                [t[0, 1] for t in self.gt_targs_traj[0]] +
                [t[0, 1] for t in self.gt_nbrs_traj[0]] +
                [self.gt_plan_traj[0][0, 1]]
            )
            max_x = max(
                [t[-1, 1] for t in self.gt_targs_traj[0] if len(t)] +
                [t[-1, 1] for t in self.gt_plan_traj] +
                [traj[..., 1].max() for traj in comp_pred_trajs_list]
            )
            self.set_lanes(ax, self.dsId, self.dataSrc, min_x, max_x, line_color='silver', zorder=ZORDER['lane'])

        plt.close(fig)
        return fig

    def preprocess(self, comp_pred_info, centerLoc, comp_lat_pred, comp_lon_pred, gen_plan_traj):
        # Preprocess predictions and necessary data
        # Add detailed preprocessing logic here
        # Returning placeholders for now
        return comp_pred_info, comp_pred_info, centerLoc, comp_lat_pred, comp_lon_pred, gen_plan_traj

    def set_lanes(self, ax, dsId, dataSrc, min_x, max_x, line_color='silver', zorder=ZORDER['lane']):
        # Set lanes on the plot
        # This function can be implemented to draw lane lines based on dataset information
        pass

