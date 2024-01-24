import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
import time
# here is my neural model ...
from model import TrajPred

# import functions 
from data_process import itsAllGroundTruthTrajs, getSize



## NGSIM and HighD datasets are post-processed in this script ...
'''
% Data: # row= data number, # column=138 (13+grid_num)
0: Dataset Id
1: Vehicle Id
2: Frame Id
3: Local X
4: Local Y
5: lane Id
6: lateral manoeuvre
7: Longetidunal manoeuvre
8: Length 
9: Width 
10: Class label 
11: Velocity 
12: Acceleration
13: 137: Neighbour Car Id at grid location  
'''
'''
% Tracks: cells: {Dataset_Id * Vehicle_Id, }
0: Frame Id
1: Local X
2: Local Y
3: Lane Id
4: lateral manoeuvre 
5: Longitudinal manoeuvre 
6: Length
7: Width 
8: Class Label 
9: Velocity 
10: Acceleration 
11-135: Neighbour Car Ids at grid Location ...
'''
GREEN = '#0F9D58'
YELLOW = '#F4B400'
RED = '#DB4437'
BLUE = '#4285F4'

ZORDER = {'lane':0,
          'nbrVeh':3,
          'targVeh':4,
          'planVeh':4,
          'histTraj':1,
          'trueTraj':2,
          'planTraj':3,
          'predTraj':3,
          'collision':6}

LEGEND_FONTSIZE = 'xx-large'
HIST_TIME = 16
FUT_TIME = 25
FEET2METER = 0.3048


## Return the most probable X maneuvers
def getTopX(self, lat_pred, lon_pred, topx): 
    jointP = np.zeros((lat_pred.shape[0], lon_pred.shape[1] * lat_pred.shape[1]))
    for k in range(lat_pred.shape[0]):
        for i in range(lon_pred.shape[1]):
            for j in range(lat_pred.shape[1]):
                jointP[k,3*i+j] = (lat_pred[k][j] * lon_pred[k][i])
        jointP[k] = jointP[k] / np.sum(jointP[k])
    return np.argsort(jointP,axis=1)[:,-topx:], jointP


class Visual(object):
    def __init__(self):
        pass
    #return nbsHist_batch, nbsMask_batch, \
    #                planFut_batch, planMask_batch, \
    #                targsHist_batch, targsEncMask_batch, \
    #                targsFut_batch, targsFutMask_batch, lat_enc_batch, lon_enc_batch, idxs
    def init(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc,
             targsFut, targsFutMask, \
             centerLoc, dsId, data_source, \
             allNeededSizes, \
             gt_plan_traj, gt_targs_traj, gt_nbrs_traj):
    '''
    nbsHist: neighbouring veh history tracks 
    nbsMask: neighbouring veh history tracks (mask) 
    planFut: plan tracks future (ego-veh)
    planMask: plan tracks mask (ego-veh)
    targsHist: target veh history tracks 
    targsEncMask: target veh history encoding mask 
    lat_enc: encoded lateral manoeuvre encoded 
    lon_enc: longitudinal lateral maneovure encoded 

    centerLoc: get the centre location of each veh
    
    dsId:      data ID 
    data_source: 

    allNeededSizes: get all the veh size  

    def getSize(self, dsId, vehId):
        length_width = self.Tracks[dsId - 1][vehId - 1][6:8, 0]
        return length_width

    ## to get the ground truth, function below is useful : 

    def itsAllGroundTruthTrajs(self, idx):
        return [self.absPlanTraj(idx), self.absTargsTraj(idx), self.absNbrsTraj(idx)]

    gt_plan_traj: ground truth planning trajectory 
    gt_targs_traj: ground truth target veh trajectory 
    gt_nbrs_traj: ground truth neighbouring veh trajectory 
    '''
        
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
        
        self.centerLoc = centerLoc  #np.array([t[-1] for t in self.gt_targs_traj[0]])
        
        
        self.dsId = dsId                      # vehicles' ID ...
        self.dataSrc = data_source            # data SOUCE ...
        self.allNeededSizes = allNeededSizes  # vehicle sizes... 
        
        self.gt_plan_traj = gt_plan_traj
        self.gt_plan_traj[1] = [self.gt_plan_traj[1]]
        self.gt_targs_traj = gt_targs_traj
        self.gt_nbrs_traj = gt_nbrs_traj

        self.numPred = len(self.gt_targs_traj[0])
        
        self.hideElements = False

    ## Return the most probable X maneuvers
    def getTopX(self, lat_pred, lon_pred, topx): 
        jointP = np.zeros((lat_pred.shape[0], lon_pred.shape[1] * lat_pred.shape[1]))
        for k in range(lat_pred.shape[0]):
            for i in range(lon_pred.shape[1]):
                for j in range(lat_pred.shape[1]):
                    jointP[k,3*i+j] = (lat_pred[k][j] * lon_pred[k][i])
            jointP[k] = jointP[k] / np.sum(jointP[k])
        return np.argsort(jointP,axis=1)[:,-topx:], jointP


    def setMarkerSize(self, dataSrc):
        self.dataSrc = dataSrc
        self.markersize = 8 if dataSrc=='ngsim' else 8
    
    # set number of index
    def setNbsIdx(self, nbsMask):
        nbsIdx = []
        for j in range(nbsMask.shape[0]):
            nbsIdx.append(sum(nbsMask[j, :, :, 0].reshape(-1)))
        self.nbsIdx = nbsIdx
        return nbsIdx

    def hide_element(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


    def colorline(self, x, y, ax,
                  z=None, cmap=plt.get_cmap('rainbow'), norm=plt.Normalize(0.0, 1.0), linewidth=4, alpha=0.9, zorder=3):

        """
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        """

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))
        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])
        z = np.asarray(z)

        segments = self.make_segments(x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, zorder= zorder,
                                  linewidth=linewidth, alpha=alpha)
        ax.add_collection(lc)
        return lc


    def make_segments(self, x, y):

        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments
    
    # might be added afterwards ...
    #def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    #    if ax is None: 
    #        ax = plt.gca() 
    #    ax.xaxis.set_visible(False) 
    #    ax.yaxis.set_visible(False) 
    #    for spine in ax.spines.itervalues(): 
    #        spine.set_visible(False) 


    def plotPredTraj(self, ax, pred_traj, lat_pred, lon_pred, top_x=1, color=False):
        idx, jointP = self.getTopX(lat_pred, lon_pred, top_x)
        for i in range(self.numPred):  # number of vehicles
            for j in range(top_x):  # number of prediction results
                k = idx[i, j]
                if color:
                    self.colorline(np.concatenate((pred_traj[k, self.gt_targs_traj[1][i].shape[0]::-1, i, 1],
                                                   self.gt_targs_traj[0][i][-1:, 1]), 0) * FEET2METER,
                                   np.concatenate((pred_traj[k, self.gt_targs_traj[1][i].shape[0]::-1, i, 0],
                                                   self.gt_targs_traj[0][i][-1:, 0]), 0) * FEET2METER,
                                   ax, zorder=ZORDER['predTraj'])

    #:                +------------------+
    #:                |         |        |
    #:              height --------------|---->
    #:                |         |        |
    #:               (xy)---- width -----+

    
    def plotHistTraj(self, ax):
        # Plot the vehicles' history
        ## Neighbours
        for i in range(len(gt_nbrs_traj_[0])):
            hist_x = gt_nbrs_traj_[:, :, 1] * FEET2METER
            hist_y = gt_nbrs_traj_[:, :, 0] * FEET2METER
                
            #width, height = self.allNeededSizes[2][i] * FEET2METER
            ax.plot(hist_x, hist_y, color='grey', linestyle='solid', linewidth=3, zorder=ZORDER['histTraj'])
                
            ax.add_patch(patches.Rectangle((hist_x[0][-1], hist_y[0][-1] ), 15, 5,
                                              facecolor='none', edgecolor='k', linewidth=3,
                                              zorder=ZORDER['nbrVeh']))
                
    def plotHistTraj(self, ax):
        # Plot the vehicles' history
        ## Neighbours
        if not self.hideTargNbrs:
            for i in range(len(self.gt_nbrs_traj[0])):
                hist_x = self.gt_nbrs_traj[0][i][:, 1] * FEET2METER
                hist_y = self.gt_nbrs_traj[0][i][:, 0] * FEET2METER
                
                width, height = self.allNeededSizes[2][i] * FEET2METER
                ax.plot(hist_x, hist_y, color='grey', linestyle='solid', linewidth=3, zorder=ZORDER['histTraj'])
                
                ax.add_patch(patches.Rectangle((hist_x[-1] - width / 2, hist_y[-1] - height / 2), width, height,
                                               facecolor='none', edgecolor='k', linewidth=3,
                                               zorder=ZORDER['nbrVeh']))
        ## targets
        for i in range(len(self.gt_targs_traj[0])):
            hist_x = self.gt_targs_traj[0][i][:, 1] * FEET2METER
            hist_y = self.gt_targs_traj[0][i][:, 0] * FEET2METER
            
            width, height = self.allNeededSizes[1][i] * FEET2METER
            
            ax.plot(hist_x, hist_y, color='grey', linestyle='solid', linewidth=3, zorder=ZORDER['histTraj'])
            ax.add_patch(patches.Rectangle((hist_x[-1] - width / 2, hist_y[-1] - height / 2), width, height,
                                           facecolor=BLUE, edgecolor='k', zorder=ZORDER['targVeh']))
        ## plan
        hist_x = self.gt_plan_traj[0][:, 1] * FEET2METER
        hist_y = self.gt_plan_traj[0][:, 0] * FEET2METER
        width, height = self.allNeededSizes[0][0] * FEET2METER
        ax.add_patch(patches.Rectangle((hist_x[-1] - width / 2, hist_y[-1] - height / 2), width, height,
                                       facecolor=RED, edgecolor='k', zorder=ZORDER['planVeh']))






#0o0o0o0o0o0o0o0o0o0o0 let's get the shit plot with plane prediction ...

    def plot_with_plan(self, genPlanFut, comp_pred_infos, comp_lat_preds, comp_lon_preds):
        # set figure and axes
        num_gen = genPlanFut.shape[0]
        figSize = (20, 3*num_gen)
        fig = plt.figure(figsize=figSize)
        axs = []
        for j in range(num_gen):
            sequence = 100 * (num_gen+1) + 10 + j + 1
            ax = fig.add_subplot(sequence)
            ax.set_aspect(aspect=2)
            axs.append(ax)
            if self.hideElements:
                self.hide_element(ax)

        comp_pred_infos_list = []
        comp_pred_trajs_list = []
        gen_plan_trajs_list = []
        comp_lat_pred_list = []
        comp_lon_pred_list = []
        
        for j in range(num_gen):
            comp_pred_info, comp_pred_traj, centerLoc, comp_lat_pred, comp_lon_pred, gen_plan_traj= \
                self.preprocess(comp_pred_infos[j], self.centerLoc, comp_lat_preds[j], comp_lon_preds[j], genPlanFut[j])
            comp_pred_infos_list.append(comp_pred_info)
            comp_pred_trajs_list.append(comp_pred_traj)
            gen_plan_trajs_list.append(gen_plan_traj)
            comp_lat_pred_list.append(comp_lat_pred)
            comp_lon_pred_list.append(comp_lon_pred)

        # Plot the vehicles' history
        for ax in axs:
            self.plotHistTraj(ax)

        for i in range(num_gen):
            ax = axs[i]
            # red line -- plan trajectory
            plan_trajs = gen_plan_trajs_list[i]
            plan_line = self.colorline(
                np.concatenate((plan_trajs[:, 0, 1], self.gt_plan_traj[0][-1:, 1])) * FEET2METER,
                np.concatenate((plan_trajs[:, 0, 0], self.gt_plan_traj[0][-1:, 0])) * FEET2METER,
                ax, zorder=ZORDER['planTraj'])
            # predictive traj
            comp_lat_pred = comp_lat_pred_list[i]
            comp_lon_pred = comp_lon_pred_list[i]
            comp_pred_traj = comp_pred_trajs_list[i]
            
            top_x = 1
            idx, jointP = self.getTopX(comp_lat_pred, comp_lon_pred, top_x)
            for j in range(self.numPred):  # number of vehicles
                P_max = jointP[j, idx[j, -1]]
                for l in range(top_x):  # number of prediction results
                    k = idx[j, l]
                    self.colorline(
                        np.concatenate((comp_pred_traj[k, self.gt_targs_traj[1][j].shape[0]::-1, j, 1],
                                        self.gt_targs_traj[0][j][-1:, 1]), 0) * FEET2METER,
                        np.concatenate((comp_pred_traj[k, self.gt_targs_traj[1][j].shape[0]::-1, j, 0],
                                        self.gt_targs_traj[0][j][-1:, 0]), 0) * FEET2METER,
                        ax, zorder=ZORDER['predTraj'])

            # plot lanes
            min_x = min(min([t[0, 1] for t in self.gt_targs_traj[0]]),
                        min([t[0, 1] for t in self.gt_nbrs_traj[0]]) if len(self.gt_nbrs_traj[0]) else 99999,
                        self.gt_plan_traj[0][0, 1])
            # To make sure t has length
            max_x = max(max([t[-1, 1] for t in self.gt_targs_traj[1] if len(t)]),
                        max([t[-1, 1] for t in self.gt_plan_traj[1]]),
                        max(t[..., 1].max() for t in comp_pred_trajs_list))
            self.setLanes(ax, self.dsId, self.dataSrc, min_x, max_x, line_color='silver', zorder=ZORDER['lane'])

        fig.tight_layout()
        plt.close(fig)
        return fig
    
# class Base:
#     def __init__(self, x):
#         print("Base")
#         self.x = x

# class A(Base):
#     def __init__(self):
#         print("A")
# class B(Base):
#     def __init__(self):
#         print("B")

# class C(A, B):
#     def __init__(self):
#         print("C")

# if __name__ == '__main__':
#     c = C()    
    
if __name__ == '__main__':
    #visual = Visual()
    x = np.linspace(0, 4.*np.pi, 1000)
    y = np.sin(x)

    fig, axes = plt.subplots()

    Visual().colorline(x, y)

    plt.xlim(x.min(), x.max())
    plt.ylim(-1.0, 1.0)
    plt.show()

