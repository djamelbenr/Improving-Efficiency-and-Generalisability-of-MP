import os  
import h5py 
import numpy as np 
import torch  

from torch.utils.data import Dataset 
from utilz import quintic_spline, fitting_traj_by_qs  


# Define the highwayTrajDataset class which inherits from the PyTorch Dataset class
class highwayTrajDataset(Dataset):
    def __init__(self, path, t_h=30, t_f=50, d_s=2,
                 enc_size=64, targ_enc_size=112, grid_size=(25, 5), fit_plan_traj=False, fit_plan_further_ds=1):
        if not os.path.exists(path):  # Check if the specified path exists
            raise RuntimeError("{} not exists!!".format(path))  # Raise an error if the path does not exist
        
        if path.endswith('.mat'):  # Check if the path ends with '.mat'
            f = h5py.File(path, 'r')  # Open the .mat file in read mode
            f_tracks = f['tracks']  # Load the 'tracks' data from the file
            track_cols, track_rows = f_tracks.shape  # Get the shape of the tracks data
            self.Data = np.transpose(f['traj'])  # Transpose and load 'traj' data into self.Data
            self.Tracks = []  # Initialize an empty list to store track data
            for i in range(track_rows):  # Iterate over the number of rows in tracks
                self.Tracks.append([np.transpose(f[f_tracks[j][i]][:]) for j in range(track_cols)])  # Append transposed track data to self.Tracks
        else:
            raise RuntimeError("Path should end with '.mat' for file or '/' for folder")  # Raise an error if the path does not end with '.mat'
        
        if int(torch.__version__[0]) >= 1 and int(torch.__version__[2]) >= 2:  # Check if PyTorch version is 1.2.0 or higher
            self.mask_num_type = torch.bool  # Use torch.bool for mask data type
        else:
            self.mask_num_type = torch.uint8  # Use torch.uint8 for mask data type if PyTorch version is lower
        
        self.t_h = t_h  # Set the length of track history
        self.t_f = t_f  # Set the length of predicted trajectory
        
        self.d_s = d_s  # Set the downsampling rate of all trajectories to be processed
        self.enc_size = enc_size  # Set the encoding size for trajectories
        self.targ_enc_size = targ_enc_size  # Set the target encoding size for trajectories
        self.hist_len = self.t_h // self.d_s + 1  # Calculate the length of the history trajectory data
        self.fut_len = self.t_f // self.d_s  # Calculate the length of the future trajectory data
        self.plan_len = self.t_f // self.d_s  # Calculate the length of the planning trajectory data
        self.fit_plan_traj = fit_plan_traj  # Set the flag to fit the future planned trajectory in testing/evaluation
        self.further_ds_plan = fit_plan_further_ds  # Set the further downsampling rate to restrict the planning info
        self.cell_length = 8  # Set the length of a cell in the social context grid
        self.cell_width = 7  # Set the width of a cell in the social context grid
        self.grid_size = grid_size  # Set the size of the social context grid
        self.grid_cells = grid_size[0] * grid_size[1]  # Calculate the total number of cells in the grid
        self.grid_length = self.cell_length * grid_size[0]  # Calculate the total length of the grid
        self.grid_width = self.cell_width * grid_size[1]  # Calculate the total width of the grid

    def __len__(self):  # Define the length function
        return len(self.Data)  # Return the length of the data

    def itsDsId(self, idx):  # Define a function to get the dataset ID
        return self.Data[idx, 0].astype(int)  # Return the dataset ID as an integer

    def itsPlanVehId(self, idx):  # Define a function to get the planned vehicle ID
        return self.Data[idx, 1].astype(int)  # Return the planned vehicle ID as an integer

    def itsTime(self, idx):  # Define a function to get the time
        return self.Data[idx, 2]  # Return the time

    def itsLocation(self, idx):  # Define a function to get the location
        return self.Data[idx, 3:5]  # Return the location (x, y) as an array

    def itsPlanVehBehavior(self, idx):  # Define a function to get the planned vehicle's behaviour
        return int(self.Data[idx, 6] + (self.Data[idx, 7] - 1) * 3)  # Return the behaviour as an integer

    def itsPlanVehSize(self, idx):  # Define a function to get the planned vehicle's size
        return self.Data[idx, 8:10]  # Return the size (length, width) as an array

    def itsPlanVehDynamic(self, idx):  # Define a function to get the planned vehicle's dynamics
        planVel, planAcc = self.getDynamic(self.itsDsId(idx), self.itsPlanVehId(idx), self.itsTime(idx))  # Get the velocity and acceleration
        return planVel, planAcc  # Return the velocity and acceleration

    def itsCentGrid(self, idx):  # Define a function to get the central grid
        return self.Data[idx, 13:].astype(int)  # Return the central grid as an array of integers

    def itsTargVehsId(self, idx):  # Define a function to get the target vehicles' IDs
        centGrid = self.itsCentGrid(idx)  # Get the central grid
        targVehsId = centGrid[np.nonzero(centGrid)]  # Get the non-zero elements from the grid (vehicle IDs)
        return targVehsId  # Return the target vehicles' IDs

    def itsNbrVehsId(self, idx):  # Define a function to get the neighbouring vehicles' IDs
        dsId = self.itsDsId(idx)  # Get the dataset ID
        planVehId = self.itsPlanVehId(idx)  # Get the planned vehicle ID
        targVehsId = self.itsTargVehsId(idx)  # Get the target vehicles' IDs
        t = self.itsTime(idx)  # Get the time
        nbrVehsId = np.array([], dtype=np.int64)  # Initialize an empty array to store neighbour vehicle IDs
        for target in targVehsId:  # Iterate over each target vehicle
            subGrid = self.getGrid(dsId, target, t)  # Get the grid for the target vehicle
            subIds = subGrid[np.nonzero(subGrid)]  # Get the non-zero elements from the grid (neighbour IDs)
            for i in subIds:  # Iterate over each neighbour ID
                if i == planVehId or any(i == targVehsId) or any(i == nbrVehsId):  # Skip if it's the planned vehicle or already added
                    continue
                else:
                    nbrVehsId = np.append(nbrVehsId, i)  # Add the neighbour ID to the array
        return nbrVehsId  # Return the neighbouring vehicles' IDs

    def itsTargsCentLoc(self, idx):  # Define a function to get the target vehicles' central locations
        dsId = self.itsDsId(idx)  # Get the dataset ID
        t = self.itsTime(idx)  # Get the time
        centGrid = self.itsCentGrid(idx)  # Get the central grid
        targsCenterLoc = np.empty((0, 2), dtype=np.float32)  # Initialize an empty array to store target locations
        for target in centGrid:  # Iterate over each target in the grid
            if target:  # If the target is not zero
                targsCenterLoc = np.vstack([targsCenterLoc, self.getLocation(dsId, target, t)])  # Add the target's location to the array
        return torch.from_numpy(targsCenterLoc)  # Convert the array to a PyTorch tensor and return it

    def itsAllAroundSizes(self, idx):  # Define a function to get the sizes of all surrounding vehicles
        dsId = self.itsDsId(idx)  # Get the dataset ID
        centGrid = self.itsCentGrid(idx)  # Get the central grid
        t = self.itsTime(idx)  # Get the time
        planVehSize = []  # Initialize a list to store the planned vehicle size
        targVehSizes = []  # Initialize a list to store the target vehicles' sizes
        nbsVehSizes = []  # Initialize a list to store the neighbouring vehicles' sizes
        planVehSize.append(self.getSize(dsId, self.itsPlanVehId(idx)))  # Get the planned vehicle size and add it to the list
        for i, target in enumerate(centGrid):  # Iterate over each target in the grid
            if target:  # If the target is not zero
                targVehSizes.append(self.getSize(dsId, target))  # Get the target vehicle size and add it to the list
                targVehGrid = self.getGrid(dsId, target, t)  # Get the grid for the target vehicle
                for targetNb in targVehGrid:  # Iterate over each neighbour in the grid
                    if targetNb:  # If the neighbour is not zero
                        nbsVehSizes.append(self.getSize(dsId, targetNb))  # Get the neighbour size and add it to the list
        return np.asarray(planVehSize), np.asarray(targVehSizes), np.asarray(nbsVehSizes)  # Return the sizes as arrays


    ## Functions for retrieving trajectory data with absolute coordinate, mainly used for visualization
    def itsAllGroundTruthTrajs(self, idx):  # Define a function to get all ground truth trajectories
        return [self.absPlanTraj(idx), self.absTargsTraj(idx), self.absNbrsTraj(idx)]  # Return the planned, target, and neighbour trajectories
          
    def absPlanTraj(self, idx):  # Define a function to get the absolute planned trajectory
        dsId = self.itsDsId(idx)  # Get the dataset ID
        planVeh = self.itsPlanVehId(idx)  # Get the planned vehicle ID
        t = self.itsTime(idx)  # Get the time
        colIndex = np.where(self.Tracks[dsId - 1][planVeh - 1][0, :] == t)[0][0]  # Find the column index for the given time
        vehTrack = self.Tracks[dsId - 1][planVeh - 1].transpose()  # Get the transposed track data for the planned vehicle
        planHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]  # Get the history of the planned vehicle
        planFut = vehTrack[(colIndex + self.d_s): (colIndex + self.t_f + 1): self.d_s, 1:3]  # Get the future trajectory of the planned vehicle
        return [planHis, planFut]  # Return the history and future trajectory

    def absTargsTraj(self, idx):  # Define a function to get the absolute target vehicles' trajectories
        dsId = self.itsDsId(idx)  # Get the dataset ID
        targVehs = self.itsTargVehsId(idx)  # Get the target vehicles' IDs
        t = self.itsTime(idx)  # Get the time
        targHisList, targFutList = [], []  # Initialize lists to store history and future trajectories
        for targVeh in targVehs:  # Iterate over each target vehicle
            colIndex = np.where(self.Tracks[dsId - 1][targVeh - 1][0, :] == t)[0][0]  # Find the column index for the given time
            vehTrack = self.Tracks[dsId - 1][targVeh - 1].transpose()  # Get the transposed track data for the target vehicle
            targHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]  # Get the history of the target vehicle
            targFut = vehTrack[(colIndex + self.d_s): (colIndex + self.t_f + 1): self.d_s, 1:3]  # Get the future trajectory of the target vehicle
            targHisList.append(targHis)  # Add the history to the list
            targFutList.append(targFut)  # Add the future trajectory to the list
        return [targHisList, targFutList]  # Return the history and future trajectories

    def absNbrsTraj(self, idx):  # Define a function to get the absolute neighbour vehicles' trajectories
        dsId = self.itsDsId(idx)  # Get the dataset ID
        nbrVehs = self.itsNbrVehsId(idx)  # Get the neighbour vehicles' IDs
        t = self.itsTime(idx)  # Get the time
        nbrHisList, nbrFutList = [], []  # Initialize lists to store history and future trajectories
        for nbrVeh in nbrVehs:  # Iterate over each neighbour vehicle
            colIndex = np.where(self.Tracks[dsId - 1][nbrVeh - 1][0, :] == t)[0][0]  # Find the column index for the given time
            vehTrack = self.Tracks[dsId - 1][nbrVeh - 1].transpose()  # Get the transposed track data for the neighbour vehicle
            targHis = vehTrack[np.maximum(0, colIndex - self.t_h): (colIndex + 1): self.d_s, 1:3]  # Get the history of the neighbour vehicle
            nbrHisList.append(targHis)  # Add the history to the list
        return [nbrHisList, nbrFutList]  # Return the history and future trajectories
   
    def batchTargetVehsInfo(self, idxs):  # Define a function to get batch information for target vehicles
        count = 0  # Initialize a counter
        dsIds = np.zeros(len(idxs) * self.grid_cells, dtype=int)  # Initialize an array to store dataset IDs
        vehIds = np.zeros(len(idxs) * self.grid_cells, dtype=int)  # Initialize an array to store vehicle IDs

        for idx in idxs:  # Iterate over each index in the batch
            dsId = self.itsDsId(idx)  # Get the dataset ID
            targets = self.itsCentGrid(idx)  # Get the central grid
            targetsIndex = np.nonzero(targets)  # Get the non-zero elements (target vehicles)

            for index in targetsIndex[0]:  # Iterate over each target
                dsIds[count] = dsId  # Store the dataset ID
                vehIds[count] = targets[index]  # Store the vehicle ID
                count += 1  # Increment the counter

        return [dsIds[:count], vehIds[:count]]  # Return the dataset IDs and vehicle IDs

    ## Avoid searching the correspond column for too many times.
    def getTracksCol(self, dsId, vehId, t):  # Define a function to get the column index for a vehicle at a given time
        return np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]  # Return the column index

    ## Get the vehicle's location from tracks
    def getLocation(self, dsId, vehId, t):  # Define a function to get the vehicle's location at a given time
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]  # Find the column index for the given time
        location = self.getLocationByCol(dsId, vehId, colIndex)  # Get the location using the column index
        return location  # Return the location

    def getLocationByCol(self, dsId, vehId, colIndex):  # Define a function to get the vehicle's location using a column index
        return self.Tracks[dsId - 1][vehId - 1][1:3, colIndex].transpose()  # Return the location as an array

    ## Get the vehicle's maneuver given dataset id, vehicle id and time point t.
    def getManeuverByCol(self, dsId, vehId, colIndex):  # Define a function to get the vehicle's manoeuvre using a column index
        return self.Tracks[dsId - 1][vehId - 1][4:6, colIndex].astype(int)  # Return the manoeuvre as an integer array

    def getManeuver(self, dsId, vehId, t):  # Define a function to get the vehicle's manoeuvre at a given time
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]  # Find the column index for the given time
        lat_lon_maneuvers = self.getManeuverByCol(dsId, vehId, colIndex)  # Get the manoeuvre using the column index
        return lat_lon_maneuvers  # Return the manoeuvre

    ## Get the vehicle's nearby neighbours
    def getGrid(self, dsId, vehId, t):  # Define a function to get the grid of nearby neighbours at a given time
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]  # Find the column index for the given time
        grid = self.getGridByCol(dsId, vehId, colIndex)  # Get the grid using the column index
        return grid  # Return the grid

    def getGridByCol(self, dsId, vehId, colIndex):  # Define a function to get the grid using a column index
        return self.Tracks[dsId - 1][vehId - 1][11:, colIndex].astype(int)  # Return the grid as an integer array

    ## Get the vehicle's dynamic (velocity & acceleration) given dataset id, vehicle id and time point t.
    def getDynamic(self, dsId, vehId, t):  # Define a function to get the vehicle's dynamics at a given time
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]  # Find the column index for the given time
        vel_acc = self.getDynamicByCol(dsId, vehId, colIndex)  # Get the dynamics using the column index
        return vel_acc  # Return the velocity and acceleration

    def getDynamicByCol(self, dsId, vehId, colIndex):  # Define a function to get the vehicle's dynamics using a column index
        return self.Tracks[dsId - 1][vehId - 1][9:11, colIndex]  # Return the dynamics as an array
    
    ## Get the vehicle's size (length & width) given dataset id and vehicle id
    def getSize(self, dsId, vehId):  # Define a function to get the vehicle's size
        length_width = self.Tracks[dsId - 1][vehId - 1][6:8, 0]  # Get the length and width from the track data
        return length_width  # Return the length and width
    
    ## Helper function to get track history
    def getHistory(self, dsId, vehId, refVehId, t, wholePeriod=False):  # Define a function to get the track history
        if vehId == 0:  # Check if the vehicle ID is 0
            return np.empty([0, 2])  # Return an empty array if there's no vehicle in that grid
        else:
            vehColIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]  # Get the column index for the vehicle
            refColIndex = np.where(self.Tracks[dsId - 1][refVehId - 1][0, :] == t)[0][0]  # Get the column index for the reference vehicle
            vehTrack = self.Tracks[dsId - 1][vehId - 1][1:3].transpose()  # Get the vehicle's track data
            refTrack = self.Tracks[dsId - 1][refVehId - 1][1:3].transpose()  # Get the reference vehicle's track data
            if wholePeriod:  # Check if the whole period is required
                refStpt = np.maximum(0, refColIndex - self.t_h)  # Calculate the start point for the reference vehicle
                refEnpt = refColIndex + 1  # Calculate the end point for the reference vehicle
                refPos = refTrack[refStpt:refEnpt:self.d_s, :]  # Get the reference vehicle's positions
            else:
                refPos = np.tile(refTrack[refColIndex, :], (self.hist_len, 1))  # Tile the last position for the reference vehicle
            stpt = np.maximum(0, vehColIndex - self.t_h)  # Calculate the start point for the vehicle
            enpt = vehColIndex + 1  # Calculate the end point for the vehicle
            vehPos = vehTrack[stpt:enpt:self.d_s, :]  # Get the vehicle's positions
            if len(vehPos) < self.hist_len:  # Check if the history length is less than expected
                histPart = vehPos - refPos[-len(vehPos):]  # Calculate the history part
                paddingPart = np.tile(histPart[0, :], (self.hist_len - len(vehPos), 1))  # Create a padding part
                hist = np.concatenate((paddingPart, histPart), axis=0)  # Concatenate the padding and history parts
                return hist  # Return the history
            else:
                hist = vehPos - refPos  # Calculate the history
                return hist  # Return the history

    ## Helper function to get track future
    def getFuture(self, dsId, vehId, t):  # Define a function to get the track future
        colIndex = np.where(self.Tracks[dsId - 1][vehId - 1][0, :] == t)[0][0]  # Get the column index for the vehicle
        futTraj = self.getFutureByCol(dsId, vehId, colIndex)  # Get the future trajectory using the column index
        return futTraj  # Return the future trajectory

    def getFutureByCol(self, dsId, vehId, colIndex):  # Define a function to get the track future using a column index
        vehTrack = self.Tracks[dsId - 1][vehId - 1].transpose()  # Get the vehicle's track data
        refPos = self.Tracks[dsId - 1][vehId - 1][1:3, colIndex].transpose()  # Get the reference position
        stpt = colIndex + self.d_s  # Calculate the start point for the future trajectory
        enpt = np.minimum(len(vehTrack), colIndex + self.t_f + 1)  # Calculate the end point for the future trajectory
        futTraj = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos  # Get the future trajectory
        return futTraj  # Return the future trajectory

    def getPlanFuture(self, dsId, planId, refVehId, t):  # Define a function to get the planned vehicle's future trajectory
        refColIndex = np.where(self.Tracks[dsId - 1][refVehId - 1][0, :] == t)[0][0]  # Get the column index for the reference vehicle
        refPos = self.Tracks[dsId - 1][refVehId - 1][1:3, refColIndex].transpose()  # Get the reference position
        planColIndex = np.where(self.Tracks[dsId - 1][planId - 1][0, :] == t)[0][0]  # Get the column index for the planned vehicle
        stpt = planColIndex  # Set the start point for the planned trajectory
        enpt = planColIndex + self.t_f + 1  # Set the end point for the planned trajectory
        
        planGroundTrue = self.Tracks[dsId - 1][planId - 1][1:3, stpt:enpt:self.d_s].transpose()  # Get the planned trajectory
        planFut = planGroundTrue.copy()  # Create a copy of the planned trajectory
        
        if self.fit_plan_traj:  # Check if the trajectory needs to be fitted
            wayPoint = np.arange(0, self.t_f + self.d_s, self.d_s)  # Create an array of waypoints
            wayPoint_to_fit = np.arange(0, self.t_f + 1, self.d_s * self.further_ds_plan)  # Create an array of waypoints for fitting
            planFut_to_fit = planFut[::self.further_ds_plan, ]  # Downsample the planned trajectory
            
            laterParam = fitting_traj_by_qs(wayPoint_to_fit, planFut_to_fit[:, 0])  # Fit the lateral parameters
            longiParam = fitting_traj_by_qs(wayPoint_to_fit, planFut_to_fit[:, 1])  # Fit the longitudinal parameters
            
            planFut[:, 0] = quintic_spline(wayPoint, *laterParam)  # Apply the quintic spline to the lateral parameters
            planFut[:, 1] = quintic_spline(wayPoint, *longiParam)  # Apply the quintic spline to the longitudinal parameters
        revPlanFut = np.flip(planFut[1:, ] - refPos, axis=0).copy()  # Flip and adjust the planned trajectory
        return revPlanFut  # Return the reversed planned future trajectory
    
    def __getitem__(self, idx):  # Define the getitem function to retrieve data by index
        dsId = self.itsDsId(idx)  # Get the dataset ID
        centVehId = self.itsPlanVehId(idx)  # Get the planned vehicle ID
        t = self.itsTime(idx)  # Get the time
        centGrid = self.itsCentGrid(idx)  # Get the central grid
        planGridLocs = []  # Initialize a list to store grid locations
        targsHists = []  # Initialize a list to store target histories
        targsFuts = []  # Initialize a list to store target futures
        targsLonEnc = []  # Initialize a list to store target longitudinal encodings
        targsLatEnc = []  # Initialize a list to store target lateral encodings
        nbsHists = []  # Initialize a list to store neighbour histories
        planFuts = []  # Initialize a list to store planned futures
        targsVehs = np.zeros(self.grid_cells)  # Initialize an array to store target vehicles
        for id, target in enumerate(centGrid):  # Iterate over each target in the grid
            if target:  # If the target is not zero
                targetColumn = self.getTracksCol(dsId, target, t)  # Get the column index for the target
                grid = self.getGridByCol(dsId, target, targetColumn)  # Get the grid for the target vehicle
                targsVehs[id] = target  # Store the target vehicle ID

                targsHists.append(self.getHistory(dsId, target, target, t))  # Get the target's history and add it to the list
                targsFuts.append(self.getFutureByCol(dsId, target, targetColumn))  # Get the target's future and add it to the list
                
                latMan, lonMan = self.getManeuverByCol(dsId, target, targetColumn)  # Get the target's manoeuvres

                lat_lon_Man = self.getManeuver(dsId, target, t)  # Get the lateral and longitudinal manoeuvres

                lat_enc = np.zeros([3])  # Initialize an array for lateral encoding
                lon_enc = np.zeros([2])  # Initialize an array for longitudinal encoding

                lat_enc[latMan - 1] = 1  # Set the appropriate index in the lateral encoding array
                lon_enc[lonMan - 1] = 1  # Set the appropriate index in the longitudinal encoding array

                targsLatEnc.append(lat_enc)  # Add the lateral encoding to the list
                targsLonEnc.append(lon_enc)  # Add the longitudinal encoding to the list
                
                nbsHists.append([self.getHistory(dsId, i, target, t, wholePeriod=True) for i in grid])  # Get the neighbours' histories
                
                planGridLocs.append(np.where(grid == centVehId)[0][0])  # Store the location of the planned vehicle in the grid
                planFuts.append(self.getPlanFuture(dsId, centVehId, target, t))  # Get the planned future and add it to the list

        return planFuts, nbsHists, \
               targsHists, targsFuts, targsLonEnc, targsLatEnc, \
               centGrid, planGridLocs, idx  # Return the relevant data


    ## Collate function for dataloader
    def collate_fn(self, samples):  # Define a collate function for batching data
        targs_batch_size = 0  # Initialize the batch size for targets
        nbs_batch_size = 0  # Initialize the batch size for neighbours
        for _, nbsHists, targsHists, _, _, _, _, _, _ in samples:  # Iterate over each sample
            targs_batch_size += len(targsHists)  # Add the number of target histories to the batch size
            nbs_number = [sum([len(nbs) > 0 for nbs in sub_nbsHist]) for sub_nbsHist in nbsHists]  # Count the number of non-empty neighbour histories
            nbs_batch_size += sum(nbs_number)  # Add the number of neighbours to the batch size
        
        # Initialize all tensors for batching
        nbsHist_batch = torch.zeros(self.hist_len, nbs_batch_size, 2)  # Initialize the neighbour history batch tensor
        targsHist_batch = torch.zeros(self.hist_len, targs_batch_size, 2)  # Initialize the target history batch tensor
        targsFut_batch = torch.zeros(self.fut_len, targs_batch_size, 2)  # Initialize the target future batch tensor
        
        lat_enc_batch = torch.zeros(targs_batch_size, 3)  # Initialize the lateral encoding batch tensor
        lon_enc_batch = torch.zeros(targs_batch_size, 2)  # Initialize the longitudinal encoding batch tensor
        
        planFut_batch = torch.zeros(self.plan_len, targs_batch_size, 2)  # Initialize the planned future batch tensor
        idxs = []  # Initialize a list to store indices
        pos = [0, 0]  # Initialize a position array
        
        # Initialize the masks for the batches
        nbsMask_batch = torch.zeros(targs_batch_size, self.grid_size[1], self.grid_size[0], self.enc_size, dtype=self.mask_num_type)
        planMask_batch = torch.zeros(targs_batch_size, self.grid_size[1], self.grid_size[0], self.enc_size, dtype=self.mask_num_type)
        targsEncMask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.targ_enc_size, dtype=self.mask_num_type)
        targsFutMask_batch = torch.zeros(self.fut_len, targs_batch_size, 2)
        
        targetCount = 0  # Initialize the target count
        nbCount = 0  # Initialize the neighbour count
        
        for i, (planFuts, nbsHists, targsHists, targsFuts, targsLonEnc, targsLatEnc, centGrid, planGridLocs, idx) in enumerate(samples):  # Iterate over each sample
            idxs.append(idx)  # Store the index
            centGridIndex = centGrid.nonzero()[0]  # Get the non-zero indices of the central grid
            for j in range(len(targsFuts)):  # Iterate over each target future
                targsHist_batch[0:len(targsHists[j]), targetCount, 0] = torch.from_numpy(targsHists[j][:, 0])  # Store the target's history (x)
                targsHist_batch[0:len(targsHists[j]), targetCount, 1] = torch.from_numpy(targsHists[j][:, 1])  # Store the target's history (y)
                targsFut_batch[0:len(targsFuts[j]), targetCount, 0] = torch.from_numpy(targsFuts[j][:, 0])  # Store the target's future (x)
                targsFut_batch[0:len(targsFuts[j]), targetCount, 1] = torch.from_numpy(targsFuts[j][:, 1])  # Store the target's future (y)
                targsFutMask_batch[0:len(targsFuts[j]), targetCount, :] = 1  # Set the future mask
                
                pos[0] = centGridIndex[j] % self.grid_size[0]  # Calculate the position in the grid (x)
                pos[1] = centGridIndex[j] // self.grid_size[0]  # Calculate the position in the grid (y)
                
                targsEncMask_batch[i, pos[1], pos[0], :] = torch.ones(self.targ_enc_size).byte()  # Set the target encoding mask
                
                lat_enc_batch[targetCount, :] = torch.from_numpy(targsLatEnc[j])  # Store the lateral encoding
                lon_enc_batch[targetCount, :] = torch.from_numpy(targsLonEnc[j])  # Store the longitudinal encoding
                
                planFut_batch[0:len(planFuts[j]), targetCount, 0] = torch.from_numpy(planFuts[j][:, 0])  # Store the planned future (x)
                planFut_batch[0:len(planFuts[j]), targetCount, 1] = torch.from_numpy(planFuts[j][:, 1])  # Store the planned future (y)
                
                for index, nbHist in enumerate(nbsHists[j]):  # Iterate over each neighbour history
                    if len(nbHist) != 0:  # Check if the history is not empty
                        nbsHist_batch[0:len(nbHist), nbCount, 0] = torch.from_numpy(nbHist[:, 0])  # Store the neighbour's history (x)
                        nbsHist_batch[0:len(nbHist), nbCount, 1] = torch.from_numpy(nbHist[:, 1])  # Store the neighbour's history (y)
                        pos[0] = index % self.grid_size[0]  # Calculate the position in the grid (x)
                        pos[1] = index // self.grid_size[0]  # Calculate the position in the grid (y)
                        nbsMask_batch[targetCount, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()  # Set the neighbour mask
                        nbCount += 1  # Increment the neighbour count
                        if index == planGridLocs[j]:  # Check if the index matches the planned vehicle's location
                            planMask_batch[targetCount, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()  # Set the planned vehicle mask
                targetCount += 1  # Increment the target count

        return nbsHist_batch, nbsMask_batch, \
               planFut_batch, planMask_batch, \
               targsHist_batch, targsEncMask_batch, \
               targsFut_batch, targsFutMask_batch, lat_enc_batch, lon_enc_batch, idxs  # Return the batch tensors and indices
