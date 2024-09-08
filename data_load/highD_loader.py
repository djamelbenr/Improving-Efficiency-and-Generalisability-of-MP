import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import savemat

class HighDDatasetProcessor:
    def __init__(self, grid_length=25, grid_width=5, cell_length=8, cell_width=7, dataset_to_use=120):
        self.grid_length = grid_length
        self.grid_width = grid_width
        self.cell_length = cell_length
        self.cell_width = cell_width
        self.grid_cells = grid_length * grid_width
        self.grid_cent_location = (grid_length * grid_width) // 2
        self.dataset_to_use = dataset_to_use
        self.raw_folder = f'./dataset/highD/{grid_length}x{grid_width}_raw/'
        self.post_folder = f'./dataset/highD/{grid_length}x{grid_width}/'
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.post_folder, exist_ok=True)
        self.traj = {}
        self.tracks = {}
    
    def load_data(self):
        print('Loading data...')
        for k in range(1, self.dataset_to_use + 1):
            if k % 2:
                dataset_name = f'./raw_highd_ngsim_format/{(k // 2) + 1:02d}-fwd.csv'
            else:
                dataset_name = f'./raw_highd_ngsim_format/{(k // 2) + 1:02d}-bck.csv'
            
            df = pd.read_csv(dataset_name)
            self.traj[k] = df.iloc[:, :14].to_numpy()
            self.traj[k] = np.hstack((np.full((self.traj[k].shape[0], 1), k), self.traj[k]))  # Add dataset ID
            
            # Reorganize columns: Dataset ID, Vehicle ID, Frame index, Local X, Local Y, Lane ID, and more
            self.traj[k] = self.traj[k][:, [0, 1, 2, 5, 6, 14, 9, 10, 11, 12, 13]]
            # Leave space for maneuver labels and grid
            self.traj[k] = np.hstack([self.traj[k][:, :6], np.zeros((self.traj[k].shape[0], 2)), 
                                       self.traj[k][:, 6:], np.zeros((self.traj[k].shape[0], self.grid_cells))])
        
        # Adjust vehicle's Y location
        self.offset = np.zeros(self.dataset_to_use)
        for k in range(1, self.dataset_to_use + 1):
            self.traj[k][:, 5] -= 0.5 * self.traj[k][:, 8]  # Adjust based on vehicle length
            min_y = np.min(self.traj[k][:, 5])
            if min_y < 0:
                self.traj[k][:, 5] -= min_y

    def parse_fields(self):
        print('Parsing fields...')
        for ii in range(1, self.dataset_to_use + 1):
            print(f'Now processing dataset {ii}')
            traj_data = self.traj[ii]
            
            for k in range(traj_data.shape[0]):
                veh_id = traj_data[k, 1]
                time = traj_data[k, 2]
                veh_traj = traj_data[traj_data[:, 1] == veh_id]
                ind = np.where(veh_traj[:, 2] == time)[0][0]
                
                lane = traj_data[k, 5]
                
                # Lateral maneuver
                ub = min(veh_traj.shape[0], ind + 40)
                lb = max(0, ind - 40)
                if veh_traj[ub, 5] > veh_traj[ind, 5] or veh_traj[ind, 5] > veh_traj[lb, 5]:
                    traj_data[k, 6] = 3  # Turn Right
                elif veh_traj[ub, 5] < veh_traj[ind, 5] or veh_traj[ind, 5] < veh_traj[lb, 5]:
                    traj_data[k, 6] = 2  # Turn Left
                else:
                    traj_data[k, 6] = 1  # Keep lane
                
                # Longitudinal maneuver
                ub = min(veh_traj.shape[0], ind + 50)
                lb = max(0, ind - 30)
                if ub == ind or lb == ind:
                    traj_data[k, 7] = 1  # Normal
                else:
                    v_hist = (veh_traj[ind, 4] - veh_traj[lb, 4]) / (ind - lb)
                    v_fut = (veh_traj[ub, 4] - veh_traj[ind, 4]) / (ub - ind)
                    if v_fut / v_hist < 0.8:
                        traj_data[k, 7] = 2  # Brake
                    else:
                        traj_data[k, 7] = 1  # Normal
                
                # Grid locations
                cent_veh_x = traj_data[k, 3]
                cent_veh_y = traj_data[k, 4]
                grid_min_x = cent_veh_x - 0.5 * self.grid_width * self.cell_width
                grid_min_y = cent_veh_y - 0.5 * self.grid_length * self.cell_length
                
                other_vehs = traj_data[traj_data[:, 2] == time][:, [1, 3, 4]]
                other_vehs_in_range = other_vehs[(np.abs(other_vehs[:, 2] - cent_veh_y) < 0.5 * self.grid_length * self.cell_length) &
                                                 (np.abs(other_vehs[:, 1] - cent_veh_x) < 0.5 * self.grid_width * self.cell_width)]
                
                if other_vehs_in_range.size > 0:
                    other_vehs_in_range[:, 1] = np.ceil((other_vehs_in_range[:, 1] - grid_min_x) / self.cell_width)
                    other_vehs_in_range[:, 2] = np.ceil((other_vehs_in_range[:, 2] - grid_min_y) / self.cell_length)
                    other_vehs_in_range[:, 2] += (other_vehs_in_range[:, 1] - 1) * self.grid_length
                    
                    for l in range(other_vehs_in_range.shape[0]):
                        grid_loc = int(other_vehs_in_range[l, 2])
                        if grid_loc != self.grid_cent_location:
                            traj_data[k, 13 + grid_loc] = other_vehs_in_range[l, 0]

    def merge_and_split(self):
        print('Merging and splitting into train, validation, and test sets...')
        traj_all = np.vstack([self.traj[k] for k in range(1, self.dataset_to_use + 1)])
        
        train_set, test_set = train_test_split(traj_all, test_size=0.2, random_state=42)
        train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=42)
        
        self.traj_tr = train_set
        self.traj_val = val_set
        self.traj_ts = test_set
        
        self.build_tracks()

    def build_tracks(self):
        print('Building tracks...')
        for k in range(1, self.dataset_to_use + 1):
            traj_set = self.traj[k]
            car_ids = np.unique(traj_set[:, 1])
            self.tracks[k] = {}
            for car_id in car_ids:
                self.tracks[k][car_id] = traj_set[traj_set[:, 1] == car_id, 2:].T

    def save_data(self):
        print('Saving data...')
        savemat(os.path.join(self.raw_folder, 'highdTrainRaw.mat'), {'trajTr': self.traj_tr, 'tracks': self.tracks})
        savemat(os.path.join(self.raw_folder, 'highdValRaw.mat'), {'trajVal': self.traj_val, 'tracks': self.tracks})
        savemat(os.path.join(self.raw_folder, 'highdTestRaw.mat'), {'trajTs': self.traj_ts, 'tracks': self.tracks})

        # Post-processed data
        savemat(os.path.join(self.post_folder, 'highdTrainAround.mat'), {'traj': self.traj_tr, 'tracks': self.tracks})
        savemat(os.path.join(self.post_folder, 'highdValAround.mat'), {'traj': self.traj_val, 'tracks': self.tracks})
        savemat(os.path.join(self.post_folder, 'highdTestAround.mat'), {'traj': self.traj_ts, 'tracks': self.tracks})

# Example of how to use the class
'''
if __name__ == "__main__":
    processor = HighDDatasetProcessor()
    processor.load_data()
    processor.parse_fields()
    processor.merge_and_split()
    processor.save_data()
'''
