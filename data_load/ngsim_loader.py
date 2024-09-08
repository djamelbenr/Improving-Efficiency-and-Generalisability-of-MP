import os
import numpy as np
from scipy.io import savemat
from multiprocessing import Pool


class NGSIMDatasetProcessor:
    """
    Class to process NGSIM dataset into required datasets.
    """
    # Initializer method to set default hyperparameters and file paths
    def __init__(self, lane_filter=False, grid_length=25, grid_width=5, cell_length=8, cell_width=7):
        """
        Initialize hyperparameters and save locations based on lane filtering.
        """
        # Assigning the value of lane_filter to a class variable
        self.lane_filter = lane_filter
        
        # Assigning grid length to the class variable
        self.grid_length = grid_length
        
        # Assigning grid width to the class variable
        self.grid_width = grid_width
        
        # Assigning cell length to the class variable
        self.cell_length = cell_length
        
        # Assigning cell width to the class variable
        self.cell_width = cell_width

        # Conditional statement to set folder paths based on lane filtering
        if self.lane_filter:
            # Creating a folder path for raw data if lane filter is applied
            self.raw_folder = f'./dataset/ngsim/{self.grid_length}x{self.grid_width}_raw/'
            # Creating a folder path for post-processed data if lane filter is applied
            self.post_folder = f'./dataset/ngsim/{self.grid_length}x{self.grid_width}/'
            # Creating a folder path for fixed target data if lane filter is applied
            self.fix_tar_folder = f'./dataset/ngsim/{self.grid_length}x{self.grid_width}_8Veh/'
        else:
            # Creating a folder path for raw data if no lane filter is applied
            self.raw_folder = f'./dataset/ngsim/{self.grid_length}x{self.grid_width}_nofL_raw/'
            # Creating a folder path for post-processed data if no lane filter is applied
            self.post_folder = f'./dataset/ngsim/{self.grid_length}x{self.grid_width}_nofL/'
            # Creating a folder path for fixed target data if no lane filter is applied
            self.fix_tar_folder = f'./dataset/ngsim/{self.grid_length}x{self.grid_width}_8Veh_nofL/'
        
        # Creating the directories if they do not exist
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.post_folder, exist_ok=True)
        os.makedirs(self.fix_tar_folder, exist_ok=True)
        
        # Calculating the total number of grid cells and assigning to a class variable
        self.grid_cells = self.grid_length * self.grid_width
        
        # Calculating the central grid location and assigning to a class variable
        self.grid_cent_location = int(np.ceil(self.grid_length * self.grid_width * 0.5))
    
    # Function to load data from given file paths
    def load_data(self, dataset_to_use, file_paths):
        """
        Load raw NGSIM data from provided file paths.
        """
        # Printing a message to indicate data loading
        print('Loading data...')
        
        # Initializing an empty dictionary to store trajectory data
        traj = {}
        
        # Looping over the dataset indices and corresponding file paths
        for i, file_path in enumerate(file_paths[:dataset_to_use]):
            # Assigning dataset ID based on the index
            dataset_id = i + 1
            # Loading the data from the file path and storing it in the trajectory dictionary
            traj[dataset_id] = np.loadtxt(file_path)
            # Adding a dataset ID column to the loaded data
            traj[dataset_id] = np.hstack([np.ones((traj[dataset_id].shape[0], 1)) * dataset_id, traj[dataset_id]])

        # Returning the loaded trajectory data
        return traj

    # Function to filter and preprocess trajectory data
    def filter_data(self, traj, dataset_to_use):
        """
        Filter and preprocess loaded trajectory data.
        """
        # Printing a message to indicate data filtering
        print('Filtering data...')
        
        # Looping over each dataset
        for k in range(1, dataset_to_use + 1):
            # Selecting specific columns from the loaded data for processing
            traj[k] = traj[k][:, [0, 1, 2, 5, 6, 14, 9, 10, 11, 12, 13]]

            # Filtering data based on lane ID if lane_filter is applied
            if self.lane_filter:
                print(f'Dataset-{k} #data: {traj[k].shape[0]} ==> ', end='')
                # Filtering out rows where lane ID is greater than or equal to 7
                traj[k] = traj[k][traj[k][:, 5] < 7, :]
                print(f'{traj[k].shape[0]} after filtering lane > 6')

            # Adjusting lane IDs for US101 dataset (datasets 1 to 3)
            if k <= 3:
                # Setting lane IDs greater than or equal to 6 to 6
                traj[k][traj[k][:, 5] >= 6, 5] = 6

            # Adding placeholder columns for maneuver labels and neighbor grid
            traj[k] = np.hstack([traj[k][:, :6], np.zeros((traj[k].shape[0], 2)), traj[k][:, 6:], 
                                 np.zeros((traj[k].shape[0], self.grid_cells))])

        # Returning the filtered trajectory data
        return traj

    # Function to parse fields like maneuver labels and neighbor grid locations
    def parse_fields(self, traj, dataset_to_use):
        """
        Parse fields like maneuver labels and neighbor grid.
        """
        # Printing a message to indicate field parsing
        print('Parsing fields...')
        
        # Using parallel processing with Pool to process multiple datasets concurrently
        with Pool(dataset_to_use) as pool:
            # Mapping the internal dataset processing function to each dataset
            pool.map(self._process_dataset, [(traj[ii], ii) for ii in range(1, dataset_to_use + 1)])
    
    # Helper function to process each dataset in parallel
    def _process_dataset(self, args):
        """
        Internal helper function to process each dataset.
        """
        # Unpacking the arguments into trajectory data and dataset index
        traj, ii = args
        
        # Looping over each row in the dataset
        for k in range(len(traj)):
            # Selecting rows corresponding to the same vehicle ID
            vehtraj = traj[traj[:, 1] == traj[k, 1], :]
            # Parsing lateral and longitudinal maneuvers for each row
            self._parse_maneuver(traj, vehtraj, k)

    # Function to parse lateral and longitudinal maneuvers for each row
    def _parse_maneuver(self, traj, vehtraj, k):
        """
        Parse lateral and longitudinal maneuvers for each row.
        """
        # Parsing lateral maneuver: check if lane ID changes within 40 frames forward or backward
        if len(vehtraj) > k + 40 and vehtraj[k + 40, 5] != vehtraj[k, 5]:
            # Mark as a right turn (3)
            traj[k, 6] = 3  # Turn right
        elif len(vehtraj) > k - 40 and vehtraj[k - 40, 5] != vehtraj[k, 5]:
            # Mark as a left turn (2)
            traj[k, 6] = 2  # Turn left
        else:
            # Mark as keeping lane (1)
            traj[k, 6] = 1  # Keep lane

        # Parsing longitudinal maneuver: check for braking or normal behavior
        if len(vehtraj) > k + 50 and (vehtraj[k + 50, 5] - vehtraj[k, 5]) / 50 < 0.8:
            # Mark as braking (2)
            traj[k, 7] = 2  # Brake
        else:
            # Mark as normal (1)
            traj[k, 7] = 1  # Normal

    # Function to merge datasets and split into training, validation, and test sets
    def merge_and_split(self, traj, dataset_to_use):
        """
        Merge datasets and split into train, validation, and test sets.
        """
        # Printing a message to indicate dataset splitting
        print('Splitting into train, validation, and test sets...')
        
        # Merging all dataset trajectories into a single array
        traj_all = np.vstack([traj[i] for i in range(1, dataset_to_use + 1)])
        
        # Initializing lists to store training, validation, and test data
        traj_tr, traj_val, traj_ts = [], [], []

        # Looping over each dataset
        for k in range(1, dataset_to_use + 1):
            # Calculating cutoffs for training and test sets based on vehicle ID
            ul1 = int(0.7 * np.max(traj_all[traj_all[:, 0] == k, 1]))
            ul2 = int(0.8 * np.max(traj_all[traj_all[:, 0] == k, 1]))

            # Appending training data for the current dataset
            traj_tr.append(traj_all[(traj_all[:, 0] == k) & (traj_all[:, 1] <= ul1), :])
            # Appending validation data for the current dataset
            traj_val.append(traj_all[(traj_all[:, 0] == k) & (traj_all[:, 1] > ul1) & (traj_all[:, 1] <= ul2), :])
            # Appending test data for the current dataset
            traj_ts.append(traj_all[(traj_all[:, 0] == k) & (traj_all[:, 1] > ul2), :])

        # Returning the concatenated training, validation, and test datasets
        return np.vstack(traj_tr), np.vstack(traj_val), np.vstack(traj_ts)

    # Function to filter edge cases based on trajectory history and future requirements
    def filter_edge_cases(self, traj_tr, traj_val, traj_ts, tracks):
        """
        Filter edge cases based on trajectory history and future requirements.
        """
        # Printing a message to indicate edge case filtering
        print('Filtering edge cases...')
        
        # Filtering training dataset for valid cases
        traj_tr = self._filter_traj(traj_tr, tracks)
        # Filtering validation dataset for valid cases
        traj_val = self._filter_traj(traj_val, tracks)
        # Filtering test dataset for valid cases
        traj_ts = self._filter_traj(traj_ts, tracks)

        # Returning the filtered training, validation, and test datasets
        return traj_tr, traj_val, traj_ts

    # Helper function to filter trajectory data based on history and future conditions
    def _filter_traj(self, traj, tracks):
        """
        Helper function to filter trajectory data.
        """
        # Initializing an array to store boolean flags for valid rows
        inds = np.zeros(traj.shape[0], dtype=bool)
        
        # Looping over each row in the trajectory data
        for i in range(traj.shape[0]):
            # Extracting the time index for the current row
            t = traj[i, 2]
            # Checking if the trajectory history and future satisfy conditions
            if tracks[int(traj[i, 0]), int(traj[i, 1])][0, 30] <= t and tracks[int(traj[i, 0]), int(traj[i, 1])][0, -1] >= t + 50:
                # Marking the row as valid
                inds[i] = True
        
        # Returning only the valid rows from the trajectory data
        return traj[inds, :]

    # Function to save processed datasets into MATLAB .mat files
    def save_data(self, traj_tr, traj_val, traj_ts, tracks):
        """
        Save processed data into .mat files.
        """
        # Printing a message to indicate saving
        print('Saving mat files...')
        
        # Saving the training dataset into a .mat file
        savemat(os.path.join(self.raw_folder, 'gridTrainAround.mat'), {'trajTr': traj_tr, 'tracks': tracks})
        
        # Saving the validation dataset into a .mat file
        savemat(os.path.join(self.raw_folder, 'gridValAround.mat'), {'trajVal': traj_val, 'tracks': tracks})
        
        # Saving the test dataset into a .mat file
        savemat(os.path.join(self.raw_folder, 'gridTestAround.mat'), {'trajTs': traj_ts, 'tracks': tracks})

    # Main function to orchestrate the entire data processing workflow
    def process(self, dataset_to_use, file_paths):
        """
        Main function to orchestrate data processing workflow.
        """
        # Load the raw data from the provided file paths
        traj = self.load_data(dataset_to_use, file_paths)
        
        # Filter and preprocess the loaded data
        traj = self.filter_data(traj, dataset_to_use)
        
        # Parse fields such as maneuver labels and neighbor grids
        self.parse_fields(traj, dataset_to_use)
        
        # Merge and split the data into train, validation, and test sets
        traj_tr, traj_val, traj_ts = self.merge_and_split(traj, dataset_to_use)
        
        # Placeholder for track data (to be implemented)
        tracks = {}
        
        # Filter edge cases from the datasets
        traj_tr, traj_val, traj_ts = self.filter_edge_cases(traj_tr, traj_val, traj_ts, tracks)
        
        # Save the processed datasets into .mat files
        self.save_data(traj_tr, traj_val, traj_ts, tracks)

# Main execution block
if __name__ == "__main__":
    # Defining the file paths to the raw NGSIM datasets
    file_paths = [
        './raw_ngsim/us101-0750am-0805am.txt',
        './raw_ngsim/us101-0805am-0820am.txt',
        './raw_ngsim/us101-0820am-0835am.txt',
        './raw_ngsim/i80-0400-0415.txt',
        './raw_ngsim/i80-0500-0515.txt',
        './raw_ngsim/i80-0515-0530.txt'
    ]
    
    # Instantiating the NGSIMDatasetProcessor class
    processor = NGSIMDatasetProcessor()
    
    # Running the process method to handle the dataset
    processor.process(dataset_to_use=6, file_paths=file_paths)
