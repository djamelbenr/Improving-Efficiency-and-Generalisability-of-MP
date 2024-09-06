function final_traj = nbhVehFunc(traj, tracks)
%% This function modifies the information saved in 'traj'
%  1. Ensures neighbour vehicles (targets for prediction) exist in 'tracks' with at least 3s of history.
%  2. Ensures the ego vehicle is included in the grid of its prediction targets (accounting for overlapping cases).
%  3. Ensures sub-neighbours of each target are included in 'tracks'.

numRows = size(traj, 1);            % Get the number of rows in the trajectory data.
numCols = size(traj, 2);            % Get the number of columns in the trajectory data.
maxVehicleID = size(tracks, 2);     % Get the maximum vehicle ID from the tracks data.
currentRow = 0;                     % Initialize a variable to track the row position for saving valid rows.
updated_traj = single(zeros(size(traj)));  % Create an empty matrix to store the updated trajectory data.

fprintf('nbhVehFunc processing: ')    % Print message indicating the start of processing.

% Loop through each row of the trajectory data.
for rowIndex = 1:numRows
    datasetID = traj(rowIndex, 1);    % Extract the dataset ID for the current row.
    egoVehicleID = traj(rowIndex, 2); % Extract the ego vehicle's ID for the current row.
    frameID = traj(rowIndex, 3);      % Extract the frame ID for the current row.
    keepRow = 1;                      % Flag to indicate whether the current row should be kept (1 = keep, 0 = discard).
    
    % Extract non-zero neighbour vehicle IDs from the current row.
    neighbourIDs = nonzeros(traj(rowIndex, 14:numCols));  
    if isempty(neighbourIDs)          % Check if there are no neighbours.
        continue                      % Skip the row if no neighbours are found.
    elseif any(neighbourIDs > maxVehicleID)  % Check if any neighbour ID exceeds the max vehicle ID.
        continue                      % Skip the row if an invalid neighbour ID is found.
    end
    
    % Iterate through all neighbour vehicles.
    for neighbourIndex = 1:length(neighbourIDs)
        neighbourID = neighbourIDs(neighbourIndex);  % Get the current neighbour ID.
        
        % Check if the neighbour exists in 'tracks' for the given dataset ID.
        if isempty(tracks{datasetID, neighbourID})
            keepRow = 0;              % Set flag to 0 if the neighbour doesn't exist in 'tracks'.
            break                     % Exit the loop as this row should be discarded.
        end
        
        % Check if the neighbour has more than 30 frames of history.
        historyCheckFrame = tracks{datasetID, neighbourID}(1, 31);  
        if frameID < historyCheckFrame  % Ensure the frame ID is greater than or equal to the 31st frame.
            keepRow = 0;              % Set flag to 0 if the neighbour doesn't have enough history.
            break                     % Exit the loop as this row should be discarded.
        end
        
        % Extract sub-neighbours of the current neighbour at the given frame ID.
        subNeighbours = nonzeros(tracks{datasetID, neighbourID}(12:(numCols-2), find(tracks{datasetID, neighbourID}(1, :) == frameID)));
        if isempty(subNeighbours)      % Check if there are no sub-neighbours.
            keepRow = 0;              % Set flag to 0 if no sub-neighbours are found.
            break                     % Exit the loop as this row should be discarded.
        elseif any(subNeighbours > maxVehicleID)  % Check if any sub-neighbour ID exceeds the max vehicle ID.
            keepRow = 0;              % Set flag to 0 if an invalid sub-neighbour ID is found.
            break                     % Exit the loop as this row should be discarded.
        elseif all(subNeighbours ~= egoVehicleID)  % Check if the ego vehicle is missing from the sub-neighbours.
            keepRow = 0;              % Set flag to 0 if the ego vehicle is not found in sub-neighbours.
            break                     % Exit the loop as this row should be discarded.
        end
        
        % Ensure all sub-neighbours exist in 'tracks'.
        for subIndex = 1:length(subNeighbours)
            if isempty(tracks{datasetID, subNeighbours(subIndex)})  % Check if the sub-neighbour exists in 'tracks'.
                keepRow = 0;          % Set flag to 0 if any sub-neighbour doesn't exist in 'tracks'.
                break                 % Exit the loop as this row should be discarded.
            end
        end
        
        if ~keepRow                   % Check if the current row should be discarded based on previous checks.
            break                     % Exit the loop if any condition fails.
        end
    end
    
    % Retain rows that pass all conditions.
    if keepRow                        % If the row passes all checks.
        currentRow = currentRow + 1;  % Increment the row counter for saving valid rows.
        updated_traj(currentRow, :) = traj(rowIndex, :);  % Save the valid row in 'updated_traj'.
    end
    
    % Print progress every 100,000 rows.
    if mod(rowIndex, 100000) == 0     % Check if the current row index is a multiple of 100,000.
        fprintf('%.2f ... ', rowIndex / numRows);  % Print the progress as a percentage.
    end
end

% Return the filtered trajectory data, excluding rows that were not filled (with zeros).
final_traj = updated_traj(updated_traj(:, 1) ~= 0, :);  %% Cleanup - remove rows with a zero in the first column (invalid rows).
fprintf('\nOriginal #rows: %d ===>>> Filtered #rows: %d \n\n', size(traj, 1), size(final_traj, 1));  % Print the number of rows before and after filtering.

end
