function final_traj = nbhCheckerFunc(traj, tracks)
%% This function modifies the information saved in 'traj'
%  1. Ensures neighbour vehicles (targets for prediction) exist in 'tracks' with at least 3s of history.
%  2. Ensures the ego vehicle is included in the grid of its prediction targets (accounting for overlapping cases).
%  3. Ensures sub-neighbours of each target are included in 'tracks'.

numRows = size(traj, 1);
numCols = size(traj, 2);
maxVehicleID = size(tracks, 2);
currentRow = 0;
updated_traj = single(zeros(size(traj)));

fprintf('nbhCheckerFunc processing: ')

% Loop through each row of the trajectory data.
for rowIndex = 1:numRows
    datasetID = traj(rowIndex, 1);  % The dataset ID for this row.
    egoVehicleID = traj(rowIndex, 2);  % The ego vehicle's ID.
    frameID = traj(rowIndex, 3);  % The frame ID for this row.
    keepRow = 1;
    
    % Extract non-zero neighbour vehicle IDs.
    neighbourIDs = nonzeros(traj(rowIndex, 14:numCols));
    if isempty(neighbourIDs)
        continue
    elseif any(neighbourIDs > maxVehicleID)
        % Skip if a neighbour ID exceeds the maximum vehicle ID.
        continue
    end
    
    % Iterate through all neighbour vehicles.
    for neighbourIndex = 1:length(neighbourIDs)
        neighbourID = neighbourIDs(neighbourIndex);
        
        % Ensure the neighbour exists in 'tracks'.
        if isempty(tracks{datasetID, neighbourID})
            keepRow = 0;
            break
        end
        
        % Check if the neighbour has more than 30 frames of history.
        historyCheckFrame = tracks{datasetID, neighbourID}(1, 31);
        if frameID < historyCheckFrame
            keepRow = 0;
            break
        end
        
        % Ensure the ego vehicle exists in the sub-neighbours of the target.
        subNeighbours = nonzeros(tracks{datasetID, neighbourID}(12:(numCols-2), find(tracks{datasetID, neighbourID}(1, :) == frameID)));
        if isempty(subNeighbours)
            keepRow = 0;
            break
        elseif any(subNeighbours > maxVehicleID)
            % Skip if a sub-neighbour ID exceeds the max vehicle ID.
            keepRow = 0;
            break
        elseif all(subNeighbours ~= egoVehicleID)
            % Ensure the ego vehicle is present in the sub-neighbours.
            keepRow = 0;
            break
        end
        
        % Ensure all sub-neighbours exist in 'tracks'.
        for subIndex = 1:length(subNeighbours)
            if isempty(tracks{datasetID, subNeighbours(subIndex)})
                keepRow = 0;
                break
            end
        end
        
        if ~keepRow
            break
        end
    end
    
    % Retain rows that meet all conditions.
    if keepRow
        currentRow = currentRow + 1;
        updated_traj(currentRow, :) = traj(rowIndex, :);
    end
    
    % Print progress every 100,000 rows.
    if mod(rowIndex, 100000) == 0
        fprintf('%.2f ... ', rowIndex / numRows);
    end
end

% Return the filtered trajectory data.
final_traj = updated_traj(updated_traj(:, 1) ~= 0, :);
fprintf('\nOriginal #rows: %d ===>>> Filtered #rows: %d \n\n', size(traj, 1), size(final_traj, 1));

end

