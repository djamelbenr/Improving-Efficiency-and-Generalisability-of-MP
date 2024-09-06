function final_traj = targVehFunc(traj, tracks)
%% This function filters target vehicles around the center vehicles
% Now the 25x5 grid is formed by the relative position of the vehicles
% This function selects qualified vehicles as prediction targets using the 7-vehicles method.
% The resulting target number is <= 8 (7+1 following) for each row of traj.

gridRows = 25;               % Define the number of rows in the grid.
gridCols = 5;                % Define the number of columns in the grid.
nbrStartIdx = 14;            % Index where neighbour vehicle IDs start in traj.
nbrEndIdx = 13 + gridRows * gridCols;  % Index where neighbour vehicle IDs end in traj.
dataNum = size(traj, 1);     % Number of rows in traj (number of data points).

final_traj = traj;           % Initialize final_traj as a copy of traj.
fprintf('targVehFunc processing: ')  % Print message to indicate processing has started.

% Loop through each row of traj.
for i = 1:dataNum
    targsGrid = traj(i, nbrStartIdx:nbrEndIdx);  % Extract the grid of neighbouring vehicles.
    targsVeh = nonzeros(targsGrid);              % Get non-zero (valid) target vehicle IDs.
    targsNum = length(targsVeh);                 % Number of valid target vehicles.
    if targsNum < 2                              % Skip if there are fewer than 2 target vehicles.
        continue;
    end
    dsId = traj(i, 1);       % Extract dataset ID from traj.
    vehId = traj(i, 2);      % Extract ego vehicle ID from traj.
    frameId = traj(i, 3);    % Extract frame ID from traj.
    centX = traj(i, 4);      % Extract ego vehicle's X coordinate from traj.
    centY = traj(i, 5);      % Extract ego vehicle's Y coordinate from traj.
    laneId = traj(i, 6);     % Extract lane ID from traj.
    
    %% Retrieve lane ID and X, Y locations of all target vehicles.
    targsInfo = zeros(4, targsNum);   % Initialize a matrix to store target vehicle info.
    targsInfo(1, :) = targsVeh;       % Store the target vehicle IDs in the first row of targsInfo.
    for j = 1:targsNum
        % Store X, Y positions and lane ID of each target vehicle in targsInfo.
        targsInfo(2:4, j) = tracks{dsId, targsVeh(j)}(2:4, tracks{dsId, targsVeh(j)}(1,:) == frameId);
    end
    
    %% Classify vehicles into different areas (preceding, following, left, right lanes).
    % Transform target locations to relative positions (relative to ego vehicle).
    targsInfo(2, :) = targsInfo(2, :) - centX;  % X relative to ego vehicle.
    targsInfo(3, :) = targsInfo(3, :) - centY;  % Y relative to ego vehicle.
    precedingInfo = targsInfo(:, (targsInfo(4,:) == laneId) & (targsInfo(3,:) > 0));  % Vehicles ahead in the same lane.
    followingInfo = targsInfo(:, (targsInfo(4,:) == laneId) & (targsInfo(3,:) < 0));  % Vehicles behind in the same lane.
    leftLaneInfo  = targsInfo(:, targsInfo(4,:) < laneId);  % Vehicles in the left lane.
    rightLaneInfo = targsInfo(:, targsInfo(4,:) > laneId);  % Vehicles in the right lane.
    
    %% Pick the qualified vehicles based on their position.
    qualVehs = zeros(1, 8);   % Initialize array to store IDs of qualified vehicles.
    
    % Select the closest preceding vehicle in the same lane.
    if size(precedingInfo, 2) == 1
        qualVehs(1) = precedingInfo(1);
    elseif size(precedingInfo, 2) > 1
        [~, index] = min(precedingInfo(3,:));  % Find the closest vehicle ahead.
        qualVehs(1) = precedingInfo(1, index);  % Store the closest vehicle's ID.
    end
    
    % Select the closest following vehicle in the same lane.
    if size(followingInfo, 2) == 1
        qualVehs(2) = followingInfo(1);
    elseif size(followingInfo, 2) > 1
        [~, index] = max(followingInfo(3,:));  % Find the closest vehicle behind.
        qualVehs(2) = followingInfo(1, index);  % Store the closest vehicle's ID.
    end
    
    % Select vehicles in the left lane.
    leftVehNum = size(leftLaneInfo, 2);   % Number of vehicles in the left lane.
    if leftVehNum == 1
        qualVehs(3) = leftLaneInfo(1);
    elseif leftVehNum == 2
        qualVehs(3:4) = leftLaneInfo(1, :);
    elseif leftVehNum > 2
        [~, index] = min(leftLaneInfo(2,:).^2 + leftLaneInfo(3,:).^2);  % Find the closest vehicle in the left lane.
        qualVehs(3) = leftLaneInfo(1, index);   % Store the closest vehicle's ID.
        leftCentVehY = leftLaneInfo(3, index);  % Y-coordinate of the closest vehicle.
        
        % Select the vehicle ahead in the left lane.
        leftFrontVehs = leftLaneInfo(:, leftLaneInfo(3,:) > leftCentVehY);
        if size(leftFrontVehs, 2) > 0
            [~, index] = min(leftFrontVehs(3,:));  % Find the closest vehicle ahead.
            qualVehs(4) = leftFrontVehs(1, index);  % Store the closest vehicle's ID.
        end
        
        % Select the vehicle behind in the left lane.
        leftBackVehs = leftLaneInfo(:, leftLaneInfo(3,:) < leftCentVehY);
        if size(leftBackVehs, 2) > 0
            [~, index] = max(leftBackVehs(3,:));  % Find the closest vehicle behind.
            qualVehs(5) = leftBackVehs(1, index);  % Store the closest vehicle's ID.
        end
    end
    
    % Select vehicles in the right lane.
    rightVehNum = size(rightLaneInfo, 2);   % Number of vehicles in the right lane.
    if rightVehNum == 1
        qualVehs(6) = rightLaneInfo(1);
    elseif rightVehNum == 2
        qualVehs(6:7) = rightLaneInfo(1, :);
    elseif size(rightLaneInfo, 2) > 2
        [~, index] = min(rightLaneInfo(2,:).^2 + rightLaneInfo(3,:).^2);  % Find the closest vehicle in the right lane.
        qualVehs(6) = rightLaneInfo(1, index);   % Store the closest vehicle's ID.
        rightCentVehY = rightLaneInfo(3, index);  % Y-coordinate of the closest vehicle.
        
        % Select the vehicle ahead in the right lane.
        rightFrontVehs = rightLaneInfo(:, rightLaneInfo(3,:) > rightCentVehY);
        if size(rightFrontVehs, 2) > 0
            [~, index] = min(rightFrontVehs(3,:));  % Find the closest vehicle ahead.
            qualVehs(7) = rightFrontVehs(1, index);  % Store the closest vehicle's ID.
        end
        
        % Select the vehicle behind in the right lane.
        rightBackVehs = rightLaneInfo(:, rightLaneInfo(3,:) < rightCentVehY);
        if size(rightBackVehs, 2) > 0
            [~, index] = max(rightBackVehs(3,:));  % Find the closest vehicle behind.
            qualVehs(8) = rightBackVehs(1, index);  % Store the closest vehicle's ID.
        end
    end
    
    %% Filter out target vehicles not needed.
    for k = nbrStartIdx:nbrEndIdx
        if traj(i, k) > 0 && ~any(qualVehs == traj(i, k))  % If the vehicle ID is not in qualVehs, set it to 0.
            final_traj(i, k) = 0;
        end
    end
    
    % Print progress every 100,000 rows.
    if mod(i, 100000) == 0
        fprintf('%.2f ... ', i / dataNum);
    end
end

% Print summary statistics of how many rows and targets remain after filtering.
fprintf('\nOriginal #rows: %d ===>>> Filtered #rows: %d\n', size(traj, 1), size(final_traj, 1));
fprintf('Original #targets: %d ===>>> Filtered #targets: %d\n', nnz(traj(:, nbrStartIdx:nbrEndIdx)), nnz(final_traj(:, nbrStartIdx:nbrEndIdx)));
end

