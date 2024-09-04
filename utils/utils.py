import math  
import logging  
import numpy as np  
import torch  
from scipy.optimize import curve_fit  
from sklearn.utils import check_array 

# Initialize network parameters with Xavier uniform initialization for convolutional layers
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):  # Check if the layer is a Conv2D layer
        torch.nn.init.xavier_uniform_(m.weight)  # Apply Xavier uniform initialization to the weights
        if m.bias is not None:  # Check if the layer has a bias
            torch.nn.init.constant_(m.bias, 0.1)  # Set the bias to a constant value of 0.1

# Initialize logging with enhanced format and console streaming
def initLogging(log_file: str, level: str = "INFO"):
    # Set up logging to write to a file with specified level and format
    logging.basicConfig(
        filename=log_file,
        filemode='a',  # Append to the log file
        level=getattr(logging, level, logging.INFO),  # Set logging level based on input
        format='[%(levelname)s %(asctime)s] %(message)s',  # Format for log messages
        datefmt='%m-%d %H:%M:%S'  # Date format for log messages
    )
    # Set up a console handler to also print logs to the console
    console = logging.StreamHandler()  
    console.setLevel(logging.INFO)  # Set log level for console output
    logging.getLogger().addHandler(console)  # Add console handler to the logger

# Quintic spline function defined for trajectory fitting
def quintic_spline(x, z, a, b, c, d, e):
    # Return the quintic polynomial: z + ax + bx^2 + cx^3 + dx^4 + ex^5
    return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5

# Fit the trajectory using a quintic spline with current location fixed
def fitting_traj_by_qs(x, y):
    # Fit the quintic spline to the data (x, y), bounding the first value of y to a small range
    return curve_fit(
        quintic_spline, x, y,
        bounds=([y[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],  # Lower bounds for parameters
                [y[0] + 1e-6, np.inf, np.inf, np.inf, np.inf, np.inf])  # Upper bounds for parameters
    )[0]  # Return the fitted parameters

# Custom activation function for output layer (Graves, 2015)
def outputActivation(x, displacement=True):
    if displacement:  # If displacement is True, the first two columns represent displacements
        # Calculate the cumulative sum along the time axis for x and y
        x[:, :, 0:2] = torch.cumsum(x[:, :, 0:2], dim=0)
    
    # Exponential function ensures positive values for sigX and sigY
    sigX = torch.exp(x[:, :, 2:3])  # Exponential to ensure positive sigma values
    sigY = torch.exp(x[:, :, 3:4])
    
    # Tanh function ensures rho stays between -1 and 1
    rho = torch.tanh(x[:, :, 4:5])  # Tanh to ensure rho is between -1 and 1
    
    # Concatenate muX, muY, sigX, sigY, and rho into the final output
    return torch.cat([x[:, :, 0:2], sigX, sigY, rho], dim=2)

# Negative Log-Likelihood loss with masking
def maskedNLL(y_pred, y_gt, mask):
    # Extract predicted muX, muY, sigX, sigY, and rho from the predictions
    muX, muY = y_pred[:, :, 0], y_pred[:, :, 1]
    sigX, sigY = y_pred[:, :, 2], y_pred[:, :, 3]
    rho = y_pred[:, :, 4]

    # Calculate the precision factor (1 / sqrt(1 - rho^2))
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    
    # Extract ground truth x and y coordinates
    x, y = y_gt[:, :, 0], y_gt[:, :, 1]

    # Compute the negative log-likelihood for the bivariate Gaussian distribution
    nll = 0.5 * ohr**2 * (sigX**2 * (x - muX)**2 + sigY**2 * (y - muY)**2 - 2 * rho * sigX * sigY * (x - muX) * (y - muY)) \
          - torch.log(sigX * sigY * ohr) + math.log(2 * math.pi)
    
    # Compute the masked loss (only for relevant data points)
    loss = torch.sum(nll * mask) / torch.sum(mask)
    return loss  # Return the computed loss

# Negative Log-Likelihood loss function for testing with maneuvers and masking
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                  num_lat_classes=3, num_lon_classes=2,
                  use_maneuvers=True, avg_along_time=False, separately=False):
    if use_maneuvers:  # If using lateral and longitudinal maneuvers
        # Initialize a tensor to accumulate the loss values
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).cuda()
        count = 0
        
        # Iterate over all combinations of lateral and longitudinal classes
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                # Calculate the weights from lateral and longitudinal predictions
                wts = lat_pred[:, l] * lon_pred[:, k]
                wts = wts.repeat(len(fut_pred[0]), 1)
                
                # Extract the predicted future trajectory for this maneuver
                y_pred = fut_pred[k * num_lat_classes + l]
                y_gt = fut  # Ground truth future trajectory
                
                # Extract the predicted parameters from the future prediction
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                
                # Calculate the precision factor
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                
                # Extract ground truth x and y coordinates
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                
                # Compute the negative log-likelihood for the bivariate Gaussian
                out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
                      - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
                      + torch.log(torch.tensor(2 * math.pi)))
                
                # Accumulate the weighted loss
                acc[:, :, count] = out + torch.log(wts)
                count += 1  # Increment the maneuver count
        
        # Compute the log-sum-exp over the accumulated losses for each maneuver
        acc = -logsumexp(acc, dim=2)
        
        # Apply the output mask to ignore irrelevant parts
        acc = acc * op_mask[:, :, 0]
        
        # If averaging along the time axis
        if avg_along_time:
            # Calculate the average loss along the time dimension
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal  # Return the loss value
        else:
            if separately:  # If calculating loss separately for each timestep
                lossVal = acc
                counts = op_mask[:, :, 0]
                return lossVal, counts  # Return the loss and counts separately
            else:
                # Sum the loss along the time axis for the entire sequence
                lossVal = torch.sum(acc, dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts  # Return the loss and counts for the whole sequence
    else:  # If not using maneuvers
        # Initialize a tensor to accumulate the loss
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred  # Predicted future trajectory
        y_gt = fut  # Ground truth future trajectory
        
        # Extract predicted parameters from the prediction
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        
        # Calculate the precision factor
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        
        # Extract ground truth x and y coordinates
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        
        # Compute the negative log-likelihood for the bivariate Gaussian
        out = +(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
              - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
              + torch.log(torch.tensor(2 * math.pi)))
        
        # Accumulate the loss
        acc[:, :, 0] = out
        
        # Apply the mask to the accumulated loss
        acc = acc * op_mask[:, :, 0:1]
        
        # If averaging along time
        if avg_along_time:
            # Calculate the average loss
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal  # Return the loss
        else:
            if separately:  # If calculating the loss separately for each timestep
                lossVal = acc[:, :, 0]
                counts = op_mask[:, :, 0]
                return lossVal, counts  # Return the loss and counts separately
            else:
                # Sum the loss along the time axis for the entire sequence
                lossVal = torch.sum(acc[:, :, 0], dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts  # Return the loss and counts for the whole sequence

# Mean Squared Error (MSE) loss with masking
def maskedMSE(y_pred, y_gt, mask):
    # Initialize a tensor to accumulate the loss
    acc = torch.zeros_like(mask)
    
    # Extract predicted and ground truth x and y coordinates
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    
    # Compute the squared error between predicted and ground truth values
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    
    # Accumulate the squared error in both dimensions
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    
    # Apply the mask to ignore irrelevant parts
    acc = acc * mask
    
    # Compute the masked MSE
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal  # Return the MSE loss

# MSE loss function for testing with optional separation
def maskedMSETest(y_pred, y_gt, mask, separately=False):
    # Initialize a tensor to accumulate the loss
    acc = torch.zeros_like(mask)
    
    # Extract predicted and ground truth x and y coordinates
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    
    # Compute the squared error between predicted and ground truth values
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    
    # Accumulate the squared error in both dimensions
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    
    # Apply the mask to ignore irrelevant parts
    acc = acc * mask
    
    # If returning loss separately for each timestep
    if separately:
        return acc[:, :, 0], mask[:, :, 0]
    else:
        # Sum the loss along the time axis for the entire sequence
        lossVal = torch.sum(acc[:, :, 0], dim=1)
        counts = torch.sum(mask[:, :, 0], dim=1)
        return lossVal, counts  # Return the loss and counts for the whole sequence

# Helper function for log-sum-exp calculation
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)  # Flatten inputs if no dimension is provided
        dim = 0
    
    # Compute the maximum value along the specified dimension
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    
    # Compute log-sum-exp: log(sum(exp(inputs - max)))
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    
    if not keepdim:  # If keepdim is False, remove the reduced dimension
        outputs = outputs.squeeze(dim)
    
    return outputs  # Return the log-sum-exp result
