import math
import logging
import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.utils import check_array

## Initialise network parameters (weights)
def weights_init(m):
    # Check if the module is a 2D convolution layer
    if isinstance(m, torch.nn.Conv2d):
        # Initialise the weights using Xavier uniform distribution
        torch.nn.init.xavier_uniform_(m.weight)
        # If the layer has bias terms, initialise them to 0.1
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)

## Initialise logging settings for training output
def initLogging(log_file: str, level: str = "INFO"):
    # Set up the logging configuration with the specified log file and level
    logging.basicConfig(filename=log_file, filemode='a',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    # Add a stream handler to display logs in the console as well
    logging.getLogger().addHandler(logging.StreamHandler())

## Define a quintic spline function, which is a polynomial of degree 5
def quintic_spline(x, z, a, b, c, d, e):
    # This function represents a quintic equation with coefficients z, a, b, c, d, and e
    return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5

## Fit a trajectory using a quintic spline while keeping the current location fixed
def fitting_traj_by_qs(x, y):
    # Use curve fitting to find the best parameters for the quintic spline that fits the data points (x, y)
    param, loss = curve_fit(quintic_spline, x, y,
        bounds=([y[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [y[0]+1e-6, np.inf, np.inf, np.inf, np.inf, np.inf]))
    # Return the fitted parameters
    return param

## Custom activation function for the output layer (based on Graves, 2015)
def outputActivation(x, displacement=True):
    # If displacement is True, the first two columns represent displacements
    if displacement:
        # Calculate cumulative sum along the time dimension for x and y displacements
        x[:, :, 0:2] = torch.stack([torch.sum(x[0:i, :, 0:2], dim=0) for i in range(1, x.shape[0] + 1)], 0)
    
    # Extract the five parameters that represent the Gaussian distribution: muX, muY, sigX, sigY, rho
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    
    # Apply exponential activation to ensure sigX and sigY are positive
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    
    # Apply tanh activation to restrict rho to the range (-1, 1)
    rho = torch.tanh(rho)
    
    # Concatenate the parameters back into a single tensor
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    
    # Return the transformed output
    return out

## Masked negative log-likelihood loss function for training
def maskedNLL(y_pred, y_gt, mask):
    # Initialise a tensor to store the accumulation of the loss
    acc = torch.zeros_like(mask)
    
    # Extract predicted parameters for the Gaussian distribution: muX, muY, sigX, sigY, rho
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    
    # Calculate the inverse of the standard deviation of the Gaussian distribution
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    
    # Extract ground truth values for x and y
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    
    # Calculate the negative log-likelihood based on the Gaussian distribution
    out = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho *
        torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) \
        + torch.log(torch.tensor(2 * math.pi))
    
    # Store the calculated loss in the accumulator
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    
    # Multiply the accumulated loss by the mask to ignore irrelevant parts
    acc = acc * mask
    
    # Calculate the final loss value by summing over the masked elements
    lossVal = torch.sum(acc) / torch.sum(mask)
    
    # Return the calculated loss
    return lossVal

## Masked NLL loss function for testing with lateral and longitudinal maneuvers
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                  num_lat_classes=3, num_lon_classes=2,
                  use_maneuvers=True, avg_along_time=False, separately=False):
    # Use maneuvers (lat_pred and lon_pred) if specified
    if use_maneuvers:
        # Initialise an accumulator for the loss with the appropriate shape
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).cuda()
        count = 0

        # Iterate over the longitudinal and lateral behaviour classes
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                # Calculate the weights based on the predicted lateral and longitudinal behaviours
                wts = lat_pred[:, l] * lon_pred[:, k]
                wts = wts.repeat(len(fut_pred[0]), 1)

                # Extract the predicted future trajectory for the current combination of behaviours
                y_pred = fut_pred[k * num_lat_classes + l]
                y_gt = fut
                
                # Extract Gaussian distribution parameters from the prediction
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                
                # Calculate the inverse of the correlation coefficient
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                
                # Extract the ground truth future trajectory
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                
                # Calculate the negative log-likelihood for the Gaussian distribution
                out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
                      - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
                      + torch.log(torch.tensor(2 * math.pi)))
                
                # Accumulate the weighted loss
                acc[:, :, count] = out + torch.log(wts)
                count += 1

        # Calculate the negative log-sum-exp of the accumulated loss
        acc = -logsumexp(acc, dim=2)
        
        # Apply the output mask to ignore irrelevant parts
        acc = acc * op_mask[:, :, 0]

        # Calculate the final loss based on the time-averaging or overall loss
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            if separately:
                lossVal = acc
                counts = op_mask[:, :, 0]
                return lossVal, counts
            else:
                lossVal = torch.sum(acc, dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts
    else:
        # If not using maneuvers, calculate loss without lateral/longitudinal behaviour
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        
        # Extract Gaussian distribution parameters
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        
        # Calculate the inverse of the correlation coefficient
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        
        # Extract the ground truth future trajectory
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        
        # Calculate the negative log-likelihood for the Gaussian distribution
        out = +(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
              - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
              + torch.log(torch.tensor(2 * math.pi)))
        
        # Store the calculated loss in the accumulator
        acc[:, :, 0] = out
        
        # Apply the output mask
        acc = acc * op_mask[:, :, 0:1]

        # Calculate the final loss value
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            if separately:
                lossVal = acc[:, :, 0]
                counts = op_mask[:, :, 0]
                return lossVal, counts
            else:
                lossVal = torch.sum(acc[:, :, 0], dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts
