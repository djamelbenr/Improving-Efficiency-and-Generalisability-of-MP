import math
import logging
import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.utils import check_array

# Initialize network parameters with Xavier uniform initialization for convolutional layers
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)

# Initialize logging with enhanced format and console streaming
def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=getattr(logging, level, logging.INFO),
        format='[%(levelname)s %(asctime)s] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

# Quintic spline function defined for trajectory fitting
def quintic_spline(x, z, a, b, c, d, e):
    return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5

# Fit the trajectory using a quintic spline with current location fixed
def fitting_traj_by_qs(x, y):
    return curve_fit(
        quintic_spline, x, y,
        bounds=([y[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], 
                [y[0] + 1e-6, np.inf, np.inf, np.inf, np.inf, np.inf])
    )[0]

# Custom activation function for output layer (Graves, 2015)
def outputActivation(x, displacement=True):
    if displacement:
        x[:, :, 0:2] = torch.cumsum(x[:, :, 0:2], dim=0)
    
    sigX = torch.exp(x[:, :, 2:3])  # Exponential to ensure positive sigma values
    sigY = torch.exp(x[:, :, 3:4])
    rho = torch.tanh(x[:, :, 4:5])  # Tanh to ensure rho is between -1 and 1
    
    return torch.cat([x[:, :, 0:2], sigX, sigY, rho], dim=2)

# Negative Log-Likelihood loss with masking
def maskedNLL(y_pred, y_gt, mask):
    muX, muY = y_pred[:, :, 0], y_pred[:, :, 1]
    sigX, sigY = y_pred[:, :, 2], y_pred[:, :, 3]
    rho = y_pred[:, :, 4]

    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x, y = y_gt[:, :, 0], y_gt[:, :, 1]

    nll = 0.5 * ohr**2 * (sigX**2 * (x - muX)**2 + sigY**2 * (y - muY)**2 - 2 * rho * sigX * sigY * (x - muX) * (y - muY)) \
          - torch.log(sigX * sigY * ohr) + math.log(2 * math.pi)
    
    loss = torch.sum(nll * mask) / torch.sum(mask)
    return loss

# Negative Log-Likelihood loss function for testing with maneuvers and masking
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                  num_lat_classes=3, num_lon_classes=2,
                  use_maneuvers=True, avg_along_time=False, separately=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes, device=fut_pred.device)
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = (lat_pred[:, l] * lon_pred[:, k]).unsqueeze(1).expand_as(fut_pred[k * num_lat_classes + l][:, :, 0:2])
                y_pred = fut_pred[k * num_lat_classes + l]
                muX, muY = y_pred[:, :, 0], y_pred[:, :, 1]
                sigX, sigY = y_pred[:, :, 2], y_pred[:, :, 3]
                rho = y_pred[:, :, 4]

                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x, y = fut[:, :, 0], fut[:, :, 1]

                nll = -(0.5 * ohr**2 * (sigX**2 * (x - muX)**2 + sigY**2 * (y - muY)**2 - 
                                         2 * rho * sigX * sigY * (x - muX) * (y - muY)) - 
                        torch.log(sigX * sigY * ohr) + math.log(2 * math.pi))
                
                acc[:, :, k * num_lat_classes + l] = nll + torch.log(wts)
        
        acc = -logsumexp(acc, dim=2)
        acc = acc * op_mask[:, :, 0]

        if avg_along_time:
            return torch.sum(acc) / torch.sum(op_mask[:, :, 0])
        if separately:
            return acc, op_mask[:, :, 0]
        return torch.sum(acc, dim=1), torch.sum(op_mask[:, :, 0], dim=1)

    acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1, device=fut_pred.device)
    y_pred = fut_pred
    muX, muY = y_pred[:, :, 0], y_pred[:, :, 1]
    sigX, sigY = y_pred[:, :, 2], y_pred[:, :, 3]
    rho = y_pred[:, :, 4]

    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x, y = fut[:, :, 0], fut[:, :, 1]

    nll = 0.5 * ohr**2 * (sigX**2 * (x - muX)**2 + sigY**2 * (y - muY)**2 - 
                          2 * rho * sigX * sigY * (x - muX) * (y - muY)) - \
          torch.log(sigX * sigY * ohr) + math.log(2 * math.pi)
    
    acc[:, :, 0] = nll
    acc = acc * op_mask[:, :, 0:1]

    if avg_along_time:
        return torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
    if separately:
        return acc[:, :, 0], op_mask[:, :, 0]
    return torch.sum(acc[:, :, 0], dim=1), torch.sum(op_mask[:, :, 0], dim=1)

# Mean Squared Error (MSE) loss with masking
def maskedMSE(y_pred, y_gt, mask):
    muX, muY = y_pred[:, :, 0], y_pred[:, :, 1]
    x, y = y_gt[:, :, 0], y_gt[:, :, 1]

    mse = (x - muX)**2 + (y - muY)**2
    mse = mse.unsqueeze(-1).expand_as(mask) * mask

    return torch.sum(mse) / torch.sum(mask)

# MSE loss function for testing with optional separation
def masked_mse_test(y_pred, y_gt, mask, separately=False):
    muX, muY = y_pred[:, :, 0], y_pred[:, :, 1]
    x, y = y_gt[:, :, 0], y_gt[:, :, 1]

    mse = (y - muY)**2 + (x - muX)**2
    mse = mse.unsqueeze(-1).expand_as(mask) * mask

    if separately:
        return mse[:, :, 0], mask[:, :, 0]
    return torch.sum(mse[:, :, 0], dim=1), torch.sum(mask[:, :, 0], dim=1)

# Log-sum-exp function with advanced handling
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0

    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    
    return outputs.squeeze(dim) if not keepdim else outputs
