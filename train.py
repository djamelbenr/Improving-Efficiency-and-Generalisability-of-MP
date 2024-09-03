import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import time
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as progress_bar  # Import tqdm with an alias

from modelTraj import TrajPred


from data4process import highwayTrajDataset
from utils import initLogging, maskedNLL, maskedMSE, maskedNLLTest
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
HDF5_USE_FILE_LOCKING='FALSE'


'''
--HOW TO RUN :

python train.py --name ngsim_demo --batch_size 64 --pretrain_epochs 5 --train_epochs 10 --train_set ../dataset/training_data.mat --val_set ../dataset/validation_data.mat

 % Features -- 1: Dataset ID, 2: Vehicle ID, 3: Frame Index,
    %             6: Local X, 7: Local Y, 15: Lane ID,
    %            10: Vehicle Length, 11: Vehicle Width, 12: Vehicle Class,
    %            13: Velocity (feet/s), 14: Acceleration (feet/s²).
    %%=== Longitudinal and Lateral locations, Velocity, and Acceleration 

    +------------+------------+------------+------------+------------+------------+
    |  param/row |      1     |      2     |    6/7     |     15     |     10     |
    +------------+------------+------------+------------+------------+------------+
    |            | Dataset ID | Vehicle ID | Local X/Y  |  Lane ID   | Vehicle    |
    |            |            |            |            |            | Length     |
    +------------+------------+------------+------------+------------+------------+
    +------------+------------+------------+------------+------------+------------+
    |  param/row |      11    |      12    |     13     |     14     |     --     |
    +------------+------------+------------+------------+------------+------------+
    |            | Vehicle    | Vehicle    | Velocity   | Accel-     |            |
    |            | Width      | Class      | (feet/s)   | eration    |            |
    |            |            |            |            | (feet/s²)  |            |
    +------------+------------+------------+------------+------------+------------+
    %-- Converted from Feet to Meters. 
    
'''


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train motion prediction model.")

    # General arguments
    parser.add_argument('--train_output_flag', action="store_false", help='Concatenate with true maneuver label (default: True)', default=True)
    parser.add_argument('--tensorboard', action='store_false', help='Use TensorBoard for logging (default: True)', default=True)

    # CUDA and module settings
    parser.add_argument('--use_cuda', action='store_false', help='Use CUDA (default: True)', default=True)
    parser.add_argument('--use_planning', action='store_false', help='Use planning module (default: False)', default=False)
    parser.add_argument('--use_fusion', action='store_false', help='Use target vehicle info fusion module (default: True)', default=True)

    # Training parameters
    parser.add_argument('--batch_size', type=int, help='Batch size for training (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='Learning rate (default: 1e-4)', default=0.001)
    parser.add_argument('--momentum', type=float, help='Momentum (default: 0.9)', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='Weight decay (default: 0.5)', default=0.5)

    # Input/Output settings
    parser.add_argument('--grid_size', type=int, nargs=2, help='Grid size in pixels (default: [25, 5])', default=[25, 5])

    # Sequence lengths
    parser.add_argument('--in_length', type=int, help='History sequence length (default: 16)', default=16)
    parser.add_argument('--out_length', type=int, help='Prediction sequence length (default: 25)', default=25)

    # Behavioral classes
    parser.add_argument('--num_lat_classes', type=int, help='Number of lateral behavior classes (default: 3)', default=3)
    parser.add_argument('--num_lon_classes', type=int, help='Number of longitudinal behavior classes (default: 2)', default=2)

    # Model parameters
    parser.add_argument('--temporal_embedding_size', type=int, help='Temporal embedding size (default: 32)', default=32)
    parser.add_argument('--encoder_size', type=int, help='LSTM encoder size (default: 64)', default=64)
    parser.add_argument('--decoder_size', type=int, help='LSTM decoder size (default: 128)', default=128)
    parser.add_argument('--soc_conv_depth', type=int, help='1st social convolution depth (default: 64)', default=64)
    parser.add_argument('--soc_conv2_depth', type=int, help='2nd social convolution depth (default: 16)', default=16)
    parser.add_argument('--dynamics_encoding_size', type=int, help='Vehicle dynamics embedding size (default: 32)', default=32)
    parser.add_argument('--social_context_size', type=int, help='Social context tensor embedding size (default: 80)', default=80)
    parser.add_argument('--fuse_enc_size', type=int, help='Feature fusion size (default: 112)', default=112)

    # File paths and other settings
    parser.add_argument('--name', type=str, help='Log name (default: "1")', default="1")
    parser.add_argument('--train_set', type=str, help='Path to training dataset')
    parser.add_argument('--val_set', type=str, help='Path to validation dataset')
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument('--pretrain_epochs', type=int, help='Pre-training epochs using MSE (default: 5)', default=5)
    parser.add_argument('--train_epochs', type=int, help='Training epochs using NLL (default: 10)', default=10)
    parser.add_argument('--IA_module', action='store_false', help='Use IA_module (default: True)', default=True)

    return parser.parse_args()

def setup_logging(log_path):
    """Initialize logging and create log directory if it doesn't exist."""
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path + 'train.log')

def initialize_model(args):
    """Initialize the Trajectory Prediction model and optimizer."""
    model = TrajPred(args)
    if args.use_cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, optimizer

def load_datasets(args):
    """Load training and validation datasets."""
    train_set = highwayTrajDataset(
        path=args.train_set,
        targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
        grid_size=args.grid_size,
        fit_plan_traj=False
    )

    val_set = highwayTrajDataset(
        path=args.val_set,
        targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
        grid_size=args.grid_size,
        fit_plan_traj=True
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_set.collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=val_set.collate_fn
    )

    return train_loader, val_loader

def train_epoch(epoch_num, model, optimizer, train_loader, args, logger):
    """Train the model for one epoch."""
    model.train()
    avg_loss_tr, avg_time_tr = 0, 0
    dataset_length = len(train_loader.dataset)  # Get the dataset length before wrapping with tqdm
    train_loader = progress_bar(train_loader, desc=f"Epoch {epoch_num + 1}/{args.pretrain_epochs + args.train_epochs}", dynamic_ncols=True)

    for i, data in enumerate(train_loader):
        start_time = time.time()
        # Ensure only tensors are sent to GPU
        inputs = [d.cuda() if isinstance(d, torch.Tensor) and args.use_cuda else d for d in data]

        nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = inputs

        # Forward pass
        fut_pred, lat_pred, lon_pred = model(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)

        # Compute loss
        if epoch_num < args.pretrain_epochs:
            loss = maskedMSE(fut_pred, targsFut, targsFutMask)
        else:
            loss = maskedNLL(fut_pred, targsFut, targsFutMask) + torch.nn.BCELoss()(lat_pred, lat_enc) + torch.nn.BCELoss()(lon_pred, lon_enc)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        # Logging and progress bar update
        avg_loss_tr += loss.item()
        avg_time_tr += time.time() - start_time

        if i % 100 == 99:
            eta = avg_time_tr / (i + 1) * (len(train_loader) - (i + 1))
            epoch_progress = (i + 1) * args.batch_size / dataset_length
            train_loader.set_postfix({
                'Avg Loss': f"{avg_loss_tr / (i + 1):.4f}",
                'Progress': f"{epoch_progress * 100:.2f}%",
                'ETA': f"{int(eta)}s"
            })
            logger.add_scalar("Loss/train", avg_loss_tr / (i + 1), epoch_progress + epoch_num)

def validate_epoch(epoch_num, model, val_loader, args, logger_val):
    """Validate the model after each epoch."""
    model.eval()
    avg_loss_val = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs = [d.cuda() if args.use_cuda else d for d in data]
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = inputs

            # Forward pass and loss computation
            fut_pred, lat_pred, lon_pred = model(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)
            loss = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, avg_along_time=True)

            avg_loss_val += loss.item()

            if i == 19:
                logger_val.add_scalar("Loss/val", avg_loss_val / 20, epoch_num)
                break

    logging.info(f"Validation | Avg Loss: {avg_loss_val / 20:.2f}")

def train_my_model():
    args = parse_arguments()

    log_path = f"./trained_models/{args.name}/"
    setup_logging(log_path)

    logger, logger_val = None, None
    if args.tensorboard:
        logger = SummaryWriter(log_path + f'train-pre{args.pretrain_epochs}-nll{args.train_epochs}')
        logger_val = SummaryWriter(log_path + f'validation-pre{args.pretrain_epochs}-nll{args.train_epochs}')

    model, optimizer = initialize_model(args)
    train_loader, val_loader = load_datasets(args)

    logging.info(f"Starting training: {args.name}")
    logging.info(f"Batch size: {args.batch_size} | Learning rate: {args.learning_rate}")
    logging.info(f"Using Planning Module: {args.use_planning} | Using Fusion Module: {args.use_fusion}")

    for epoch_num in range(args.pretrain_epochs + args.train_epochs):
        train_epoch(epoch_num, model, optimizer, train_loader, args, logger)
        validate_epoch(epoch_num, model, val_loader, args, logger_val)

        model_save_path = log_path + f"{args.name}-pre{args.pretrain_epochs if epoch_num < args.pretrain_epochs else 0}-nll{epoch_num}.tar"
        torch.save(model.state_dict(), model_save_path)

    final_model_save_path = log_path + f"{args.name}.tar"
    torch.save(model.state_dict(), final_model_save_path)
    logging.info(f"Model saved at {final_model_save_path}")

if __name__ == '__main__':
    train_my_model()

