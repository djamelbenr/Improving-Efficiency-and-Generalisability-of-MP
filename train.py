import os
import time
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as progress_bar  # Import tqdm with an alias
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from modelTraj import TrajPred
from data4process import TrajDataset
from utils import initLogging, maskedNLL, maskedMSE, maskedNLLTest

# Ensuring correct environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
HDF5_USE_FILE_LOCKING = 'FALSE'


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
        model = model.cuda()  # Move model to GPU if CUDA is enabled
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


def plot_and_save_loss(epoch_num, train_loss_history, val_loss_history, log_path):
    """Plot and save the training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Over Epochs (Epoch {epoch_num + 1})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_path}/loss_curve_epoch_{epoch_num + 1}.png")
    plt.close()


def train_epoch(epoch_num, model, optimizer, train_loader, args, logger, train_loss_history):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    avg_time_per_batch = 0

    with progress_bar(total=len(train_loader), dynamic_ncols=True, desc=f"Epoch {epoch_num + 1}/{args.pretrain_epochs + args.train_epochs}") as pbar:
        for batch_idx, data in enumerate(train_loader):
            start_time = time.time()

            # Ensure all data tensors are moved to the correct device
            inputs = [d.cuda() if isinstance(d, torch.Tensor) and args.use_cuda else d for d in data]
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = inputs

            # Forward pass
            fut_pred, lat_pred, lon_pred = model(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)

            # Compute loss
            if epoch_num < args.pretrain_epochs:
                loss = maskedMSE(fut_pred, targsFut, targsFutMask)
            else:
                loss = maskedNLL(fut_pred, targsFut, targsFutMask) + torch.nn.BCELoss()(lat_pred, lat_enc) + torch.nn.BCELoss()(lon_pred, lon_enc)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            total_loss += loss.item()
            batch_time = time.time() - start_time
            avg_time_per_batch += batch_time

            # Update progress bar with detailed info
            pbar.set_postfix({
                'Batch': f"{batch_idx + 1}/{len(train_loader)}",
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}",
                'ETA': f"{int(avg_time_per_batch / (batch_idx + 1) * (len(train_loader) - batch_idx - 1))}s"
            })
            pbar.update(1)

            # Log batch-level loss
            if (batch_idx + 1) % 100 == 0:
                logger.add_scalar("Loss/train_batch", total_loss / (batch_idx + 1), epoch_num * len(train_loader) + batch_idx)

    avg_loss = total_loss / len(train_loader)
    train_loss_history.append(avg_loss)  # Append to the history list for plotting
    logger.add_scalar("Loss/epoch_train", avg_loss, epoch_num)


def validate_epoch(epoch_num, model, val_loader, args, logger_val, val_loss_history):
    """Validate the model after each epoch."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Ensure data is on the correct device
            inputs = [d.cuda() if isinstance(d, torch.Tensor) and args.use_cuda else d for d in data]
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = inputs

            fut_pred, lat_pred, lon_pred = model(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)
            loss = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, avg_along_time=True)

            total_loss += loss.item()

            if i == 19:
                logger_val.add_scalar("Loss/val_batch", total_loss / 20, epoch_num)
                break

    avg_loss_val = total_loss / len(val_loader)
    val_loss_history.append(avg_loss_val)
    logger_val.add_scalar("Loss/epoch_val", avg_loss_val, epoch_num)

    logging.info(f"Validation | Avg Loss: {avg_loss_val:.4f}")


def train_my_model():
    """Main training function."""
    args = parse_arguments()

    log_path = f"./trained_models/{args.name}/"
    setup_logging(log_path)

    logger, logger_val = None, None
    if args.tensorboard:
        logger = SummaryWriter(log_path + f'train-pre{args.pretrain_epochs}-nll{args.train_epochs}')
        logger_val = SummaryWriter(log_path + f'validation-pre{args.pretrain_epochs}-nll{args.train_epochs}')

    model, optimizer = initialize_model(args)
    train_loader, val_loader = load_datasets(args)

    train_loss_history = []  # List to store training loss per epoch
    val_loss_history = []  # List to store validation loss per epoch

    logging.info(f"Starting training: {args.name}")
    logging.info(f"Batch size: {args.batch_size} | Learning rate: {args.learning_rate}")
    logging.info(f"Using Planning Module: {args.use_planning} | Using Fusion Module: {args.use_fusion}")

    for epoch_num in range(args.pretrain_epochs + args.train_epochs):
        train_epoch(epoch_num, model, optimizer, train_loader, args, logger, train_loss_history)
        validate_epoch(epoch_num, model, val_loader, args, logger_val, val_loss_history)

        plot_and_save_loss(epoch_num, train_loss_history, val_loss_history, log_path)

        model_save_path = log_path + f"{args.name}-pre{args.pretrain_epochs if epoch_num < args.pretrain_epochs else 0}-nll{epoch_num}.tar"
        torch.save(model.state_dict(), model_save_path)

    final_model_save_path = log_path + f"{args.name}.tar"
    torch.save(model.state_dict(), final_model_save_path)
    logging.info(f"Model saved at {final_model_save_path}")


if __name__ == '__main__':
    train_my_model()

