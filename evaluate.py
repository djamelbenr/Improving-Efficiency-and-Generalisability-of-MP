# Required libraries
import os
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from time import perf_counter
from prettytable import PrettyTable

from data4process import highwayTrajDataset
from utils import initLogging, maskedNLL, maskedMSE, maskedNLLTest
from modelTraj import TrajPred

# Constants
FEET2METER = 0.3048

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluation: Planning-informed Trajectory Prediction for Autonomous Driving")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA (default: False)')
    parser.add_argument('--use_fusion', action='store_true', default=False, help='Use fusion module (default: False)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--train_output_flag', action='store_true', default=False, help='Concatenate with true maneuver label (default: False)')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[25, 5], help='Grid size (default: [25, 5])')
    parser.add_argument('--in_length', type=int, default=16, help='History sequence length (default: 16)')
    parser.add_argument('--out_length', type=int, default=25, help='Prediction sequence length (default: 25)')
    parser.add_argument('--num_lat_classes', type=int, default=3, help='Lateral behavior classes (default: 3)')
    parser.add_argument('--num_lon_classes', type=int, default=2, help='Longitudinal behavior classes (default: 2)')
    parser.add_argument('--temporal_embedding_size', type=int, default=32, help='Temporal embedding size (default: 32)')
    parser.add_argument('--encoder_size', type=int, default=64, help='LSTM encoder size (default: 64)')
    parser.add_argument('--decoder_size', type=int, default=128, help='LSTM decoder size (default: 128)')
    parser.add_argument('--soc_conv_depth', type=int, default=64, help='First social conv depth (default: 64)')
    parser.add_argument('--soc_conv2_depth', type=int, default=16, help='Second social conv depth (default: 16)')
    parser.add_argument('--dynamics_encoding_size', type=int, default=32, help='Vehicle dynamics embedding size (default: 32)')
    parser.add_argument('--social_context_size', type=int, default=80, help='Social context tensor embedding size (default: 80)')
    parser.add_argument('--fuse_enc_size', type=int, default=112, help='Fused feature size (default: 112)')
    parser.add_argument('--name', type=str, required=True, help='Model name (required)')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test datasets (required)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (default: 0)')
    parser.add_argument('--metric', type=str, default="agent", choices=['agent', 'sample'], help='Evaluation metric: RMSE & NLL by agent/sample (default: "agent")')
    parser.add_argument('--IA_module', action='store_true', default=False, help='Enable IA module (default: False)')
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize RMSE and NLL results (default: False)')
    parser.add_argument('--plan_info_ds', type=int, default=1, help='Planning information downsample rate (default: 1)')
    return parser.parse_args()

def model_evaluate():
    """
    Evaluate the trajectory prediction model on the test dataset.
    """
    args = parse_arguments()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Initialize logging
    model_dir = f'./trained_models/{args.name.split("-")[0]}'
    os.makedirs(model_dir, exist_ok=True)
    initLogging(log_file=os.path.join(model_dir, 'evaluation.log'))

    # Load the trained model
    model = TrajPred(args)
    model_path = os.path.join(model_dir, f'{args.name}.tar')
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    model.train_output_flag = False

    logging.info(f"Loading test data from {args.test_set}...")

    # Load test dataset
    if not os.path.exists(args.test_set):
        logging.error(f"Test dataset not found at {args.test_set}")
        return
    test_dataset = highwayTrajDataset(
        path=args.test_set,
        targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
        grid_size=args.grid_size,
        fit_plan_traj=False,
        fit_plan_further_ds=args.plan_info_ds
    )
    logging.info(f"TOTAL :: {len(test_dataset)} test samples.")

    # Create DataLoader for test set
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=test_dataset.collate_fn
    )

    total_batches = len(test_dataloader)
    logging.info(f"<{args.name}> evaluated by {args.metric}-based NLL & RMSE, with planning input of {args.plan_info_ds * 0.2}s step.")

    # Initialize loss statistics based on the evaluation metric
    if args.metric == 'agent':
        # Initialize for agent-based evaluation
        max_dsID = int(np.max(test_dataset.Data[:, 0]))
        max_targID = int(np.max(test_dataset.Data[:, 13:(13 + test_dataset.grid_cells)]))
        nll_loss_stat = np.zeros((max_dsID + 1, max_targID + 1, args.out_length))
        rmse_loss_stat = np.zeros_like(nll_loss_stat)
        both_count_stat = np.zeros_like(nll_loss_stat)
    elif args.metric == 'sample':
        # Initialize for sample-based evaluation
        rmse_loss = torch.zeros(args.out_length, device=device)
        rmse_counts = torch.zeros(args.out_length, device=device)
        nll_loss = torch.zeros(args.out_length, device=device)
        nll_counts = torch.zeros(args.out_length, device=device)
    else:
        logging.error("Invalid evaluation metric specified.")
        return

    timer = []  # Timer to track evaluation time
    avg_eva_time = 0  # Initialize average evaluation time

    # Iterate through batches in the test set
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            start_time = perf_counter()

            # Unpack data
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, idxs = data

            # Move data to device
            nbsHist = nbsHist.to(device)
            nbsMask = nbsMask.to(device)
            planFut = planFut.to(device)
            planMask = planMask.to(device)
            targsHist = targsHist.to(device)
            targsEncMask = targsEncMask.to(device)
            targsFut = targsFut.to(device)
            targsFutMask = targsFutMask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)

            # Prediction
            tic = perf_counter()
            fut_pred, lat_pred, lon_pred = model(
                nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc
            )
            prediction_time = perf_counter() - tic
            timer.append(prediction_time)

            # Update evaluation metrics based on the specified metric
            if args.metric == 'agent':
                dsIDs, targsIDs = test_dataset.batchTargetVehsInfo(idxs)
                nll, count_nll = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, separately=True)
                mse, count_mse = maskedMSE(fut_pred, targsFut, targsFutMask, separately=True)

                # Store statistics for each vehicle
                for j, targ in enumerate(targsIDs):
                    dsID = dsIDs[j]
                    nll_loss_stat[dsID, targ, :] += nll[:, j].cpu().numpy()
                    rmse_loss_stat[dsID, targ, :] += mse[:, j].cpu().numpy()
                    both_count_stat[dsID, targ, :] += count_nll[:, j].cpu().numpy()

            elif args.metric == 'sample':
                # Sample-based evaluation
                nll, count_nll = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask)
                nll_loss += nll
                nll_counts += count_nll

                mse, count_mse = maskedMSE(fut_pred, targsFut, targsFutMask)
                rmse_loss += mse
                rmse_counts += count_mse

            # Calculate batch time and provide progress
            batch_time = perf_counter() - start_time
            avg_eva_time += batch_time / args.batch_size

            if (i + 1) % 100 == 0 or (i + 1) == total_batches:
                eta = avg_eva_time * (total_batches - (i + 1))
                logging.info(f"Evaluation progress: {(i + 1)/total_batches * 100:.2f}% | ETA: {int(eta)}s")
                avg_eva_time = 0  # Reset average evaluation time after logging

    # Final evaluation metrics calculation
    epsilon = 1e-9  # To avoid division by zero
    if args.metric == 'agent':
        # Aggregate metrics for agent-based evaluation
        ds_ids, veh_ids = np.nonzero(both_count_stat[:, :, 0])
        rmseOverall = []
        nllOverall = []
        for i in range(len(veh_ids)):
            dsID = ds_ids[i]
            targID = veh_ids[i]
            counts = both_count_stat[dsID, targID, :] + epsilon
            rmse_avg = rmse_loss_stat[dsID, targID, :] / counts
            nll_avg = nll_loss_stat[dsID, targID, :] / counts

            # Convert RMSE from feet to meters
            rmseOverall.append(np.sqrt(rmse_avg) * FEET2METER)
            nllOverall.append(nll_avg)

        rmseOverall = np.mean(rmseOverall, axis=0)
        nllOverall = np.mean(nllOverall, axis=0)

    elif args.metric == 'sample':
        # For sample-based evaluation, normalize RMSE and NLL by the counts
        rmseOverall = (torch.sqrt(rmse_loss / (rmse_counts + epsilon)) * FEET2METER).cpu().numpy()
        nllOverall = (nll_loss / (nll_counts + epsilon)).cpu().numpy()

    # Log final RMSE and NLL results
    logging.info(f"Final RMSE (m): {rmseOverall[4::5]}, Mean: {np.mean(rmseOverall[4::5]):.3f}")
    logging.info(f"Final NLL (nats): {nllOverall[4::5]}, Mean: {np.mean(nllOverall[4::5]):.3f}")

    # Log prediction time statistics
    time2Pre = np.array(timer)
    logging.info(f"Average prediction time per batch: {np.mean(time2Pre):.4f}s")

    # Generate a table summarizing the results
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Mean", "Std"]
    summary_table.add_row(["RMSE (m)", np.mean(rmseOverall), np.std(rmseOverall)])
    summary_table.add_row(["NLL (nats)", np.mean(nllOverall), np.std(nllOverall)])
    print(summary_table)

    # Optionally, visualize the RMSE and NLL results
    if args.visualize:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(x=np.arange(len(rmseOverall)), y=rmseOverall, label='RMSE')
        plt.title("RMSE over time steps")
        plt.xlabel("Time step")
        plt.ylabel("RMSE (m)")

        plt.subplot(1, 2, 2)
        sns.lineplot(x=np.arange(len(nllOverall)), y=nllOverall, label='NLL')
        plt.title("NLL over time steps")
        plt.xlabel("Time step")
        plt.ylabel("NLL (nats)")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    model_evaluate()
