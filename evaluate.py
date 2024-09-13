# Required libraries
import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import pandas as pd

from data4process import highwayTrajDataset
from utils import initLogging, maskedNLL, maskedMSE, maskedNLLTest
from modelTraj import TrajPred

# Constants
FEET2METER = 0.3048

# Argument parser setup
parser = argparse.ArgumentParser(description="Evaluation: Planning-informed Trajectory Prediction for Autonomous Driving")
parser.add_argument('--use_cuda', action='store_false', default=True, help='Use CUDA (default: True)')
parser.add_argument('--use_fusion', action="store_false", default=True, help='Use fusion module (default: True)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
parser.add_argument('--train_output_flag', action="store_true", default=False, help='Concatenate with true maneuver label (default: False)')
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
parser.add_argument('--name', type=str, default="1", help='Model name (default: "1")')
parser.add_argument('--test_set', type=str, help='Path to test datasets')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (default: 0)')
parser.add_argument('--metric', type=str, default="agent", help='Evaluation metric: RMSE & NLL by agent/sample (default: "agent")')
parser.add_argument('--IA_module', action='store_false', default=True, help='IA module (default: True)')


def model_evaluate():
    args = parser.parse_args() 
    # Load the trained model
    model = TrajPred(args)
    model.load_state_dict(torch.load(f'./trained_models/{args.name.split("-")[0]}/{args.name}.tar'))
    if args.use_cuda:
        model = model.cuda()

    model.eval()  # Set model to evaluation mode
    model.train_output_flag = False
    initLogging(log_file=f'./trained_models/{args.name.split("-")[0]}/evaluation.log')

    logging.info(f"Loading test data from {args.test_set}...")
    
    # Load test dataset
    tsSet = highwayTrajDataset(
        path=args.test_set,
        targ_enc_size=args.social_context_size + args.dynamics_encoding_size,
        grid_size=args.grid_size,
        fit_plan_traj=False,
        fit_plan_further_ds=args.plan_info_ds
    )
    logging.info(f"TOTAL :: {len(tsSet)} test samples.")
    
    # Create DataLoader for test set
    tsDataloader = DataLoader(
        tsSet, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=tsSet.collate_fn
    )

    logging.info(f"<{args.name}> evaluated by {args.metric}-based NLL & RMSE, with planning input of {args.plan_info_ds * 0.2}s step.")
    
    # Initialize loss statistics based on the evaluation metric
    if args.metric == 'agent':
        # Initialize for agent-based evaluation
        nll_loss_stat = np.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                  np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length))
        rmse_loss_stat = np.zeros_like(nll_loss_stat)
        both_count_stat = np.zeros_like(nll_loss_stat)
    elif args.metric == 'sample':
        # Initialize for sample-based evaluation
        rmse_loss = torch.zeros(25).cuda()
        rmse_counts = torch.zeros(25).cuda()
        nll_loss = torch.zeros(25).cuda()
        nll_counts = torch.zeros(25).cuda()
    else:
        raise RuntimeError("Invalid evaluation metric specified.")

    timer = []  # Timer to track evaluation time

    # Iterate through batches in the test set
    with torch.no_grad():
        for i, data in enumerate(tsDataloader):
            start_time = time.time()
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, idxs = data

            # Move data to GPU if necessary
            if args.use_cuda:
                nbsHist = nbsHist.cuda()
                nbsMask = nbsMask.cuda()
                planFut = planFut.cuda()
                planMask = planMask.cuda()
                targsHist = targsHist.cuda()
                targsEncMask = targsEncMask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                targsFut = targsFut.cuda()
                targsFutMask = targsFutMask.cuda()

            tic = time.time()
            # Make predictions using the model
            fut_pred, lat_pred, lon_pred = model(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)
            prediction_time = time.time() - tic
            timer.append(prediction_time)

            # Update evaluation metrics based on the specified metric
            if args.metric == 'agent':
                dsIDs, targsIDs = tsSet.batchTargetVehsInfo(idxs)
                nll, count_nll = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, separately=True)
                mse, count_mse = maskedMSE(fut_pred, targsFut, targsFutMask, separately=True)
                # Update accuracy metrics for lateral and longitudinal predictions
                lat_acc = (torch.sum(torch.argmax(lat_pred, dim=1) == torch.argmax(lat_enc, dim=1)).item() / lat_enc.size(0))
                lon_acc = (torch.sum(torch.argmax(lon_pred, dim=1) == torch.argmax(lon_enc, dim=1)).item() / lon_enc.size(0))

                # Store statistics for each vehicle
                for j, targ in enumerate(targsIDs):
                    dsID = dsIDs[j]
                    nll_loss_stat[dsID, targ, :] += nll[:, j]
                    rmse_loss_stat[dsID, targ, :] += mse[:, j]
                    both_count_stat[dsID, targ, :] += count_nll[:, j]

            elif args.metric == 'sample':
                # Sample-based evaluation
                nll, count_nll = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask)
                nll_loss += nll.detach()
                nll_counts += count_nll.detach()

                mse, count_mse = maskedMSE(fut_pred, targsFut, targsFutMask)
                rmse_loss += mse.detach()
                rmse_counts += count_mse.detach()

            # Calculate batch time and provide progress
            batch_time = time.time() - start_time
            avg_eva_time = batch_time / args.batch_size

            if i % 100 == 99:
                eta = avg_eva_time * (len(tsSet) / args.batch_size - i)
                logging.info(f"Evaluation progress: {i/(len(tsSet)/args.batch_size) * 100:.2f}% | ETA: {int(eta)}s")
                avg_eva_time = 0

    # Final evaluation metrics calculation
    if args.metric == 'agent':
        # Aggregate metrics for agent-based evaluation
        ds_ids, veh_ids = both_count_stat[:, :, 0].nonzero()
        rmseOverall, nllOverall = [], []
        for i in range(len(veh_ids)):
            # Normalize RMSE and NLL by the number of predictions for each vehicle
            rmse_avg = rmse_loss_stat[ds_ids[i], veh_ids[i], :] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
            nll_avg = nll_loss_stat[ds_ids[i], veh_ids[i], :] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)

            # Convert RMSE from feet to meters
            rmseOverall.append(np.sqrt(rmse_avg) * FEET2METER)
            nllOverall.append(nll_avg)

        rmseOverall = np.mean(rmseOverall, axis=0)
        nllOverall = np.mean(nllOverall, axis=0)

    elif args.metric == 'sample':
        # For sample-based evaluation, normalize RMSE and NLL by the counts
        rmseOverall = (torch.sqrt(rmse_loss / rmse_counts) * FEET2METER).cpu().numpy()
        nllOverall = (nll_loss / nll_counts).cpu().numpy()

    # Log final RMSE and NLL results
    logging.info(f"Final RMSE (m): {rmseOverall[4::5]}, Mean: {rmseOverall[4::5].mean():.3f}")
    logging.info(f"Final NLL (nats): {nllOverall[4::5]}, Mean: {nllOverall[4::5].mean():.3f}")

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
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 2, 1)
    #sns.lineplot(data=rmseOverall, label='RMSE')
    #plt.title("RMSE over time steps")
    #plt.xlabel("Time step")
    #plt.ylabel("RMSE (m)")

    #plt.subplot(1, 2, 2)
    #sns.lineplot(data=nllOverall, label='NLL')
    #plt.title("NLL over time steps")
    #plt.xlabel("Time step")
    #plt.ylabel("NLL (nats)")

    #plt.tight_layout()
    #plt.show()

if __name__ == '__main__':
    model_evaluate()

