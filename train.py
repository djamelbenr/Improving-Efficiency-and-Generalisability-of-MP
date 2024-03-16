import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import time
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from data_process import highwayTrajDataset
from utils import initLogging, maskedNLL, maskedMSE, maskedNLLTest

from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter

HDF5_USE_FILE_LOCKING='FALSE'
'''
python train.py --name ngsim_demo --batch_size 64 --pretrain_epochs 5 --train_epochs 10 --train_set ./datasets/NGSIM/train.mat --val_set ./datasets/NGSIM/val.mat
'''
'''
    % Loading 1:dataset id, 2:Vehicle id, 3:Frame index, 
    %         6:Local X, 7:Local Y, 15:Lane id.
    %         10:v_length, 11:v_Width, 12:v_Class
    %         13:Velocity (feet/s), 14:Acceleration (feet/s2).
    %%=== Longitudinal and Lateral locations, Velocity and Acceleration 
    %%%
    % ----------+------------+----------+-----------+---------+--------+--------+
    % param/row +     1      +     2    +     6/7   +    15   +    10  +  11
    % ----------+------------+----------+-----------+---------+--------+--------- 
    % --        + Vehicle id + Frame Idx+ Local X/Y + Lane id + Veh_len+Veh_Width
    % ----------+------------+----------+-----------+---------+--------+--------- 
    % ----------+------------+----------+-----------+---------+--------+--------+
    % param/row +     12     +     13   +     14    + --      + --     + --
    % ----------+------------+----------+-----------+---------+--------+--------- 
    % --        + Veh_Class  + Velocity + Acc       + --      + --     + -- 
    % ----------+------------+----------+-----------+---------+--------+---------
    %% should be converted from feet to meter ...
    
'''

## General arguments ... 
parser = argparse.ArgumentParser(description="train my motion predictor ...") 
parser.add_argument('--train_output_flag', action="store_false", help='if concatenate with true maneuver label (default: True)', default = True)
parser.add_argument('--tensorboard', action='store_false', help='if using tensorboard (default : True otherwise use Tensorboard, cuda : False)', default=True) # use cuda ...


## setting up training- cuda and fusion (planning and targets)...
parser.add_argument('--use_cuda', action='store_false', help='if use cuda (default: True)', default = True) # use cuda ...
parser.add_argument('--use_planning', action='store_false', help='if use planning coupled module, otherwise turn it off (default: True)', default=False) # use planning module ...
parser.add_argument('--use_fusion', action='store_false', help='if use target vehicle info fusion module (default : True otherwise turn it off, default : False)', default=True) 
# learning parameters ...
parser.add_argument('--batch_size', type=int, help='data batch size to be used (default: 32)',  default=32) # start with 32 and increase to 64 and 128...

#parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=0.0001) # default 0.001
parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=0.001) # default 0.001
parser.add_argument('--momentum', type=float, help='momentum (default: 0.9)', default=0.9)
parser.add_argument('--Weight decay', type=float, help='Weight decay (default: 0.1)', default=0.5)

## arguments for I/ O 
# setting up the grid size ...
parser.add_argument('--grid_size', type=int, help='default value (25, 5)pixels which is equal to (60.96X10.67) meters', default=[25, 5]) # size of the grid size to capture social-consistancy of the info ... 


## in-out length 
parser.add_argument('--in_length', type=int,  help='History sequence (default: 16)',default = 16) # which means 3s History tarj at 5Hz or 200 ms
parser.add_argument('--out_length', type=int, help='Predict sequence (default: 25)',default = 25) # which means 5s future traj at 5Hz or 200 ms

##--------------
# The manoeuvres are classified into lateral and longitidunal behaviours: 
# Lateral manoeuvre:          
# Longitudinal maanoeuvres: 
##
parser.add_argument('--num_lat_classes', type=int, help='Classes of the Lateral behaviours',      default=3)
parser.add_argument('--num_lon_classes', type=int, help='Classes of the Longitudinal behaviours', default=2)
##--------------

## Social-LSTM + CNN EncoderDecoder Network parameters ...
parser.add_argument('--temporal_embedding_size',type=int, help='Embedding size of the input traj', default=32)

parser.add_argument('--encoder_size', type=int, help='LSTM encoder size', default=64)
parser.add_argument('--decoder_size', type=int, help='LSTM decoder size', default=128)

parser.add_argument('--soc_conv_depth', type=int, help='The 1st social conv depth',  default = 64)
parser.add_argument('--soc_conv2_depth', type=int, help='The 2nd social conv depth',  default = 16)

parser.add_argument('--dynamics_encoding_size', type=int,  help='Embedding size of the vehicle dynamic-Velocity and acceleration',  default = 32)

parser.add_argument('--social_context_size', type=int,  help='Embedding size of the social context tensor',  default = 80)
parser.add_argument('--fuse_enc_size', type=int,  help='Feature size to be fused',  default = 112)

parser.add_argument('--name', type=str, help='log name (default: "1")', default="1")
parser.add_argument('--train_set', type=str, help='Path to train datasets')
parser.add_argument('--val_set', type=str, help='Path to validation datasets')

# data loader number of workders
parser.add_argument("--num_workers", type=int, default=0, help="number of workers used for dataloader")
parser.add_argument('--pretrain_epochs', type=int, help='epochs of pre-training using MSE', default = 5)
parser.add_argument('--train_epochs',    type=int, help='epochs of training using NLL', default = 10)

parser.add_argument('--IA_module', action='store_false', help='if use IA_module it off, default : false', default=True)
## --- 

def train_my_model():
    args=parser.parse_args()
    ## login to the training path ...
    log_path="./trained_models/{}/".format(args.name)
    os.makedirs(log_path, exist_ok=True) # 
    initLogging(log_file=log_path+'train.log') # 
    # use tensorboard ...
    """
    example of using tensorboard oin pytorch ...
    https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    """
    if args.tensorboard:
        logger = SummaryWriter(log_path + 'train-pre{}-nll{}'.format(args.pretrain_epochs, args.train_epochs))
        logger_val = SummaryWriter(log_path + 'validation-pre{}-nll{}'.format(args.pretrain_epochs, args.train_epochs))
        
  
    ## login with the default values ...
    ## ...
    logging.info("--->>>>".format(args.name))
    logging.info("Batch size : {}".format(args.batch_size))
    logging.info("Learning rate: {} ".format(args.learning_rate))
    # fuse info ...
    logging.info("Use Planning Coupled: {}".format(args.use_planning))
    logging.info("Use Target Fusion: {}".format(args.use_fusion))
    
    
    # Initia parameters ...
    TraJ= TrajPred(args) 
    if args.use_cuda:
        TraJ=TraJ.cuda() 
    optimizer = torch.optim.Adam(TraJ.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.SGD(TraJ.parameters(),lr=args.learning_rate, momentum=args.momentum)
    #optimizer = torch.optim.Adam(TraJ.parameters(),lr=args.learning_rate, momentum=args.momentum)# pre: 0.01
    
    crossEnt = torch.nn.BCELoss() # cross-entropy function as a Loss...
    
    
    ## Initialize training ...
    pretrainEpochs = args.pretrain_epochs
    trainEpochs    = args.train_epochs
    batch_size     = args.batch_size
    
    # initlize data loaders ...
    ## ==== load out data ...
    logging.info("get the damn data...: {}".format(args.train_set))
    
    trainSet=highwayTrajDataset(path=args.train_set,
                                targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                                grid_size=args.grid_size,
                                fit_plan_traj=False)
    
    #print('Training Data: ', trainSet)
    logging.info("Validation dataset: {}".format(args.val_set))
    
    #logging.info("Validation dataset: {}".format(args.valid_data))
    
    valSet=highwayTrajDataset(path=args.val_set,
                                targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                                grid_size=args.grid_size,
                                fit_plan_traj=True)
    
    
    trainDataloader=DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=trainSet.collate_fn)
    valDataloader  =DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=valSet.collate_fn)
    
    print('Validation Data: ', valDataloader)
    
    
    
    logging.info("DataSet Prepared : {} train data, {} validation data\n".format(len(trainSet), len(valSet)))
    logging.info("Network structure: {}\n".format(TraJ))
    
    
    ## Training process
    for epoch_num in range( pretrainEpochs + trainEpochs ):
        if epoch_num == 0:
            logging.info('Pretrain with MSE loss')
        elif epoch_num == pretrainEpochs:
            logging.info('Train with NLL loss')
        ## Variables to track training performance:
        avg_time_tr, avg_loss_tr, avg_loss_val = 0, 0, 0
        ## Training status, reclaim after each epoch
        TraJ.train()
        TraJ.train_output_flag = True
        
        for i, data in enumerate(trainDataloader):
            st_time = time.time()
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = data
            
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

            # Forward pass
            fut_pred, lat_pred, lon_pred = TraJ(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc)
            
            if epoch_num < pretrainEpochs:
                # Pre-train with MSE loss to speed up training
                l = maskedMSE(fut_pred, targsFut, targsFutMask)
            else:
                # Train with NLL loss
                
                l = maskedNLL(fut_pred, targsFut, targsFutMask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                
            logger.add_scalar("Loss/train", l,        epoch_num)
            logger.add_scalar("Loss/avg_loss_val", l, epoch_num)
            
            #model =TraJ.train()
            #logger.add_graph('TrajPred',model)
            
            
            # Back-prop and update weights
            optimizer.zero_grad()
            l.backward()
            prev_vec_norm = torch.nn.utils.clip_grad_norm_(TraJ.parameters(), 10)
            optimizer.step()

            # Track average train loss and average train time:
            batch_time = time.time()-st_time
            avg_loss_tr += l.item()

            ##print("training average loss: ", avg_loss_tr)
            avg_time_tr += batch_time

            # For every 100 batches: record loss, validate model, and plot.
            if i%100 == 99:
                eta = avg_time_tr/100*(len(trainSet)/batch_size-i)
                epoch_progress = i * batch_size / len(trainSet)
                
                logging.info(f"Epoch no:{epoch_num+1}"+
                             f" | Epoch progress(%):{epoch_progress*100:.2f}"+
                             f" | Avg train loss:{avg_loss_tr/100:.2f}"+
                             f" | ETA(s):{int(eta)}")
                
                
                
                if args.tensorboard:
                    logger.add_scalar("RMSE" if epoch_num < pretrainEpochs else "NLL", avg_loss_tr / 100, (epoch_progress + epoch_num) * 100)

                ## Validatation during training:
                eval_batch_num = 20
                with torch.no_grad():
                    TraJ.eval()
                    TraJ.train_output_flag = False
                    
                    for i, data in enumerate(valDataloader):
                        nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, _ = data
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
                            
                        if epoch_num < pretrainEpochs:
                            # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                            TraJ.train_output_flag = True
                            fut_pred, _, _ = TraJ(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
                                                 lat_enc, lon_enc)
                            l = maskedMSE(fut_pred, targsFut, targsFutMask)
                        else:
                            # During training with NLL loss, validate with NLL over multi-modal distribution
                            fut_pred, lat_pred, lon_pred = TraJ(nbsHist, nbsMask, planFut, planMask, targsHist,
                                                               targsEncMask, lat_enc, lon_enc)
                            l = maskedNLLTest(fut_pred, lat_pred, lon_pred, targsFut, targsFutMask, avg_along_time=True)
                        avg_loss_val += l.item()
                        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
                        logging.info(f"Epoch no:{epoch_num+1}"               +
                             f" | Epoch progress(%):{epoch_progress*100:.2f}"+
                             f" | Avg train loss:{avg_loss_tr/100:.2f}"      +
                             f" | Avg val loss:{avg_loss_val/100:.2f}"       +
                             f" | ETA(s):{int(eta)}")
                        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
                        if i==(eval_batch_num-1):
                            if args.tensorboard:
                                logger_val.add_scalar("RMSE" if epoch_num < pretrainEpochs else "NLL", avg_loss_val / eval_batch_num, (epoch_progress + epoch_num) * 100)
                            break
                # Clear statistic
                avg_time_tr, avg_loss_tr, avg_loss_val = 0, 0, 0
                # Revert to train mode after in-process evaluation.
                TraJ.train()
                TraJ.train_output_flag = True

        ## Save the model after each epoch______________________________________________________________________________
        epoCount = epoch_num + 1
        if epoCount < pretrainEpochs:
            torch.save(TraJ.state_dict(), log_path + "{}-pre{}-nll{}.tar".format(args.name, epoCount, 0))
        else:
            torch.save(TraJ.state_dict(), log_path + "{}-pre{}-nll{}.tar".format(args.name, pretrainEpochs, epoCount - pretrainEpochs))

    # All epochs finish________________________________________________________________________________________________
    torch.save(TraJ.state_dict(), log_path+"{}.tar".format(args.name))
    logging.info("Model saved in trained_models/{}/{}.tar\n".format(args.name, args.name))

if __name__ == '__main__':
    train_my_model()
