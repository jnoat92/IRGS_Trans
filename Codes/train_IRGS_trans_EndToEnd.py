# No@
import os
from datetime import datetime
import time
import argparse
import wandb

from utils.utils import AvgMeter, boolean_string, Metrics, aux_obj
from utils.dataloader import Data_proc, RadarSAT2_Dataset, Split_in_Patches_no_padding, \
                             Load_patches, Load_patches_on_the_fly, \
                             Extract_segments, Load_patches_segments

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.utils.data as data
from lib.unet.unet_model import UNet
from lib.custom_networks import Trans_no_patch_embed, IRGS_Trans

from icecream import ic
from tqdm import tqdm
# from plyer import notification
import joblib
import json


def Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')    
    parser.add_argument('--nodes', default=1, type=int, help='')    

    parser.add_argument('--sweep', type=boolean_string, default=False, help='hyperparameter tunning mode')
    parser.add_argument('--epochs', type=int, default=50, help='epoch number')
    parser.add_argument('--patience', type=int, default=15, help='number of epochs after no improvements (stop criteria)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')

    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--patch_size', type=int, default=4000, help='input image size (square)')
    parser.add_argument('--patch_overlap', type=float, default=0.2, help='Overlap between patches')

    parser.add_argument('--token_size', type=int, default=16, help='sub-image/word size (square) for transformer input')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads on self-attention module')
    parser.add_argument('--trans_depth', type=int, default=8, help='Number of transformer blocks')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embeding depth')

    parser.add_argument('--mode', type=str, default='end_to_end', choices=['end_to_end', 'multi_stage'], help='.......')
    parser.add_argument('--stage', type=str, default='cnn', choices=['cnn', 'transformer'], help='specifiy STAGE on multi-stage mode')
    parser.add_argument('--loss_term', type=str, default='end_to_end', choices=['end_to_end', 'transformer'], help='Specify loss terms for end_to_end approach')
    parser.add_argument('--max_length', type=int, default=500, help='Maximum sequence length')
    parser.add_argument('--mix_images', type=boolean_string, default=True, help='Mix tokens from different images in the batch')
    parser.add_argument('--random_tokens', type=boolean_string, default=True, help='random/oredered tokens')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight associated to Transformer Loss in end_to_end approach')

    parser.add_argument('--CNN', type=str, default='resnet18', help='CNN backbone')

    parser.add_argument('--save_samples', type=boolean_string, default=True, help='If True, the samples will be saved in the HD to be loaded during training (It requires less RAM) \
                                                                          If False, the samples are extracted directly from the scene (faster)')
    # parser.add_argument('--Dataset_dir', type=str,
    #                     default='../../Dataset/RADARSAT-2-CP/', help='dataset_path')
    # parser.add_argument('--train_path', type=str,
    #                     default='Scene58/', help='path to train dataset')
    parser.add_argument('--Datasets_dir', type=str, default='../../Dataset/', help='datasets path')
    parser.add_argument('--Dataset_name', type=str, default='21-scenes-less_resolution', help='dataset name')
    parser.add_argument('--train_path', type=str, default='20100418_20100426', help='list of scenes separates by "_"')
    parser.add_argument('--test_path', type=str, default='20110725', help='list of scenes separates by "_"')
    parser.add_argument('--in_chans', type=int, default=2, help='Number of classes')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')

    parser.add_argument('--irgs_classes', type=int, default=10, help='Number of classes considered on IRGS')
    parser.add_argument('--irgs_iter', type=int, default=100, help='Number of iterations on IRGS')
    parser.add_argument('--token_option', type=str, default='superpixels', choices=['superpixels', 'clusters'], help='.......')
    
    parser.add_argument('--data_info_dir', type=str, default='../data_info/')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    args = parser.parse_args()
    return args

def Load_data(args, norm=False):

    # =========== TRAINING DATA
    Train_samples, Val_samples, norm_params = Data_proc(args, sliding_window=True, norm=norm, aug=False, padding=False)
    
    # =========== TEST DATA
    test_data =  RadarSAT2_Dataset(args, name = args.test_path[0], phase="test")
    test_data.test_patches = Split_in_Patches_no_padding(args.patch_size, test_data.background)
    if norm:
        test_data.image = norm_params.Normalize(test_data.image)
    if args.save_samples:
        scene_data_dir = args.data_info_dir + '/{}/Patches/'.format(test_data.name)
        # test_data.save_patches(output_dir=scene_data_dir)
        Test_samples = Load_patches([scene_data_dir], stage="test")
    else:
        test_patches = []
        for j in test_data.test_patches:
            test_patches.append(tuple([0]) + j)
        Test_samples = Load_patches_on_the_fly([test_data], test_patches, stage="test")

    return Train_samples, Val_samples, Test_samples, norm_params


def Loss_func(y_pred, y_true, mask):
    return (mask * CrossEntropyLoss(reduction='none')(y_pred, y_true)).mean()

def train(model, model_no_ddp, data_loader, epoch, ckpt_dir, args, 
           stage='end_to_end', loss_term=None, cnn_optimizer=None, 
           trans_optimizer=None, device='cuda'):

    assert stage in ['end_to_end', 'cnn', 'transformer'], \
        "Not valid stage value"
    if stage == 'end_to_end':
        assert loss_term in ['end_to_end', 'transformer'], \
            "Specify valid losses with end_to_end approach"
        assert cnn_optimizer != None and trans_optimizer != None, \
            "Both optimizers required for end_to_end approach"
    if stage == 'cnn':
        assert cnn_optimizer != None, \
            "CNN optimizer required"
    if stage == 'transformer':
        assert trans_optimizer != None, \
            "Transformer optimizer required"

    if stage == 'cnn':
        model_no_ddp.cnn.train()
        model_no_ddp.transformer.eval()
    elif stage == 'transformer':
        model_no_ddp.cnn.eval()
        model_no_ddp.transformer.train()
    else:
        model.train()

    # Data
    data_loader_size = len(data_loader)
    sample_per_epochs = 5000
    n_batches = min(sample_per_epochs//data_loader.batch_size, data_loader_size)
    data_iterator = iter(data_loader)

    # Batch loop
    total_loss = AvgMeter()
    start_time = time.time()
    for i in tqdm(range(n_batches), ncols=50):
        
        # ------------ Prepare data
        images, gts, bckg, segments, n_tokens, _ = next(data_iterator)

        images = torch.permute(images, (0, 3, 1, 2)).float()
        gts = gts.long()
        bckg = bckg.float()
        
        images   = images.to(device)
        gts      = gts.to(device)
        bckg     = bckg.to(device)
        segments = segments.to(device)
        n_tokens = n_tokens.to(device)
        
        # ------------ Forward
        if cnn_optimizer   is not None: cnn_optimizer.zero_grad()
        if trans_optimizer is not None: trans_optimizer.zero_grad()
        cnn_logits, trans_logits, super_labels, mask_pad_tk, _, _ = model(images, gts, segments, n_tokens, 
                                                                    stage=stage, device=device)

        # ------------ Loss
        L0, L1 = -1, -1
        if stage == 'cnn' or (stage == 'end_to_end' and loss_term == 'end_to_end'):
            L0 = Loss_func(cnn_logits, gts, bckg)
        if stage in ['end_to_end', 'transformer']:
            trans_logits = torch.permute(trans_logits, (0, 2, 1))
            L1 = Loss_func(trans_logits, super_labels, mask_pad_tk)
        
        if   L0 == -1: L = L1
        elif L1 == -1: L = L0
        else: L = L0 + args.alpha * L1
        
        if epoch:
            # ------------ Backward
            L.backward() 
            if cnn_optimizer   is not None:
                torch.nn.utils.clip_grad_norm_(model_no_ddp.cnn.parameters(), args.grad_norm)
                cnn_optimizer.step()
            if trans_optimizer is not None: 
                torch.nn.utils.clip_grad_norm_(model_no_ddp.transformer.parameters(), args.grad_norm)
                trans_optimizer.step()
        
        # ------------ Recording Loss
        total_loss.update(L.data, data_loader.batch_size)
        if (i+1) % (1+n_batches//4) == 0 or (i+1) == n_batches or epoch == 0:
        
            print_line = 'Tr - Epoch [{:03d}], Step [{:04d}/{:04d}], time: [{:04d} segs], [loss: {:0.4f}]\n'\
                         .format(epoch, i+1, n_batches, int(time.time()-start_time), total_loss.show())
            print ('\n'+print_line)
        if epoch == 0: break

    # Save Logs
    with open(ckpt_dir + "/Log.txt", "a") as f:
        f.write(print_line)
    
    return model, model_no_ddp, cnn_optimizer, trans_optimizer

def Validate(model, data_loader, epoch, ckpt_dir, args, 
             stage='end_to_end', loss_term=None, set_='Vl'):

    cnn_loss, trans_loss, loss, cnn_acc, \
    trans_acc, cnn_IoU, trans_IoU  = test(model, data_loader, args, stage=stage, loss_term=loss_term)

    if stage == 'end_to_end' and loss_term == 'transformer':
        print_line = '{} - Epoch [{:03d}], [L_trans: {:.4f}, OA_trans: {:.2f}%, IoU_trans: {:.2f}%]\n'\
                     .format(set_, epoch, trans_loss, trans_acc, trans_IoU)
    elif stage == 'cnn':
        print_line = '{} - Epoch [{:03d}], [L_cnn: {:.4f}, OA_cnn: {:.2f}%, IoU_cnn: {:.2f}%]\n'\
                     .format(set_, epoch, cnn_loss, cnn_acc, cnn_IoU)
    else:
        print_line = '{} - Epoch [{:03d}], [L_cnn: {:.4f}, L_trans: {:.4f}, Loss: {:.4f}, '.format(set_, epoch, cnn_loss, trans_loss, loss) + \
                                           'OA_cnn: {:.2f}%, OA_trans: {:.2f}%, '.format(cnn_acc, trans_acc) + \
                                           'IoU_cnn: {:.2f}%, IoU_trans: {:.2f}%]\n'.format(cnn_IoU, trans_IoU)
    with open(ckpt_dir + "/Log.txt", "a") as f:
        print(print_line)
        f.write(print_line)
        
    wandb.log({
        '%s_cnn_loss'%(set_):   cnn_loss,
        '%s_trans_loss'%(set_): trans_loss,
        '%s_loss'%(set_):       loss,
        '%s_cnn_acc'%(set_):    cnn_acc,
        '%s_trans_acc'%(set_):  trans_acc,
        '%s_cnn_IoU'%(set_):    cnn_IoU,
        '%s_trans_IoU'%(set_):  trans_IoU,
    }, step=epoch)

    return loss

def test(model, data_loader, args, stage='end_to_end', loss_term=None, device='cuda'):

    assert stage in ['end_to_end', 'cnn', 'transformer'], \
        "Not valid stage value"
    if stage == 'end_to_end':
        assert loss_term in ['end_to_end', 'transformer'], \
            "Specify valid losses with end_to_end approach"

    model.eval()

    # Data
    data_loader_size = len(data_loader)
    sample_per_epochs = 5000
    n_batches = min(sample_per_epochs//data_loader.batch_size, data_loader_size)
    data_iterator = iter(data_loader)

    # Batch loop
    cnn_loss, trans_loss, loss, trans_acc, \
    cnn_acc, trans_IoU, cnn_IoU = [[] for i in range(7)]
    for i in tqdm(range(n_batches), ncols=50):

        # ------------ Prepare data
        images, gts, bckg, segments, n_tokens, _ = next(data_iterator)
        
        images = torch.permute(images, (0, 3, 1, 2)).float()
        gts = gts.long()
        bckg = bckg.float()
        
        images   = images.to(device)
        gts      = gts.to(device)
        bckg     = bckg.to(device)
        segments = segments.to(device)
        n_tokens = n_tokens.to(device)

        # ------------ Forward
        with torch.no_grad():
            cnn_logits, trans_logits, super_labels, mask_pad_tk, _, _ = model(images, gts, segments, n_tokens, 
                                                                        stage=stage, device=device)

        # ------------ Loss
        L0, L1 = -1, -1
        if not (stage == 'end_to_end' and loss_term == 'transformer'):
            L0 = Loss_func(cnn_logits, gts, bckg)

            probs = cnn_logits.softmax(1)
            cnn_pred = torch.argmax(probs, dim=1).detach().cpu().numpy()
            gts  = gts.detach().cpu().numpy()
            bckg = bckg.detach().cpu().numpy()

            cnn_pred = cnn_pred[bckg==1]
            gts  = gts[bckg==1]
            acc, IoU = Metrics(gts, cnn_pred, None, None)

            cnn_loss.append(L0.item())
            cnn_acc.append(acc)
            cnn_IoU.append(IoU.mean())

        if stage in ['end_to_end', 'transformer']:
            trans_logits = torch.permute(trans_logits, (0, 2, 1))
            L1 = Loss_func(trans_logits, super_labels, mask_pad_tk)

            probs = trans_logits.softmax(1)
            trans_pred = torch.argmax(probs, dim=1).detach().cpu().numpy()
            super_labels  = super_labels.detach().cpu().numpy()
            mask_pad_tk = mask_pad_tk.detach().cpu().numpy()

            trans_pred = trans_pred[mask_pad_tk==1]
            super_labels  = super_labels[mask_pad_tk==1]
            acc, IoU = Metrics(super_labels, trans_pred, None, None)

            trans_loss.append(L1.item())
            trans_acc.append(acc)
            trans_IoU.append(IoU.mean())

        if   L0 == -1: L = L1
        elif L1 == -1: L = L0
        else: L = L0 + args.alpha * L1
        loss.append(L.item())

    if stage == 'end_to_end' and loss_term == 'transformer':
        return 0,                 np.mean(trans_loss), np.mean(loss), 0,                np.mean(trans_acc), 0,                np.mean(trans_IoU)
    elif stage == 'cnn':
        return np.mean(cnn_loss), 0,                   np.mean(loss), np.mean(cnn_acc), 0,                  np.mean(cnn_IoU), 0

    return np.mean(cnn_loss), np.mean(trans_loss), np.mean(loss), np.mean(cnn_acc), np.mean(trans_acc), np.mean(cnn_IoU), np.mean(trans_IoU)
    
def Train_loop(loader, sampler, model, model_no_ddp, best_loss,
               cnn_optimizer, cnn_scheduler, trans_optimizer, trans_scheduler,
               args, ckpt_dir, epoch, stage='end_to_end', loss_term='end_to_end'):
    
    # ================ TRAINING ================ #
    print("#"*20, "Start Training", "#"*20)
    with open(ckpt_dir + "/Log.txt", "a") as f:
        f.write("New run!\n")
        f.write("{}".format(datetime.now()).split('.')[0] + "\n")

        # _ =     Validate(model, loader.train, epoch, ckpt_dir, args, 
        #                          stage=stage, loss_term=loss_term, set_='Tr')
    stop_criteria = args.patience

    while True:
        epoch += 1

        if cnn_optimizer is not None: cr_lr = cnn_optimizer.param_groups[0]['lr']
        if trans_optimizer is not None: cr_lr = trans_optimizer.param_groups[0]['lr']
        wandb.log({'epoch': epoch, 'lr':    cr_lr}, step=epoch)

        if sampler.train is not None: sampler.train.set_epoch(epoch)
        if sampler.val   is not None: sampler.val.set_epoch(epoch)
        if sampler.test  is not None: sampler.test.set_epoch(epoch)

        # ------------ Training
        model, model_no_ddp, cnn_optimizer, \
            trans_optimizer = train(model, model_no_ddp, loader.train, epoch, ckpt_dir, args, 
                                    stage=stage, loss_term=loss_term, 
                                    cnn_optimizer=cnn_optimizer, trans_optimizer=trans_optimizer)
        _ =     Validate(model, loader.train, epoch, ckpt_dir, args, 
                                 stage=stage, loss_term=loss_term, set_='Tr')
        
        # ------------ Validation
        loss =  Validate(model, loader.val, epoch, ckpt_dir, args, 
                                 stage=stage, loss_term=loss_term, set_='Vl')
        if cnn_scheduler is not None: cnn_scheduler.step(loss)
        if trans_scheduler is not None: trans_scheduler.step(loss)
        
        # ------------ Test
        _ =     Validate(model, loader.test, epoch, ckpt_dir, args, 
                                 stage=stage, loss_term=loss_term, set_='Ts')

        if loss < best_loss:
            best_loss = loss
            stop_criteria = args.patience
            checkpoint = {'epoch': epoch, 'valid_loss': best_loss}
            f = open(ckpt_dir + "/Log.txt", "a")

            if stage != 'transformer':
                checkpoint['model'] = model_no_ddp.cnn.state_dict()
                checkpoint['optimizer'] = cnn_optimizer.state_dict()
                if cnn_scheduler is not None:
                    checkpoint['scheduler'] = cnn_scheduler.state_dict()
                torch.save(checkpoint, '{}/{}_model.pt'.format(ckpt_dir, model_no_ddp.cnn.net_name) )
                print('[Saving Snapshot:] {}/{}_model.pt'.format(ckpt_dir, model_no_ddp.cnn.net_name))
                f.write("===== model {} saved =====\n".format(model_no_ddp.cnn.net_name))
                # notification.notify(title = "model {} saved\n".format(model.cnn.net_name), 
                #                     message = 'loss = {:.4f}'.format(best_loss), 
                #                     app_icon = None, 
                #                     timeout = 50)
    
            if stage != 'cnn':
                checkpoint['model'] = model_no_ddp.transformer.state_dict()
                checkpoint['optimizer'] = trans_optimizer.state_dict()
                if trans_scheduler is not None:
                    checkpoint['scheduler'] = trans_scheduler.state_dict()
                torch.save(checkpoint, '{}/{}_model.pt'.format(ckpt_dir, model_no_ddp.transformer.net_name) )

                print('[Saving Snapshot:] {}/{}_model.pt'.format(ckpt_dir, model_no_ddp.transformer.net_name))
                f.write("===== model {} saved =====\n".format(model_no_ddp.transformer.net_name))
            
                # notification.notify(title = "model {} saved\n".format(model.transformer.net_name), 
                #                     message = 'loss = {:.4f}'.format(best_loss), 
                #                     app_icon = None, 
                #                     timeout = 50)
            f.close()

        else: stop_criteria -= 1
        if not stop_criteria:
            print_line = "{} training finished\n".format(stage)
            print(print_line)
            with open(ckpt_dir + "/Log.txt", "a") as f:
                f.write(print_line)
            break
        
    return model, model_no_ddp, cnn_optimizer, trans_optimizer

def main(config=None):

    args = Arguments()
    args.Datasets_dir = os.path.join(args.Datasets_dir, args.Dataset_name)
    args.train_path = args.train_path.split('_')
    args.test_path = args.test_path.split('_')
    args.data_info_dir = os.path.join(args.data_info_dir, args.Dataset_name)

#%% ============== HYPER-PAREMETER TUNNING =============== #
    run_name = ''
    if args.sweep:
        run = wandb.init(config=config)
        args.alpha = wandb.config.alpha
        args.lr = wandb.config.lr
        args.batch_size = wandb.config.batch_size
        # run_name = run._name
        run_name = run._run_id

#%% ============== MULTIPLE GPU SETUP =============== #
    if torch.cuda.is_available(): 
        ngpus_per_node = torch.cuda.device_count()
        args.batch_size = args.batch_size  // ngpus_per_node * args.nodes
        # current_device = torch.cuda.current_device()

        if os.environ.get("SLURM_LOCALID") is not None and ngpus_per_node > 1:
    
            local_rank = int(os.environ.get("SLURM_LOCALID")) 
            rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
            current_device = local_rank
            torch.cuda.set_device(current_device)
            print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
            #init the process group
            dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
            print("process group ready!")

#%% ============== BUILD MODELS =============== #
    cnn = UNet(args.in_chans, args.n_classes)
    transformer = Trans_no_patch_embed(n_in_feat=32, num_classes=args.n_classes, 
                                        embed_dim=args.embed_dim, depth=args.trans_depth, 
                                        num_heads=args.num_heads)
    # ================ MERGE CNN - TRANSFORMER
    model = IRGS_Trans(cnn, transformer, args.max_length, 
                                  args.mix_images, args.random_tokens)
    model.cuda()
    model_no_ddp = model
    if ngpus_per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device],
                                                          find_unused_parameters=True)
        model_no_ddp = model.module
    
    # ================ OPTIMIZERS
    cnn_optimizer   = torch.optim.Adam(model_no_ddp.cnn.parameters(), args.lr, betas=(args.beta1, args.beta2))
    trans_optimizer = torch.optim.Adam(model_no_ddp.transformer.parameters(), args.lr, betas=(args.beta1, args.beta2))
    # cnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, factor=0.5, patience=5, verbose=True)
    # trans_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trans_optimizer, factor=0.5, patience=5, verbose=True)
    cnn_scheduler, trans_scheduler = None, None

#%% ============== LOAD CHECKPOINT =============== #
    if args.mode == 'end_to_end':
        if args.loss_term == 'transformer':
            ckpt_irgs_trans = os.path.join(args.ckpt_path, args.Dataset_name, 
                                        model_no_ddp.net_name + '_' + args.token_option, 
                                        'end_to_end', 'Loss_transformer', args.model_name)
        else:
            ckpt_irgs_trans = os.path.join(args.ckpt_path, args.Dataset_name, 
                                        model_no_ddp.net_name + '_' + args.token_option, 
                                        'end_to_end', 'Loss_end_to_end', args.model_name)
        ckpt_CNN = ckpt_irgs_trans
    
    elif args.mode == 'multi_stage':
        ckpt_irgs_trans = os.path.join(args.ckpt_path, args.Dataset_name, 
                                    model_no_ddp.net_name + '_' + args.token_option, 'multi_stage', args.model_name)
        ckpt_CNN = os.path.join(args.ckpt_path, args.Dataset_name, cnn.net_name, args.model_name)

    ckpt_CNN = os.path.join(ckpt_CNN, run_name)
    ckpt_irgs_trans = os.path.join(ckpt_irgs_trans, run_name)

    os.makedirs(ckpt_irgs_trans, exist_ok=True)
    os.makedirs(ckpt_CNN, exist_ok=True)
        
    # ================ CNN
    cnn_epoch = -1
    cnn_best_lost = 2**16
    cnn_last_line = 0
    if os.path.exists('{}/{}_model.pt'.format(ckpt_CNN, model_no_ddp.cnn.net_name)):
        checkpoint = torch.load('{}/{}_model.pt'.format(ckpt_CNN, model_no_ddp.cnn.net_name))
        model_no_ddp.cnn.load_state_dict(checkpoint['model'])
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        if cnn_scheduler is not None and 'scheduler' in checkpoint.keys():
            cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        cnn_epoch = checkpoint['epoch']
        cnn_best_lost = checkpoint['valid_loss']
        print("===== {} Checkpoint loaded =====".format(model_no_ddp.cnn.net_name))
        
        with open(ckpt_CNN + "/Log.txt", 'r') as f:
            cnn_last_line = f.read().splitlines()[-1]
    
    else:
        print("There is not checkpoint for {}".format(model_no_ddp.cnn.net_name))

    # ================ TRANSFORMER
    epoch = -1
    best_lost = 2**16
    trans_last_line = 0
    if os.path.exists('{}/{}_model.pt'.format(ckpt_irgs_trans, model_no_ddp.transformer.net_name)):
        checkpoint = torch.load('{}/{}_model.pt'.format(ckpt_irgs_trans, model_no_ddp.transformer.net_name))
        model_no_ddp.transformer.load_state_dict(checkpoint['model'])
        trans_optimizer.load_state_dict(checkpoint['optimizer'])
        if trans_scheduler is not None and 'scheduler' in checkpoint.keys():
            trans_scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        best_lost = checkpoint['valid_loss']
        print("===== {} Checkpoint loaded =====".format(model_no_ddp.transformer.net_name))
        
        with open(ckpt_irgs_trans + "/Log.txt", 'r') as f:
            trans_last_line = f.read().splitlines()[-1]
    else:
        print("There is not checkpoint for {}".format(model_no_ddp.transformer.net_name))

    # ================ CHECK IF THE MODELS WERE ALREADY COMPLETELY TRAINED
    assert trans_last_line != "end_to_end training finished", \
        "The {} model has already been trained".format(model_no_ddp.net_name)
    assert trans_last_line != "transformer training finished", \
        "The {} model has already been trained".format(model_no_ddp.transformer.net_name)
    if cnn_last_line == "cnn training finished":
        print("The {} model has already been trained".format(model_no_ddp.cnn.net_name))
        if args.mode == 'multi_stage':
            assert args.stage != 'cnn', 'Stage cnn already trained'

    if cnn_last_line != 0:
        with open(ckpt_CNN + "/Log.txt", "a") as f: 
            f.write("===== {} Checkpoint loaded =====\n".format(model_no_ddp.cnn.net_name))
    if trans_last_line != 0: 
        with open(ckpt_irgs_trans + "/Log.txt", "a") as f: 
            f.write("===== {} Checkpoint loaded =====\n".format(model_no_ddp.transformer.net_name))
    
#%% ============== PREPARING THE DATA =============== #
    dataset, sampler, loader = aux_obj(), aux_obj(), aux_obj()

    _, _, _, norm_params = Load_data(args, norm=True)
    dataset.train, dataset.val, dataset.test, _ = Load_data(args, norm=False)

    # =========== Image segmentation using IRGS
    start_time = time.time()
    # Extract_segments(dataset.train, norm_params, args)
    # Extract_segments(dataset.val, norm_params, args)
    # Extract_segments(dataset.test, norm_params, args)
    print('parallel time:', time.time() - start_time)

    # =========== Datasets matching images + segments
    dataset.train = Load_patches_segments(dataset.train.file_paths)
    dataset.val   = Load_patches_segments(dataset.val.file_paths)
    dataset.test  = Load_patches_segments(dataset.test.file_paths)

#%% ============== DATALOADERS =============== #
    if ngpus_per_node > 1:
        sampler.train = torch.utils.data.distributed.DistributedSampler(dataset.train, shuffle=True)
        sampler.val   = torch.utils.data.distributed.DistributedSampler(dataset.val,   shuffle=False)
        sampler.test  = torch.utils.data.distributed.DistributedSampler(dataset.test,  shuffle=False)
    
    loader.train = data.DataLoader(dataset=dataset.train, batch_size=args.batch_size, shuffle=(sampler.train is None), 
                                   num_workers=args.num_workers, sampler=sampler.train)
    loader.val = data.DataLoader(dataset=dataset.val, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=args.num_workers, sampler=sampler.val)
    loader.test = data.DataLoader(dataset=dataset.test, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=args.num_workers, sampler=sampler.test)

#%% ============== SAVE NORM PARAMS =============== #
    joblib.dump(norm_params, ckpt_CNN + '/norm_params.pkl')
    joblib.dump(norm_params, ckpt_irgs_trans + '/norm_params.pkl')

#%% ============== TRAINING =============== #
    if args.mode == 'end_to_end':

        if not args.sweep:
            project_name = '_'.join(sum([i.split('/') for i in ckpt_irgs_trans.split('\\')], [])[-4:])
            wandb.init(project=project_name)

        with open(ckpt_irgs_trans + '/commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        _, _, _, _ = Train_loop(loader, sampler, model, model_no_ddp, best_lost,
                             cnn_optimizer, cnn_scheduler, trans_optimizer, trans_scheduler,
                             args, ckpt_irgs_trans, epoch, stage='end_to_end', loss_term=args.loss_term)

    elif args.stage == 'cnn':
        if not args.sweep:
            project_name = '_'.join(sum([i.split('/') for i in ckpt_CNN.split('\\')], [])[-4:])
            wandb.init(project=project_name)

        with open(ckpt_CNN + '/commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        model, model_no_ddp, \
        cnn_optimizer, _ = Train_loop(loader, sampler, model, model_no_ddp, cnn_best_lost,
                                        cnn_optimizer, cnn_scheduler, None, None,
                                        args, ckpt_CNN, cnn_epoch, stage='cnn')
        
    elif args.stage == 'transformer':

        if not args.sweep:
            project_name = '_'.join(sum([i.split('/') for i in ckpt_irgs_trans.split('\\')], [])[-4:])
            wandb.init(project=project_name)

        with open(ckpt_irgs_trans + '/commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        _, _, _, _ = Train_loop(loader, sampler, model, model_no_ddp, best_lost,
                                None, None, trans_optimizer, trans_scheduler,
                                args, ckpt_irgs_trans, epoch, stage='transformer')
    

if __name__ == '__main__':

    args = Arguments()

    if args.sweep:
        #%% HYPER-PARAMETER TUNNING
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {'goal': 'minimize', 'name': 'val_loss'},
            'parameters': {
                'alpha':        {'min': 0.2,  'max': 5.0,   'distribution': 'uniform'},
                'lr':           {'min': 1e-6, 'max': 1e-3,  'distribution': 'uniform'},
                'batch_size':   {'values': [4, 8, 16, 32]}
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='Hyper-parameter tunning')
        wandb.agent(sweep_id, function=main, count=1)
    else:
        main()
