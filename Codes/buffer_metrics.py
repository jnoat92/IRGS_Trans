'''
No@
Jan 31st, 2024
'''

import os
from config import Arguments_test
from lib.unet.unet_model import UNet
from lib.custom_networks import Trans_no_patch_embed, IRGS_Trans
from icecream import ic
import wandb
from utils.dataloader import RadarSAT2_Dataset
import numpy as np
from skimage.morphology import disk, binary_dilation
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
from utils.utils import Metrics
import csv

def get_contours(lbl):
    for lvl in np.unique(lbl):
        level_ctrs = find_contours(lbl, level=lvl)
        for c in level_ctrs:
            try:
                contours = np.concatenate((contours, c), axis=0)
            except:
                contours = c
    return np.uint16(contours)


if __name__ == '__main__':

    args = Arguments_test()

    if args.stage == 'cnn' and args.mode == 'end_to_end' and args.loss_term == 'transformer':
        raise AssertionError("Model trained using ONLY TRANSFORMER LOSS does not work for the cnn prediction")

#%% ============== MODEL NAME =============== #
    cnn = UNet(args.in_chans, args.n_classes)
    transformer = Trans_no_patch_embed(n_in_feat=32, num_classes=args.n_classes, 
                                        embed_dim=args.embed_dim, depth=args.trans_depth, 
                                        num_heads=args.num_heads)
    # ================ MERGE CNN - TRANSFORMER
    model = IRGS_Trans(cnn, transformer, args.max_length, 
                                  args.mix_images, args.random_tokens)

#%% ============== DIRECTORY =============== #
    if args.mode == 'end_to_end':
        if args.loss_term == 'transformer':
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'end_to_end', 'Loss_transformer', args.model_name)
        else:
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'end_to_end', 'Loss_end_to_end', args.model_name)
        project_name = '-'.join([model.net_name, args.token_option, args.mode, 'Loss_' + args.loss_term])

    elif args.mode == 'multi_stage':
        if args.loss_term == 'end_to_end':
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'multi_stage', 'Loss_end_to_end', args.model_name)
            project_name = '-'.join([model.net_name, args.token_option, args.mode, 'Loss_' + args.loss_term])

        elif args.loss_term == 'transformer':
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'multi_stage', 'Loss_transformer', args.model_name)
            project_name = '-'.join([model.net_name, args.token_option, args.mode, 'Loss_' + args.loss_term])
            args.stage = 'transformer'

        elif args.loss_term == 'cnn':
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, cnn.net_name, args.model_name)
            args.stage = 'cnn'
            project_name = cnn.net_name

    # ic(args.save_path)

#%% ============== TESTING =============== #
    output_folder = os.path.join(args.save_path, args.test_path[0])
    if args.loss_term == 'end_to_end':
        output_folder = os.path.join(output_folder, args.stage)

    # =========== LOAD DATA AND PRED MAPS
    test_data =  RadarSAT2_Dataset(args, name = args.test_path[0], set_="test")
    landmask_idx = test_data.background==0
    probs_map = np.load(output_folder + '/probabilities_predict_%s.npy'%(args.stage))
    pred_map = np.argmax(probs_map, 0)

    if len(np.unique(test_data.gts[landmask_idx==0])) > 1:      # single class scenes don't show edges

        output_folder = os.path.join(output_folder, 'buffers')
        os.makedirs(output_folder, exist_ok=True)

        wandb.init(project=project_name, name=args.stage, group=args.model_name, job_type='edge_buffer')

        # =========== CREATE BUFFER
        contours = get_contours(test_data.gts)
        contour_mask = np.zeros_like(test_data.gts)
        contour_mask[contours[:,0], contours[:,1]] = 1

        for width in range(10, 20):

            edge_buffer = np.uint8(binary_dilation(contour_mask, disk(width)))
            edge_buffer[landmask_idx] = 0
            Image.fromarray(np.uint8(edge_buffer*255)).save(output_folder + '/edge_buffer_%d.png'%(width))

            # =========== METRICS
            y_true = test_data.gts[edge_buffer==1]
            y_pred = pred_map[edge_buffer==1]
            acc, _ = Metrics(y_true, y_pred, None, None)

            wandb.summary['buffer_%d_OA'%(width+1)] = acc
            with open(output_folder + '/metrics_buffer.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['buffer_%d_OA'%(width+1), acc])


#python buffer_metrics.py --mode multi_stage --loss_term cnn --stage cnn --test_path 20100605 --model_name model_4