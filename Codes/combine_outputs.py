'''
No@
Feb 28th, 2024
'''

import os
from config import Arguments_test
from lib.unet.unet_model import UNet
from lib.custom_networks import Trans_no_patch_embed, IRGS_Trans
from mycolorpy import colorlist as mcp
from utils.utils import hex_to_rgb, get_contours
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

if __name__ == '__main__':

    args = Arguments_test()

    assert args.loss_term == 'end_to_end', "Only end-to-end protocols can provide combined outputs"

    # =========== MODEL NAME 
    cnn = UNet(args.in_chans, args.n_classes)
    transformer = Trans_no_patch_embed(n_in_feat=32, num_classes=args.n_classes, 
                                        embed_dim=args.embed_dim, depth=args.trans_depth, 
                                        num_heads=args.num_heads)
    # =========== MERGE CNN - TRANSFORMER
    model = IRGS_Trans(cnn, transformer, args.max_length, 
                                  args.mix_images, args.random_tokens)

    # =========== DIRECTORY
    # Only end-to-end protocols can provide combined outputs
    if args.mode == 'end_to_end':
        args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                        model.net_name + '_' + args.token_option, 
                                        'end_to_end', 'Loss_end_to_end', args.model_name)

    elif args.mode == 'multi_stage':
        args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                        model.net_name + '_' + args.token_option, 
                                        'multi_stage', 'Loss_end_to_end', args.model_name)

    project_name = '-'.join([model.net_name, args.token_option, args.mode, 'Loss_' + args.loss_term])
    output_folder = os.path.join(args.save_path, args.test_path[0])
    # ic(output_folder)

    # =========== LOAD DATA AND PRED MAPS
    test_data =  RadarSAT2_Dataset(args, name = args.test_path[0], set_="test")
    landmask_idx = test_data.background==0
    cnn_probs_map = np.load(output_folder + '/cnn/probabilities_predict_cnn.npy')
    trans_probs_map = np.load(output_folder + '/transformer/probabilities_predict_transformer.npy')

    output_folder = os.path.join(output_folder, 'combined output')
    os.makedirs(output_folder, exist_ok=True)

    colors = mcp.gen_color(cmap="jet", n=256)
    colors_rgb = np.asarray([hex_to_rgb(i) for i in colors])

    # =========== GENERATE BUFFERS
    contours = get_contours(test_data.gts)
    contour_mask = np.zeros_like(test_data.gts)
    contour_mask[contours[:,0], contours[:,1]] = 1
    edge_buffer = np.zeros((20, test_data.gts.shape[0], test_data.gts.shape[1]))
    for width in range(0, 20):
        edge_buffer[width] = np.uint8(binary_dilation(contour_mask, disk(width)))
        edge_buffer[width][landmask_idx] = 0
        plt.imshow(edge_buffer[width])
        plt.savefig('%d'%(width), dpi=200)
    

    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

        # =========== COMBINED MAP
        wandb.init(project=project_name, name='%.1fcnn_%.1ftrans'%(i, 1-i), group=args.model_name, job_type='combine_score')
        combined_prob = i*cnn_probs_map + (1-i)*trans_probs_map

        # =========== SOFT-LABELS
        colored_softlbl = np.uint8(colors_rgb[np.uint8(255*combined_prob[1])])
        colored_softlbl[landmask_idx] = [255, 255, 255]
        Image.fromarray(colored_softlbl).save(output_folder + '%.1f_soft_lbl.png'%(i))
        exit()

        # =========== METRICS
        pred_map = np.argmax(combined_prob, 0)
        y_true = test_data.gts[landmask_idx==0]
        y_pred = pred_map[landmask_idx==0]
        acc, _ = Metrics(y_true, y_pred, None, None)
        wandb.summary['OA'] = acc

        # =========== BUFFER METRICS
        if len(np.unique(test_data.gts[landmask_idx==0])) > 1:      # single class scenes don't show edges
            for width in range(0, 20):
                y_true = test_data.gts[edge_buffer[width]==1]
                y_pred = pred_map[edge_buffer[width]==1]
                acc, _ = Metrics(y_true, y_pred, None, None)
                wandb.summary['buffer_%d_OA'%(width+1)] = acc

# python combine_outputs.py --mode multi_stage --loss_term end_to_end  --test_path 20100605 --model_name model_4