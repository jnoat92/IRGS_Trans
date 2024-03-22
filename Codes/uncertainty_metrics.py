'''
No@
March 21st, 2024
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

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.utils import Metrics, get_contours, hex_to_rgb
from mycolorpy import colorlist as mcp
import csv
from scipy.stats import entropy

colors = mcp.gen_color(cmap="jet", n=256)
colors_rgb = np.asarray([hex_to_rgb(i) for i in colors])

def histogram(data, name, norm_ref):
    # nbins = int(2*(norm_ref**(1/3))) #Rice criterion
    hist, bins = np.histogram(data, bins=np.linspace(0, 1, 513))
    plt.plot(bins[1:], hist/norm_ref)
    plt.title("{}".format(name))
    plt.xlabel('bins')
    plt.ylabel('count')
    plt.tight_layout()
    
    plt.savefig(output_folder + "/%s.png"%(name), dpi=500)
    plt.close()
    np.save(output_folder + "/%s.npy"%(name), hist)
    wandb.save(os.path.split(os.getcwd())[0] + output_folder[2:] + "/%s.npy"%(name))
    


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
    wandb.init(project=project_name, name=args.stage, group=args.model_name, job_type='uncertainty')

#%% ============== TESTING =============== #
    output_folder = os.path.join(args.save_path, args.test_path[0])
    if args.loss_term == 'end_to_end':
        output_folder = os.path.join(output_folder, args.stage)
    ic(output_folder)

    # =========== LOAD PROBABILITY MAP
    test_data =  RadarSAT2_Dataset(args, name = args.test_path[0], set_="test")
    landmask_idx = test_data.background==0
    probs_map = np.load(output_folder + '/probabilities_predict_%s.npy'%(args.stage))

    output_folder = os.path.join(output_folder, 'uncertainty')
    os.makedirs(output_folder, exist_ok=True)

    # # =========== ENTROPY
    entropy_ = entropy(probs_map, base=2, axis=0)
    colored_entropy = np.uint8(colors_rgb[np.uint8(255*entropy_)])
    colored_entropy[landmask_idx] = [255, 255, 255]
    Image.fromarray(colored_entropy).save(output_folder + '/entropy_%s.png'%(args.stage))

    # # =========== CONFIDENCE HISTOGRAMS
    pred_map = np.argmax(probs_map, axis=0)
    pred_map[landmask_idx] = 1; test_data.gts[landmask_idx] = 0
    correct_pixels_idx = pred_map == test_data.gts
    pred_map[landmask_idx] = 0
    misscla_pixels_idx = pred_map != test_data.gts

    correct_pixels_entropy = entropy_[correct_pixels_idx]
    misscla_pixels_entropy = entropy_[misscla_pixels_idx]
    all_pixels_entropy = entropy_[landmask_idx==0]

    histogram(all_pixels_entropy, "conf_hist_entropy", len(all_pixels_entropy))
    histogram(correct_pixels_entropy, "conf_hist_correct_pixels", len(all_pixels_entropy))
    histogram(misscla_pixels_entropy, "conf_hist_misscla_pixels", len(all_pixels_entropy))

    wandb.summary['Average_entropy'] = all_pixels_entropy.mean()
    wandb.summary['Std_entropy']     = all_pixels_entropy.std()


#python uncertainty_metrics.py --mode multi_stage --loss_term cnn --stage cnn --test_path 20100605 --model_name model_4
