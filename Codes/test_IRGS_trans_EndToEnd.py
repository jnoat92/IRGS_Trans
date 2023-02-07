import joblib
import torch
import torch.utils.data as data
import numpy as np
import os, argparse
from config import Arguments_test

from tqdm import tqdm
from utils.dataloader import RadarSAT2_Dataset, irgs_segments_parallel, Extract_segments, \
                             Load_patches_segments, Enhance_image
from utils.utils import Metrics, Map_labels, hex_to_rgb, \
                        subplot_grid, boolean_string
from major_voting import MV_per_CC, MV_per_Class
from PIL import Image
from icecream import ic

from lib.unet.unet_model import UNet
from lib.custom_networks import Trans_no_patch_embed, IRGS_Trans
from magic_irgs import IRGS

import matplotlib.pyplot as plt
from matplotlib import cm
from mycolorpy import colorlist as mcp
from matplotlib.colors import LinearSegmentedColormap

import copy
import json
import shutil
import time
from numba import cuda, jit
import numba
import wandb


def M_Voting(test_data, token_option, n_tokens, segments, cnn_pred, 
             landmask_idx, output_folder):
        
    if token_option == 'superpixels':
        # ------ CNN
        # m_v_per_CC_cnn_ = MV_per_CC().do(segments, n_tokens, cnn_pred, background=-1)            # per superpixels
        m_v_per_CC_cnn = MV_per_CC().CPU_parallel(segments, n_tokens, cnn_pred.astype('int64'), background=-1)            # per superpixels
        colored_m_v_per_CC_cnn = test_data.class_colors[m_v_per_CC_cnn.astype(int)+1]
        colored_m_v_per_CC_cnn[landmask_idx] = 0
        Image.fromarray(colored_m_v_per_CC_cnn).save(output_folder + '/CNN_colored_m_v_per_CC.png')

        # ------ Ground truth        
        # m_v_per_CC_ = MV_per_CC().do(segments, n_tokens, test_data.gts, background=-1)           # per superpixels
        m_v_per_CC = MV_per_CC().CPU_parallel(segments, n_tokens, test_data.gts.astype('int64'), background=-1)           # per superpixels
        colored_m_v_per_CC = test_data.class_colors[m_v_per_CC.astype(int)+1]
        colored_m_v_per_CC[landmask_idx] = 0
        Image.fromarray(colored_m_v_per_CC).save(output_folder + '/GT_colored_m_v_per_CC.png')

        Metrics( test_data.gts[landmask_idx==0], 
                m_v_per_CC_cnn[landmask_idx==0], 
                "CNN-MV-superpixels          ", output_folder)

    elif token_option == 'clusters':
        # ------ CNN
        m_v_per_Class_cnn = MV_per_Class(segments, cnn_pred)                 # cluster
        colored_m_v_per_Class_cnn = test_data.class_colors[m_v_per_Class_cnn.astype(int)+1]
        colored_m_v_per_Class_cnn[landmask_idx] = 0
        Image.fromarray(colored_m_v_per_Class_cnn).save(output_folder + '/CNN_colored_m_v_per_Class.png')
        
        # ------ Ground truth        
        m_v_per_Class = MV_per_Class(segments, id, test_data.gts)                # per cluster
        colored_m_v_per_Class = test_data.class_colors[m_v_per_Class.astype(int)+1]
        colored_m_v_per_Class[landmask_idx] = 0
        Image.fromarray(colored_m_v_per_Class).save(output_folder + '/GT_colored_m_v_per_Class.png')

        Metrics(    test_data.gts[landmask_idx==0], 
                m_v_per_Class_cnn[landmask_idx==0], 
                "CNN-MV-clusters             ", output_folder)

    '''
    CC_Class_dif = np.uint8(255*(m_v_per_CC != m_v_per_Class))  # difference
                                                                # if all() == '0's then the MV using clusters
                                                                # is consistent with the ground truth
    Image.fromarray(CC_Class_dif).save(output_folder + '/GT_CC_Class_dif.png')

    a = m_v_per_CC[landmask_idx==False]
    b = m_v_per_Class[landmask_idx==False]
    print_line = "Similarity m_v_per_CC vs m_v_per_Class: %.2f %%\n"%(100 * np.sum(a==b)/len(a))
    with open(output_folder + '/log.txt', 'w') as f:
        f.write(print_line)
    print(print_line)
    '''


def Build_Prediction_map_Sequential(tokens_ids, segments, logits):
    
    logits_map = np.zeros((segments.shape[0], segments.shape[1], logits.shape[1])).astype(logits.dtype)
    for j in tqdm(range(len(tokens_ids)), ncols=50):
        pos = segments == tokens_ids[j]
        logits_map[pos] = logits[j]
    return logits_map

def Build_Prediction_map_CPU_parallel(tokens_ids, segments, logits):
    print('CPU parallel...')
    
    @jit(nopython=True, parallel=True)
    def CPU_parallel(tokens_ids, segments, logits):
        logits_map = np.zeros((segments.shape[0], segments.shape[1], logits.shape[1])).astype(logits.dtype)
        for j in numba.prange(len(tokens_ids)):
            pos = np.where(segments == tokens_ids[j])
            for k, l in zip(pos[0], pos[1]):
                for m in range(logits.shape[1]):
                    logits_map[k,l,m] = logits[j,m]
        return logits_map
    
    return CPU_parallel(tokens_ids, segments, logits)
        
def Build_Prediction_map_GPU(tokens_ids, segments, logits, blocks_per_grid=32, threads_per_block=128):

    @cuda.jit
    def GPU(tokens_ids_d, segments_d, logits_d, logits_map_d):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for j in range(start, len(tokens_ids_d), stride):
            for k in range(segments_d.shape[0]):
                for l in range(segments_d.shape[1]):
                    for m in range(segments_d.shape[2]):
                        if segments_d[k,l,m] == tokens_ids_d[j]:
                            for n in range(logits_d.shape[1]):
                                logits_map_d[k,l,m,n] = logits_d[j,n]
    
    tokens_ids_d = cuda.to_device(tokens_ids)
    segments_d   = cuda.to_device(segments)
    logits_d     = cuda.to_device(logits)
    
    logits_map   = np.zeros((segments.shape[0], segments.shape[1], 
                             segments.shape[2], logits.shape[1])).astype(logits.dtype)
    logits_map_d = cuda.device_array_like(logits_map)

    GPU[blocks_per_grid, threads_per_block] \
        (tokens_ids_d, segments_d, logits_d, logits_map_d)
    # cuda.synchronize()
    return logits_map_d.copy_to_host()


def Inference(model, data_loader, Logits_ave, count_mat,
              args, patches_idx, device='cuda'):

    assert args.stage in ['cnn', 'transformer'], \
        "Not valid stage value"

    model.eval()

    # Data
    n_batches = len(data_loader)
    data_iterator = iter(data_loader)
    idx_iterator = iter(patches_idx)

    # Batch loop
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
            cnn_logits, trans_logits, _, _, tokens_ids, seg = model(images, gts, copy.deepcopy(segments), n_tokens, 
                                                             stage=args.stage, device=device)

        if args.stage == 'cnn': 
            logits_map = cnn_logits
        elif args.stage == 'transformer':

            trans_logits = trans_logits.view(-1, trans_logits.shape[-1])

            # REMOVE PADDED TOKENS
            for j in range(n_tokens.shape[0]):
                if not j:
                    nonpad_trans_logits = trans_logits[:n_tokens[j]]
                else:
                    nonpad_trans_logits = torch.cat((nonpad_trans_logits, trans_logits[:n_tokens[j]]), 0)

                aux = n_tokens[j] % args.max_length
                sample_j_tokens = n_tokens[j] + (args.max_length-aux) if aux else n_tokens[j]
                trans_logits = trans_logits[sample_j_tokens:]
            # BUILD PREDICTION MAP PER SAMPLE IMAGE
            tokens_ids = tokens_ids.detach().cpu().numpy()
            seg = seg.detach().cpu().numpy()
            nonpad_trans_logits = nonpad_trans_logits.detach().cpu().numpy()
            logits_map = Build_Prediction_map_GPU(tokens_ids, seg, nonpad_trans_logits, 
                                                  blocks_per_grid = 128, threads_per_block = 256)
            logits_map = torch.from_numpy(logits_map).to(device)
            logits_map = torch.permute(logits_map, (0, 3, 1, 2))

        # LOCATE PREDICTED LOGITS ON THE WHOLE SCENE
        for j in range(logits_map.shape[0]):
            y1, y2, x1, x2 = next(idx_iterator)
            Logits_ave[:, y1:y2, x1:x2] += logits_map[j, :, 0:(y2-y1), 0:(x2-x1)]
            count_mat[y1:y2, x1:x2] += 1

    # print("Maximun number of predictions per pixel: {0}".format(count_mat.max()))
    Logits_ave /= count_mat
    probs_map = Logits_ave.softmax(0)
    pred_map = torch.argmax(probs_map, 0)

    return probs_map.detach().cpu().numpy(), pred_map.detach().cpu().numpy()


class Slide_patches_index(data.Dataset):
    def __init__(self, data, patch_size, overlap_percent):
        super(Slide_patches_index, self).__init__()

        h_img, w_img, _ = data.image.shape
        self.h_crop = patch_size if patch_size < h_img else h_img
        self.w_crop = patch_size if patch_size < w_img else w_img

        self.h_stride = self.h_crop - round(self.h_crop * overlap_percent) if self.h_crop < h_img else h_img
        self.w_stride = self.w_crop - round(self.w_crop * overlap_percent) if self.w_crop < w_img else w_img

        self.h_grids = max(h_img - self.h_crop + self.h_stride - 1, 0) // self.h_stride + 1
        self.w_grids = max(w_img - self.w_crop + self.w_stride - 1, 0) // self.w_stride + 1

        self.patches_list = []
        
        for h_idx in range(self.h_grids):
            for w_idx in range(self.w_grids):
                y1 = h_idx * self.h_stride
                x1 = w_idx * self.w_stride
                y2 = min(y1 + self.h_crop, h_img)
                x2 = min(x1 + self.w_crop, w_img)
                y1 = max(y2 - self.h_crop, 0)
                x1 = max(x2 - self.w_crop, 0)

                self.patches_list.append((y1, y2, x1, x2))

    def __getitem__(self, index):
        return self.patches_list[index]
    
    def __len__(self):
        return len(self.patches_list)

class Slide_patches(data.Dataset):
    def __init__(self, data, patches_idx):
        super(Slide_patches, self).__init__()

        self.data = data
        self.h_crop, self.w_crop = patches_idx.h_crop, patches_idx.w_crop
        self.patches_list = patches_idx.patches_list
        self.file_paths = []

    def __getitem__(self, index):
        y1, y2, x1, x2 = self.patches_list[index]
        im = self.data.image[y1:y2, x1:x2]
        gt = self.data.gts[y1:y2, x1:x2]
        bc = self.data.background[y1:y2, x1:x2]

        if self.h_crop > np.abs(y1-y2):
            im = np.pad(im, (0, self.h_crop - np.abs(y1-y2), 0, 0), 'symmetric')
            gt = np.pad(gt, (0, self.h_crop - np.abs(y1-y2), 0, 0), 'symmetric')
            bc = np.pad(bc, (0, self.h_crop - np.abs(y1-y2), 0, 0), 'symmetric')

        if self.w_crop > np.abs(x1-x2):
            im = np.pad(im, (0, 0, 0, self.w_crop - np.abs(x1-x2)), 'symmetric')
            gt = np.pad(gt, (0, 0, 0, self.w_crop - np.abs(x1-x2)), 'symmetric')
            bc = np.pad(bc, (0, 0, 0, self.w_crop - np.abs(x1-x2)), 'symmetric')

        return im, gt, bc
    
    def save_samples(self, folder):
        # Remove previous files
        if os.path.exists(folder): shutil.rmtree(folder)
        # Create directory
        os.makedirs(folder, exist_ok=True)

        for k in tqdm(range(len(self.patches_list)), ncols=50):
            
            im, gt, bc =  self.__getitem__(k)
            data_dict = {}
            data_dict["img"] = im
            data_dict["lbl"] = gt
            data_dict["bck"] = bc       # '0' values mask unlabeled pixels or pixels that we do not use
            
            file_name = folder + "/{:05d}.pkl".format(k)
            joblib.dump(data_dict, file_name)
            self.file_paths.append(file_name)


    def __len__(self):
        return len(self.patches_list)


if __name__ == '__main__':

    args = Arguments_test()

    if args.stage == 'cnn' and args.mode == 'end_to_end' and args.loss_term == 'transformer':
        raise AssertionError("Model trained using ONLY TRANSFORMER LOSS does not work for the cnn prediction")

#%% ============== BUILD MODELS =============== #
    cnn = UNet(args.in_chans, args.n_classes)
    transformer = Trans_no_patch_embed(n_in_feat=32, num_classes=args.n_classes, 
                                        embed_dim=args.embed_dim, depth=args.trans_depth, 
                                        num_heads=args.num_heads)
    # ================ MERGE CNN - TRANSFORMER
    model = IRGS_Trans(cnn, transformer, args.max_length, 
                                  args.mix_images, args.random_tokens)
    model.cuda()
    # ic(next(model.parameters()).device)


#%% ============== LOAD CHECKPOINT =============== #
    if args.mode == 'end_to_end':
        if args.loss_term == 'transformer':
            ckpt_irgs_trans = os.path.join(args.ckpt_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'end_to_end', 'Loss_transformer', args.model_name)
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'end_to_end', 'Loss_transformer', args.model_name)
        else:
            ckpt_irgs_trans = os.path.join(args.ckpt_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'end_to_end', 'Loss_end_to_end', args.model_name)
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                           model.net_name + '_' + args.token_option, 
                                           'end_to_end', 'Loss_end_to_end', args.model_name)
        ckpt_CNN = ckpt_irgs_trans
    
        project_name = '-'.join([model.net_name, args.token_option, args.mode, 'Loss_' + args.loss_term])
        wandb.init(project=project_name, name=args.stage, group=args.model_name, job_type='test')

    elif args.mode == 'multi_stage':
        ckpt_irgs_trans = os.path.join(args.ckpt_path, args.Dataset_name, 
                                       model.net_name + '_' + args.token_option, 'multi_stage', args.model_name)
        ckpt_CNN = os.path.join(args.ckpt_path, args.Dataset_name, cnn.net_name, args.model_name)

        if args.stage == 'transformer':
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, 
                                        model.net_name + '_' + args.token_option, 'multi_stage', args.model_name)
            project_name = '-'.join([model.net_name, args.token_option, args.mode])
            wandb.init(project=project_name, name=args.stage, group=args.model_name, job_type='test')

        elif args.stage == 'cnn':
            args.save_path =  os.path.join(args.save_path, args.Dataset_name, cnn.net_name, args.model_name)
            wandb.init(project=cnn.net_name, name=args.stage, group=args.model_name, job_type='test')
        
    # ================ CNN
    if os.path.exists('{}/{}_model.pt'.format(ckpt_CNN, model.cnn.net_name)):
        with open(ckpt_CNN + "/Log.txt", 'r') as f:
            cnn_last_line = f.read().splitlines()[-9:]
            assert "cnn training finished" in cnn_last_line or \
                   "end_to_end training finished" in cnn_last_line, \
                "*** The %s for network %s has NOT BEEN trained completely ***"%(args.model_name, model.cnn.net_name)

        checkpoint = torch.load('{}/{}_model.pt'.format(ckpt_CNN, model.cnn.net_name), map_location=torch.device('cuda'))
        model.cnn.load_state_dict(checkpoint['model'])
        print("===== {} Checkpoint loaded =====".format(model.cnn.net_name))

        norm_params = joblib.load(ckpt_CNN + '/norm_params.pkl')
        
    else: raise AssertionError("There is not checkpoint for {}".format(model.cnn.net_name))

    # ================ TRANSFORMER
    if not (args.mode == 'multi_stage' and args.stage == 'cnn'):
        if os.path.exists('{}/{}_model.pt'.format(ckpt_irgs_trans, model.transformer.net_name)):
            with open(ckpt_irgs_trans + "/Log.txt", 'r') as f:
                trans_last_line = f.read().splitlines()[-9:]
                assert "transformer training finished" in trans_last_line or \
                       "end_to_end training finished" in trans_last_line, \
                    "*** The %s for network %s has NOT BEEN trained completely ***"%(args.model_name, model.transformer.net_name)

            checkpoint = torch.load('{}/{}_model.pt'.format(ckpt_irgs_trans, model.transformer.net_name), map_location=torch.device('cuda'))
            model.transformer.load_state_dict(checkpoint['model'])
            print("===== {} Checkpoint loaded =====".format(model.transformer.net_name))
        
            norm_params = joblib.load(ckpt_irgs_trans + '/norm_params.pkl')
            
        else: raise AssertionError("There is not checkpoint for {}".format(model.transformer.net_name))

    if not args.use_gpu: model.cpu()
    model.eval()

#%% ============== TESTING =============== #    
    for i in args.test_path:
        
        print('evaluating model: ', args.model_name)
        
        output_folder = os.path.join(args.save_path, i)
        if args.mode == 'end_to_end' and args.loss_term == 'end_to_end':
            output_folder = os.path.join(output_folder, args.stage)
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        with open(output_folder + '/commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
                
        # =========== LOAD DATA
        test_data =  RadarSAT2_Dataset(args, name = i, set_="test")
        patches_idx = Slide_patches_index(copy.deepcopy(test_data), args.patch_size, args.patch_overlap)
        patches = Slide_patches(copy.deepcopy(test_data), patches_idx)

        scene_data_dir = './results_test/Test_samples/'
        if os.path.exists('/home/' + os.getenv('LOGNAME') + '/scratch/'):
            scene_data_dir = scene_data_dir + '_'.join(output_folder.split('/')[1:])
            scene_data_dir = os.path.join('/home/' + os.getenv('LOGNAME') + '/scratch/', os.getenv('LOGNAME'), scene_data_dir[2:])
        patches.save_samples(scene_data_dir)

        # =========== IMAGE SEGMENTATION USING IRGS
        Extract_segments(patches, norm_params, args)

        # =========== NEW DATALOADERS MATCHING IMAGES + SEGMENTS
        patches_segments = Load_patches_segments(patches.file_paths)
        patches_segments_loader = data.DataLoader(dataset=patches_segments, batch_size=args.batch_size, 
                                                  shuffle=False, num_workers=args.num_workers)

        # =========== INFERENCE
        device='cuda' if args.use_gpu else 'cpu'
        Logits_ave = torch.zeros((args.n_classes, test_data.image.shape[0], 
                                  test_data.image.shape[1])).to(device)
        count_mat = torch.zeros((test_data.image.shape[0], test_data.image.shape[1])).to(device)
        probs_map, pred_map = Inference(model, patches_segments_loader, Logits_ave, count_mat, 
                                        args, patches_idx, device=device)
        
        # ============== METRICS 
        landmask_idx = test_data.background==0
        y_true = test_data.gts[landmask_idx]
        y_pred = pred_map     [landmask_idx]
        Metrics(y_true, y_pred, "Prediction-Map-%s          "%(args.stage), output_folder)

        # ------ SAVE IMAGE
        image_enhanced = Enhance_image(test_data.image, test_data.background)
        Image.fromarray(image_enhanced[:, :, 0]).save(output_folder + "/HH.png")
        Image.fromarray(image_enhanced[:, :, 1]).save(output_folder + "/HV.png")
       
        # ------ SAVE GROUND TRUTH
        colored_gts = test_data.class_colors[test_data.gts.astype(int)+1]
        colored_gts[landmask_idx] = 0
        Image.fromarray(colored_gts).save(output_folder + '/colored_gts.png')
        
        # ------ SAVE PREDICTION
        colored_pred_map = test_data.class_colors[pred_map+1]
        colored_pred_map[landmask_idx] = 0
        Image.fromarray(colored_pred_map).save(output_folder + '/colored_predict_%s.png'%(args.stage))
        np.save(output_folder + '/probabilities_predict_%s.npy'%(args.stage), probs_map)


        if len(patches_idx) == 1:
            _, _, _, segments, n_tokens, boundaries = patches_segments.__getitem__(0)
            segments[np.isinf(segments)] = -1
            
        if args.mode == 'multi_stage' and args.stage == 'cnn':
            # ------ MAJORITY VOTING
            if len(patches_idx) > 1:
                segments, n_tokens, boundaries = irgs_segments_parallel(args.irgs_classes, args.irgs_iter, args.token_option, None, True, 
                                                                        ((test_data.image, 0, test_data.background), 0, 0))
                segments[np.isinf(segments)] = -1
            M_Voting(test_data, args.token_option, n_tokens, segments, pred_map, landmask_idx, output_folder)

        if len(patches_idx) == 1 or (args.mode == 'multi_stage' and args.stage == 'cnn'):
            # ------ SAVE IRGS AND BOUNDARIES
            irgs_colors = mcp.gen_color(cmap="jet", n=n_tokens+1)
            np.random.seed(0); np.random.shuffle(irgs_colors)
            irgs_colors_rgb = np.asarray([hex_to_rgb(i) for i in irgs_colors])

            colored_irgs_output = np.uint8(irgs_colors_rgb[segments.astype('int64')])
            colored_irgs_output[landmask_idx] = 0
            Image.fromarray(colored_irgs_output).save(output_folder + "/colored_irgs_output.png")
            
            Image.fromarray(255*np.uint8((landmask_idx==0)*(boundaries==-1))).save(output_folder + "/irgs_boundaries.png")
            aux = np.uint8(image_enhanced.repeat(3, axis=2)); aux[boundaries==-1] = 2*[255, 165, 0]
            aux[landmask_idx] = 0
            Image.fromarray(aux[:,:,0:3]).save(output_folder + "/HH_boundaries.png")
            Image.fromarray(aux[:,:,3: ]).save(output_folder + "/HV_boundaries.png")

