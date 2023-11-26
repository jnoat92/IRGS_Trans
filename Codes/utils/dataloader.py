import os
from tkinter import image_names
from PIL import Image
from matplotlib import patches
from sklearn.feature_extraction import image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# Added by No@
# -----------------------------
import itertools
import scipy.io as sio
from icecream import ic
from sklearn.preprocessing._data import _handle_zeros_in_scale
import joblib
import glob
from tqdm import tqdm
import shutil
import multiprocessing
from functools import partial
from utils.utils import plot_hist, Parallel, aux_obj
from utils.maxRectangle import maxRectangle
import pandas as pd
from matplotlib import cm
from skimage import measure


# ============ NORMALIZATION ============
def filter_outliers(img, bins=2**16-1, bth=0.001, uth=0.999, train_pixels=None):
    img[np.isnan(img)] = np.mean(img) # Filter NaN values.
    rows, cols, bands = img.shape

    if train_pixels is None:
        h = np.arange(0, rows)
        w = np.arange(0, cols)
        train_pixels = np.asarray(list(itertools.product(h, w))).transpose()

    min_value, max_value = [], []
    for band in range(bands):
        hist = np.histogram(img[train_pixels[0], train_pixels[1], band].ravel(), bins=bins) # select training pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        min_value.append(hist[1][len(cum_hist[cum_hist<bth])])
        max_value.append(hist[1][len(cum_hist[cum_hist<uth])])
        
    return [np.array(min_value), np.array(max_value)]

def median_filter(img, clips, mask):
    kernel_size = 50
    outliers = ((img < clips[0]) + (img > clips[1])) * np.expand_dims(mask, axis=2)
    # plt.imshow(outliers[:,:,0], cmap='gray')
    # plt.imshow(outliers[:,:,1], cmap='gray')
    # plt.show()
    out_idx = np.asarray(np.where(outliers))

    img_ = img.copy()
    for i in range(out_idx.shape[1]):
        x = out_idx[0][i]
        y = out_idx[1][i]
        a = x - kernel_size//2 if x - kernel_size//2 >=0 else 0
        c = y - kernel_size//2 if y - kernel_size//2 >=0 else 0
        b = x + kernel_size//2 if x + kernel_size//2 <= img.shape[0] else img.shape[0]
        d = y + kernel_size//2 if y + kernel_size//2 <= img.shape[1] else img.shape[1]
        win = img[a:b, c:d][mask[a:b, c:d]==True]
        img_[x, y] = np.median(win, axis=0)
        # img_[x, y] = np.mean(win, axis=0)
    
    return img_

class Min_Max_Norm_Denorm():

    def __init__(self, img, mask, feature_range=[-1, 1]):

        self.feature_range = feature_range
        train_pixels = np.asarray(np.where(mask==0))

        # self.clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.0005, uth=0.9995, train_pixels=train_pixels)
        # img = self.median_filter(img)
        
        self.min_val = np.nanmin(img[train_pixels[0], train_pixels[1]], axis=0)
        self.max_val = np.nanmax(img[train_pixels[0], train_pixels[1]], axis=0)
    
    def median_filter(self, img):
        kernel_size = 25
        outliers = (img < self.clips[0]) + (img > self.clips[1])
        out_idx = np.asarray(np.where(outliers))

        img_ = img.copy()
        for i in range(out_idx.shape[1]):
            x = out_idx[0][i]
            y = out_idx[1][i]
            a = x - kernel_size//2 if x - kernel_size//2 >=0 else 0
            c = y - kernel_size//2 if y - kernel_size//2 >=0 else 0
            b = x + kernel_size//2 if x + kernel_size//2 <= img.shape[0] else img.shape[0]
            d = y + kernel_size//2 if y + kernel_size//2 <= img.shape[1] else img.shape[1]
            img_[x, y] = np.median(img[a:b, c:d], axis=(0, 1))
        
        return img_

    def clip_image(self, img):
        return np.clip(img.copy(), self.clips[0], self.clips[1])

    def Normalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale
        
        # img = self.clip_image(img)
        # img = self.median_filter(img)
        img *= scale
        img += min_
        return img

    def Denormalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale

        img = img.copy() - min_
        img /= scale
        return img

def Enhance_image(img, mask):

    clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.001, uth=0.999, 
                            train_pixels=np.asarray(np.where(mask!=0)))
    img = median_filter(img, clips, mask!=0)
    
    img[mask==0]=[img[:,:,0].min(), img[:,:,1].min()]

    min_ = img[mask!=0].min(0)
    max_ = img[mask!=0].max(0)
    img = np.uint8(255*((img - min_) / (max_ - min_)))
    img[mask == 0] = 0

    return img
    
# ============   SET SPLIT   ============
def Split_Image(rows=5989, cols=2985, no_tiles_h=5, no_tiles_w=5, val_tiles=None):
    '''
    Split the image in tiles to define regions of training and validation
    for dense prediction approaches like semantic seegmentation

    returns mask (same size of the image).
    '''    

    xsz = rows // no_tiles_h
    ysz = cols // no_tiles_w

    # Tiles coordinates
    h = np.arange(0, rows, xsz)
    w = np.arange(0, cols, ysz)
    if (rows % no_tiles_h): h = h[:-1]
    if (cols % no_tiles_w): w = w[:-1]
    tiles = list(itertools.product(h, w))

    if val_tiles is None:
         val_tiles = np.random.randint(0, len(tiles), len(tiles)*20//100)
    # Choose tiles by visual inspection to guaratee all classes in both sets
    # (train and validation)

    mask = np.zeros((rows, cols))       # 0 Training Tiles
                                        # 1 Validation Tiles
    
    for i in val_tiles:
        t = tiles[i]
        finx = rows if (rows-(t[0] + xsz) < xsz) else (t[0] + xsz)
        finy = cols if (cols-(t[1] + ysz) < ysz) else (t[1] + ysz)
        mask[t[0]:finx, t[1]:finy] = 1
    
    new_tiles = []
    for t in tiles:
        finx = rows if (rows-(t[0] + xsz) < xsz) else (t[0] + xsz)
        finy = cols if (cols-(t[1] + ysz) < ysz) else (t[1] + ysz)
        new_tiles.append([t[0], finx, t[1], finy])
        
    return mask, new_tiles

def Split_in_Patches(patch_size, mask, lbl, percent=0, ref_r=0, ref_c=0, padding=True):

    """
    Extract patches coordinates for each set, training, validation, and test

    Everything  in this function is made operating with
    the upper left corner of the patch

    (ref_r, ref_c)  Optional coordinates (patch upper-left corner) from which
                    we start the sliding window in all directions. This guarantees taking
                    the patch (ref_r : ref_r+patch_size, ref_c : ref_c+patch_size)
                    Useful to randomize the patch extraction
    """

    rows, cols = lbl.shape
    # Percent of overlap between consecutive patches.
    overlap = round(patch_size * percent)
    stride = patch_size - overlap

    # Add Padding to the image to match with the patch size
    lower_row_pad = (stride - ref_r        % stride) % stride
    upper_row_pad = (stride - (rows-ref_r-patch_size) % stride) % stride
    lower_col_pad = (stride - ref_c        % stride) % stride
    upper_col_pad = (stride - (cols-ref_c-patch_size) % stride) % stride
    
    pad_tuple_msk = ( (lower_row_pad, upper_row_pad), (lower_col_pad, upper_col_pad) )
    lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')

    # Extract patches coordinates
    new_r, new_c = lbl.shape
    k1 = (new_r-patch_size)//stride
    k2 = (new_c-patch_size)//stride
    print('Total number of patches: %d x %d' %(k1, k2))
    print('Checking divisibility: %d , %d' %((new_r-patch_size)%stride, 
                                             (new_c-patch_size)%stride))

    train_mask, val_mask, test_mask = [np.zeros_like(mask_pad) for i in range(3)]
    train_mask[mask_pad==0] = 1
    val_mask  [mask_pad==1] = 1
    test_mask [mask_pad==2] = 1

    train_patches, val_patches, test_patches = [[] for i in range(3)]
    only_bck_patches = 0

    for i in range(k1):
        if not padding and ((i == 0    and pad_tuple_msk[0][0]) \
                        or  (i == k1-1 and pad_tuple_msk[0][1])): continue
        for j in range(k2):
            if not padding and ((j == 0    and pad_tuple_msk[1][0]) \
                            or  (j == k2-1 and pad_tuple_msk[1][1])): continue
            
            aux = np.sum(lbl[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size] == 0)
            if aux >= 0.75*(patch_size**2):
                only_bck_patches += 1
                continue

            # Train
            if train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                train_patches.append((i*stride, j*stride, patch_size, patch_size))
            # Val                 !!!!!Not necessary with high overlap!!!!!!!!
            elif val_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                val_patches.append((i*stride, j*stride, patch_size, patch_size))
            # Test                !!!!!Not necessary with high overlap!!!!!!!!
            elif test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                test_patches.append((i*stride, j*stride, patch_size, patch_size))
            
    print('Background patches: %d' %(only_bck_patches))
    
    return train_patches, val_patches, test_patches, pad_tuple_msk

def Split_in_Patches_no_padding(patch_size, lbl, tile=None, percent=0, ref_r=0, ref_c=0):

    """
    Extract patches coordinates for an input image tile
    Instead of adding padding for patch_size/overlap divisibility, the remanent is removed

    Everything  in this function is made operating with
    the upper left corner of the patch

    (ref_r, ref_c)  Optional coordinates (patch upper-left corner) from which
                    we start the sliding window in all directions. This guarantees taking
                    the patch (ref_r : ref_r+patch_size, ref_c : ref_c+patch_size)
                    Useful to randomize the patch extraction
    """

    if tile is None:
        tile = [0, 0, lbl.shape[0], lbl.shape[1]]

    ini_x, ini_y, rows, cols = tile
    patch_size = np.array([min(patch_size, rows), min(patch_size, cols)])

    # Overlap between consecutive patches.
    overlap = np.round(patch_size * percent).astype('int32')
    stride = patch_size - overlap

    # Removing borders of the tile that does not match with the patch size/overlap config
    lower_row_pad = (stride[0] - ref_r % stride[0]) % stride[0]
    lower_col_pad = (stride[1] - ref_c % stride[1]) % stride[1]

    lims = (patch_size - np.array([lower_row_pad, lower_col_pad])) % patch_size

    global_ini_x = ini_x + lims[0]
    global_ini_y = ini_y + lims[1]
    lbl = lbl[global_ini_x:ini_x+rows, global_ini_y:ini_y+cols]

    # Extract patches coordinates
    new_r, new_c = lbl.shape
    k1 = np.ceil((new_r-patch_size[0])/stride[0]).astype('int32')
    k2 = np.ceil((new_c-patch_size[1])/stride[1]).astype('int32')
    k1 += not k1 
    k2 += not k2
    print('Total number of patches: %d x %d' %(k1, k2))

    patches = []
    only_bck_patches = 0

    for i in range(k1):
        for j in range(k2):
            aux = np.sum(lbl[i*stride[0]:i*stride[0] + patch_size[0], 
                             j*stride[1]:j*stride[1] + patch_size[1]] == 0)
            if aux >= 0.75*(patch_size[0]*patch_size[1]):
                only_bck_patches += 1
                continue

            patches.append((global_ini_x + i*stride[0], 
                            global_ini_y + j*stride[1], 
                            patch_size[0], 
                            patch_size[1]))

    print('Background patches: %d' %(only_bck_patches))
    
    return patches


# ============   SET SPLIT -- NEWEST - MORE EFFICIENT   ============
class Slide_patches_index(data.Dataset):
    def __init__(self, h_img, w_img, patch_size, overlap_percent):
        super(Slide_patches_index, self).__init__()

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


# ============    DATASETS   ============
def Load_simulated_CPdata_Saeid(image_root, gt_root):

    # Loading simulated CP data Saeid
    image = sio.loadmat(image_root)
    image = image[list(image)[-1]]
    gts  = sio.loadmat(gt_root)
    gts = gts[list(gts)[-1]].astype("float")

    gts -= 1                                # classes [0; n_clases-1]
                                            # background = -1        

    background = np.ones_like(gts)
    background[gts < 0] = 0                 # Mask to cancel background pixels
                                            # background = 0
    gts[gts<0] = 0

    classes = ["Background", "Young ice", "First-year ice", "Multi-year ice", "Open water"]
    class_colors = np.uint8(np.array([[0, 0, 0],           # Background
                                        [204, 0, 255],       # Young ice
                                        [230, 184, 0],       # First year ice
                                        [255, 0, 0],         # Multi-year ice
                                        [255, 204, 239]]))   # Open water
    
    return image, gts, background, classes, class_colors

def Load_21Scenes(scene_dir):

    hh_file = 'imagery_HH_UW_4_by_4_average.tif'
    hv_file = 'imagery_HV_UW_4_by_4_average.tif'
    HH = np.asarray(Image.open(scene_dir + hh_file)).astype(float)
    HV = np.asarray(Image.open(scene_dir + hv_file)).astype(float)
    image = np.concatenate((HH[..., np.newaxis], HV[..., np.newaxis]), axis=2)

    # Noa Labels
    gts_file = glob.glob(scene_dir + 'Noa_labels/*.png')[0]
    gts = np.asarray(Image.open(gts_file)).astype(float)

    # # Max Labels
    # gts_file = glob.glob(scene_dir + 'Max_labels/*.png')[0]
    # lbl = np.asarray(Image.open(gts_file))[:,:,0].astype(float)
    # # Map ground truth labels to 0, 1, 2
    # # Correcting interpolation effect in boundaries
    # gts = np.ones_like(lbl)
    # gts[np.abs(lbl) < 20 ] = 0
    # gts[np.abs(lbl - 149) < 20 ] = 1
    # gts[np.abs(lbl - 255) < 20 ] = 2
    # ###########
    # gts[gts==2] = 1 # consider all ice types in a single class
    # ###########
    # # Image.fromarray(np.uint8(self.gts*255/10)).save(scene_dir + 'Max_labels/labels_corrected_boundaries.tif')
    
    background = np.asarray(Image.open(scene_dir + 'landmask.bmp')).astype(float) / 255
    classes = ["Background", "Open water", "Young ice", "Multi-year ice", ]
    class_colors = np.uint8(np.array([[0, 0, 0],           # Background
                                        [255, 204, 239],     # Open water
                                        [204, 0, 255],       # Young ice
                                        [255, 0, 0]          # Multi-year ice
                                        ]))
    
    return image, gts, background, classes, class_colors

class RadarSAT2_Dataset():
    '''
    Load the dataset -> complete scene
    saves information related to classes like the color composition of segmented maps

    defines training, validation, and test sets

    save samples (sample, label)
    '''
    # def __init__(self, image_root, gt_root, args, name = 'scene_i', phase="train"):
    def __init__(self, args, validation_tiles=None, 
                 name = 'scene_i', set_="train"):

        self.name = name
        self.train_patches, self.val_patches, self.test_patches = [], [], []

        scene_dir = '{}/{}/'.format(args.Datasets_dir, name)
        print("Dataset --- > {}".format(scene_dir))

        # Loading 21 Scenes Dataset
        self.image, self.gts,  self.background, \
            self.classes, self.class_colors = Load_21Scenes(scene_dir)

        if set_ == "train":
            self.patch_size = args.patch_size
            self.patch_overlap = args.patch_overlap
            self.data_info_dir = args.data_info_dir
            os.makedirs(self.data_info_dir, exist_ok=True)
            
            # Split image in Train and validation sets
            # The scene is divided in tiles of equal size aproximatelly
            # Some tiles are chossen for training and others for validation
            rows, cols = self.gts.shape
            no_tiles_h, no_tiles_w = 5, 3
            self.mask_train_val, self.tiles = Split_Image(rows=rows, cols=cols, 
                                                          no_tiles_h=no_tiles_h, 
                                                          no_tiles_w=no_tiles_w, 
                                                          val_tiles=validation_tiles[name])
            
            self.filepath = "{}/{}/".format(self.data_info_dir, self.name)
            os.makedirs(self.filepath, exist_ok=True)
            Image.fromarray(np.uint8(self.mask_train_val*255)).save(self.filepath + "/mask_tr_0_vl_1.png")

            # ========= NORMALIZE IMAGE =========
            self.norm = Min_Max_Norm_Denorm(self.image, self.mask_train_val)
            # joblib.dump(self.norm, self.filepath + '/norm_params.pkl')
            # self.image = self.norm.Normalize(self.image)

    def define_sets(self):
        '''
        Extract patches from each set (these are the images that feed the neural network)
        '''

        # Extract patches coordinates
        for y1, y2, x1, x2 in self.tiles:
            if self.mask_train_val[y1: y2, x1: x2].all():
                patches_idx = Slide_patches_index(y2-y1, x2-x1, self.patch_size, 0)
                self.val_patches.extend(np.array(patches_idx.patches_list) + 
                                        np.array([y1, y1, x1, x1]))
            else:
                patches_idx = Slide_patches_index(y2-y1, x2-x1, self.patch_size, self.patch_overlap)
                self.train_patches.extend(np.array(patches_idx.patches_list) + 
                                          np.array([y1, y1, x1, x1]))
        
        # print("--------------")
        # print("Training Patches:   %d"%(len(self.train_patches)))
        # print("Validation Patches: %d"%(len(self.val_patches)))
        # print("--------------")

        # Efective regions
        mask = 0.5*np.ones_like(self.gts)
        for y1, y2, x1, x2 in self.train_patches:
            mask[y1: y2, x1: x2] = 0.0
        for y1, y2, x1, x2 in self.val_patches:
            mask[y1: y2, x1: x2] = 1.0

        Image.fromarray(np.uint8(mask*255)).save(self.filepath + "/mask_tr_0_vl_1_patches.png")


    def define_sets_irgs_trans(self):
        '''
        2nd:    Extract patches from each set (these are the images that feed the neural network)
        '''

        # Extract patches coordinates
            # training
        print('training patches')
        mask = np.zeros_like(self.mask_train_val).astype(np.uint16)
        mask[self.mask_train_val==0] = 1

        while(mask.any()):
            _, tile = maxRectangle().Calc(mask.copy())
            ini_x, ini_y, rows, cols = tile
            self.train_patches += Split_in_Patches_no_padding(self.patch_size, self.background, tile=tile,
                                                              percent=self.patch_overlap)
            mask[ini_x:ini_x+rows, ini_y:ini_y+cols] = 0
        
            # validation
        print('validation patches')
        mask = np.zeros_like(self.mask_train_val).astype(np.uint16)
        mask[self.mask_train_val==1] = 1
        # Image.fromarray(np.uint8(mask*255)).show()
        while(mask.any()):
            _, tile = maxRectangle().Calc(mask.copy())
            ini_x, ini_y, rows, cols = tile
            self.val_patches += Split_in_Patches_no_padding(self.patch_size, self.background, tile=tile,
                                                            percent=self.patch_overlap)
            mask[ini_x:ini_x+rows, ini_y:ini_y+cols] = 0

        print("--------------")
        print("Training Patches:   %d"%(len(self.train_patches)))
        print("Validation Patches: %d"%(len(self.val_patches)))
        print("--------------")

        # Efective regions
        mask = 0.5*np.ones_like(self.gts)
        patches = self.train_patches
        for i in range(len(patches)):
            x = patches[i][0]
            y = patches[i][1]
            sz_x = patches[i][2]
            sz_y = patches[i][3]
            mask[x : x + sz_x, y : y + sz_y] = 0.0
        patches = self.val_patches
        for i in range(len(patches)):
            x = patches[i][0]
            y = patches[i][1]
            sz_x = patches[i][2]
            sz_y = patches[i][3]
            mask[x : x + sz_x, y : y + sz_y] = 1.0
        Image.fromarray(np.uint8(mask*255)).save(self.filepath + "/mask_tr_0_vl_1_patches_irgs.png")

    def save_patches(self, output_dir='./Patches/'):

        def save_routine(dir_, patches):

            if len(patches):
                # Remove previous files
                if os.path.exists(dir_): shutil.rmtree(dir_, ignore_errors=True)
            # Create directory
            try:
                os.makedirs(dir_, exist_ok=True)
            except FileExistsError:
                print('FileExistsError exception handled...')
            
            data_dict = {}
            only_bck_patches = 0
            for i in tqdm(range(len(patches)), ncols=50):
                y1, y2, x1, x2 = patches[i]

                bc = self.background[y1: y2, x1: x2]
                if np.sum(bc == 0) >= 0.75 * ((y2-y1) * (x2-x1)):
                    only_bck_patches += 1
                    continue                                        # Do not include only-background patches
                im = self.image     [y1: y2, x1: x2]
                gt = self.gts       [y1: y2, x1: x2]

                data_dict["img"] = im
                data_dict["lbl"] = gt
                data_dict["bck"] = bc       # '0' values mask unlabeled pixels or pixels that we do not use
                try:
                    joblib.dump(data_dict, dir_ + "/{:05d}.pkl".format(i))
                except FileNotFoundError:
                    print('FileNotFoundError exception handled...')
                    continue

                # # just to check im-gt-bc alignment
                # s_im = np.uint8(255*(im+self.norm.min_val)/(self.norm.max_val-self.norm.min_val))
                # Image.fromarray(s_im[:,:,0]).save(dir_ + '/{:05d}_im_hh.png'.format(i))
                # Image.fromarray(s_im[:,:,1]).save(dir_ + '/{:05d}_im_hv.png'.format(i))
                # Image.fromarray(np.uint8(gt*255/4)).save(dir_ + '/{:05d}_gt.png'.format(i))
                # Image.fromarray(np.uint8(bc*255  )).save(dir_ + '/{:05d}_bc.png'.format(i))
            print('Background patches: %d' %(only_bck_patches))

            
        save_routine(os.path.join(output_dir, 'Tr'), self.train_patches)
        save_routine(os.path.join(output_dir, 'Vl'), self.val_patches)
        save_routine(os.path.join(output_dir, 'Ts'), self.test_patches)
        print("-------- Patches saved --------")


# ============  DATA LOADER PATCHES  ============
transform = A.ReplayCompose(
                        [
                            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5),
                            A.HorizontalFlip(),
                            A.VerticalFlip()
                        ],
                        p=0.5
                    )

class Load_patches(data.Dataset):
    '''
    generator that returns samples in batches
    For PCs with low RAM.
    It loads the batch from the computer storage
    (Make sure to run save_patches first)
    '''
    def __init__(self, sets_folders, stage="train", data_augmentation = False):

        super(Load_patches, self).__init__()

        if stage == "train": folder = "Tr"
        elif stage == "validation": folder = "Vl"
        elif stage == "test": folder = "Ts"

        self.file_paths = []
        for i in sets_folders:
            self.file_paths.extend(glob.glob(i + '/' + folder + "/*.pkl"))
        
        self.data_augmentation = data_augmentation
        # self.stage = stage
        # self.len = np.ceil(len(self.file_paths) / self.batch_size).astype(int)
        
        print("{} set size: {:0d}".format(stage, self.__len__()))
        
    def get_batch(self, batch_size=1, shuffle = False):
        '''not used'''
        files = self.file_paths.copy()
        if shuffle: np.random.shuffle(files)
        files = np.array_split(files, np.arange(batch_size, len(files), batch_size))
        
        for batch in files:
            images, gts, bckg = [], [], []

            for i in range(len(batch)):                
                data = joblib.load(batch[i])
                im = data["img"]
                gt = data["lbl"]
                bc = data["bck"]

                # Random data augmentation
                if self.data_augmentation:
                    transformed = transform(image=im, masks=[gt, bc])
                    im = transformed['image']
                    gt = transformed['masks'][0]
                    bc = transformed['masks'][1]

                # if (gt == 0).any() and (gt == 1).any() and (gt == 2).any():
                #     Image.fromarray(np.uint8(gt*255/2)).save('gts.png')
                #     Image.fromarray(np.uint8(bc*255)).save('bckg.png')
                #     Image.fromarray(np.uint8((im[:,:,0]+1)*127.5)).save('hh.png')
                #     Image.fromarray(np.uint8((im[:,:,1]+1)*127.5)).save('hv.png')

                #     plot_hist(im[:,:,0][gt == 0], 10000, [-1,1], 'hh_hist_water', './data_info')
                #     plot_hist(im[:,:,0][gt == 1], 10000, [-1,1], 'hh_hist_young', './data_info')
                #     plot_hist(im[:,:,0][gt == 2], 10000, [-1,1], 'hh_hist_mult', './data_info')
                #     plot_hist(im[:,:,1][gt == 0], 10000, [-1,1], 'hv_hist_water', './data_info')
                #     plot_hist(im[:,:,1][gt == 1], 10000, [-1,1], 'hv_hist_young', './data_info')
                #     plot_hist(im[:,:,1][gt == 2], 10000, [-1,1], 'hv_hist_mult', './data_info')
                #     exit(0)
                
                images.append(im)                
                gts.append(gt)
                bckg.append(bc)
            
            yield np.asarray(images), np.asarray(gts), np.asarray(bckg)

    def __getitem__(self, index):
        
        data = joblib.load(self.file_paths[index])
        im = data["img"]
        gt = data["lbl"]
        bc = data["bck"]

        # Random data augmentation
        if self.data_augmentation:
            transformed = transform(image=im, masks=[gt, bc])
            im = transformed['image']
            gt = transformed['masks'][0]
            bc = transformed['masks'][1]

        return im, gt, bc

    def __len__(self):
        return len(self.file_paths)

class Load_patches_on_the_fly(data.Dataset):
    '''
    generator that returns samples in batches
    Faster, but memory consuming.
    '''
    def __init__(self, data, patches, stage="train", data_augmentation = False):

        super(Load_patches_on_the_fly, self).__init__()
        
        self.data = data
        self.patches = patches
        self.stage = stage
        self.data_augmentation = data_augmentation
        
        print("{} set size: {:0d}".format(stage, self.__len__()))

    def get_batch(self, batch_size=1, shuffle = False):
        '''Not used'''
        samples = self.patches.copy()
        if shuffle:
            np.random.shuffle(samples)
        samples = np.array_split(samples, np.arange(batch_size, len(samples), batch_size))

        for i in range(len(samples)):
            images, gts, bckg = [], [], []
            for sc, x, y, sz_x, sz_y in samples[i]:
                scene = self.data[sc]
                im = scene.image     [x : x + sz_x, y : y + sz_y]
                gt = scene.gts       [x : x + sz_x, y : y + sz_y]
                bc = scene.background[x : x + sz_x, y : y + sz_y]

                # Random data augmentation
                if self.data_augmentation:
                    transformed = transform(image=im, masks=[gt, bc])
                    im = transformed['image']
                    gt = transformed['masks'][0]
                    bc = transformed['masks'][1]
                
                images.append(im)
                gts.append(gt)
                bckg.append(bc)
            
            yield np.asarray(images), np.asarray(gts), np.asarray(bckg)
    
    def __getitem__(self, index):
        sc, x, y, sz_x, sz_y = self.patches[index]
        scene = self.data[sc]
        im = scene.image     [x : x + sz_x, y : y + sz_y]
        gt = scene.gts       [x : x + sz_x, y : y + sz_y]
        bc = scene.background[x : x + sz_x, y : y + sz_y]

        # Random data augmentation
        if self.data_augmentation:
            transformed = transform(image=im, masks=[gt, bc])
            im = transformed['image']
            gt = transformed['masks'][0]
            bc = transformed['masks'][1]
        
        return im, gt, bc   
    
    def __len__(self):
        return len(self.patches)

def Calculate_norm_params(args):
    # Predefined validation tiles
    validation_tiles = pd.read_csv(args.data_info_dir  + '/validation_tiles.csv')
    validation_tiles = dict(zip(list(validation_tiles.columns),\
                                validation_tiles.values.transpose()))

    # Calculate Normalization parameters
    norm_params = None
    for i in args.train_path:
        train_data =  RadarSAT2_Dataset(args, validation_tiles, name = i)
        if norm_params is None:
            norm_params = train_data.norm
        else:
            # norm_params.clips[0] = np.minimum(norm_params.clips[0], train_data.norm.clips[0])
            # norm_params.clips[1] = np.maximum(norm_params.clips[1], train_data.norm.clips[1])
            norm_params.min_val  = np.minimum(norm_params.min_val , train_data.norm.min_val)
            norm_params.max_val  = np.maximum(norm_params.max_val , train_data.norm.max_val)

    # joblib.dump(norm_params, args.ckpt_path + '/norm_params.pkl')

    return norm_params

def Data_proc(args, set_='train', sliding_window=True, norm_params=None, aug=True, padding=True):

    # Predefined validation tiles
    validation_tiles = pd.read_csv(args.data_info_dir  + '/validation_tiles.csv')
    validation_tiles = dict(zip(list(validation_tiles.columns),\
                                validation_tiles.values.transpose()))
    sets_folders = []
    
    paths = args.train_path if set_ == 'train' else args.test_path
    for i in paths:
        # scene_data_dir = os.path.join(args.data_info_dir, args.model_name, str(i), 'Patches')
        scene_data_dir = os.path.join(args.data_info_dir, str(i), 'Patches')
        # SAVE PATCHES ON SCRATCH FOLDER 
        if os.path.exists('/home/' + os.getenv('LOGNAME') + '/scratch/'):
            scene_data_dir = os.path.join('/home/' + os.getenv('LOGNAME') + '/scratch/', os.getenv('LOGNAME'), scene_data_dir[3:])
        sets_folders.append(scene_data_dir)

        # SAVE PATCHES
        if args.save_samples:
            data =  RadarSAT2_Dataset(args, validation_tiles, name = i, set_=set_)
            if set_ == 'train' and sliding_window: data.define_sets()
            else:
                data.test_patches = Slide_patches_index(data.image.shape[0], data.image.shape[1], 
                                                        args.patch_size, 0).patches_list
            
            if norm_params is not None:
                data.image = norm_params.Normalize(data.image)

            data.save_patches(output_dir=scene_data_dir)
    
    dataset = aux_obj()
    if set_ == 'train':
        dataset.train = Load_patches(sets_folders, stage="train", data_augmentation=aug)
        dataset.val   = Load_patches(sets_folders, stage="validation", data_augmentation=False)
    else:
        dataset.test  = Load_patches(sets_folders, stage="test", data_augmentation=False)

    return dataset.train, dataset.val, dataset.test
    

# ============  DATA LOADER TOKENS  ============
from statistics import mode
from magic_irgs import IRGS

def tokens_parallel(cnn, norm_params, sequence_dir, irgs_classes, 
                  irgs_iter, token_option, max_length, item):

    sample, i = item
    image, gts, bckg = sample

    print(sequence_dir + "/{:05d}".format(i))

    # Image.fromarray(np.uint8(255*bckg[0, :, :])).save(sequence_dir + "/{:05d}_bckg.png".format(i))
    # Image.fromarray(np.uint8(image[0, :, :, 0])).save(sequence_dir + "/{:05d}_HH.png".format(i))
    # Image.fromarray(np.uint8(image[0, :, :, 1])).save(sequence_dir + "/{:05d}_HV.png".format(i))

    # ============== IRGS using HV band
    img = np.uint8(image[0, :, :, 1])
    mask = np.uint8(bckg[0]*255)
    irgs_output, boundaries = IRGS(img, irgs_classes, irgs_iter, mask=None)
    irgs_output[mask == 0] = -1
    boundaries[mask == 0] = -1

    # Image.fromarray(irgs_output).save(sequence_dir + "/{:05d}_irgs_output.png".format(i))
    # irgs_output_colored = np.uint8(255*cm.jet((irgs_output+1)/irgs_classes))[:,:,:3]
    # irgs_output_colored[irgs_output==-1] = 0
    # Image.fromarray(irgs_output_colored).save(sequence_dir + "/{:05d}_irgs_output_colored.png".format(i))

    # ============== CNN features
    img = norm_params.Normalize(image[0].detach().cpu().numpy())
    img = torch.from_numpy(img[np.newaxis,...])
    img = torch.permute(img, (0, 3, 1, 2)).float()
    with torch.no_grad():
        _, features = cnn(img)
        features = torch.permute(features, (0, 2, 3, 1))
        features = features[0,:,:,:].detach().cpu().numpy()

    # ============== Extract token sequence
    # Pool
    tokens, super_labels = [], []
    gts = gts.detach().cpu().numpy()
    if token_option == 'superpixels':
        # components, num = measure.label(boundaries, background=-1, return_num=True, connectivity=2)
        components, num = measure.label(irgs_output, background=-1, return_num=True, connectivity=2)
        for j in range(1, num+1):
            pos = components == j
            tokens.append(features[pos].mean(0))        # Average feature on each superpixel
            super_labels.append(mode(gts[0][pos]))      # Semantic label most repeated into the superpixels    
        
    elif token_option == 'clusters':
        id = np.unique(irgs_output)
        if id[0] == -1: id = id[1:]         # last class refers to landmask and boundaries
        for j in range(len(id)):
            pos = irgs_output == id[j]
            tokens.append(features[pos].mean(0))        # Average feature on each IRGS class
            super_labels.append(mode(gts[0][pos]))      # Semantic label most repeated into IRGS class
    
    tokens = np.asarray(tokens)
    super_labels = np.asarray(super_labels)

    # Control sequence length
    n_subsequences = np.ceil(len(tokens) / max_length).astype(int)

    tokens_ids = np.arange(len(tokens))
    np.random.shuffle(tokens_ids)
    tokens = tokens[tokens_ids]
    super_labels = super_labels[tokens_ids]
    for j in range(n_subsequences):
        tk = tokens      [j*max_length: j*max_length+max_length]
        sl = super_labels[j*max_length: j*max_length+max_length]
        # Add padding in the sequences so that they can have the same length. 
        # Save pad value to create attention mask afterwards
        pad = max_length - len(tk)
        # assert pad >= 0, 'padding should be non-negative ---- len(tokens) = %d'%(len(tokens))
        
        data_dict = {}
        data_dict["tokens"] = np.pad(tk, ((0, pad), (0, 0)))
        data_dict["labels"] = np.pad(sl,  (0, pad))
        data_dict["pad"]    = pad
        joblib.dump(data_dict, sequence_dir + "/{:05d}{:d}.pkl".format(i, j))

def Extract_token_sequence(cnn, norm_params, data_loader, output_folder, stage, args):
    
    cnn.cpu()                   # More RAM available for bigger input patches
    cnn.eval()
    
    # # Remove previous files
    sequence_dir = os.path.join(output_folder, stage)
    if os.path.exists(sequence_dir): shutil.rmtree(sequence_dir)
    # Create directory
    os.makedirs(sequence_dir, exist_ok=True)

    data_loader_size = len(data_loader)
    data_loader = iter(data_loader)

    iterable = zip(data_loader, range(data_loader_size))

    # test line
    # tokens_parallel(cnn, norm_params, sequence_dir, args.irgs_classes, 
    #               args.irgs_iter, args.token_option, args.max_length, next(iterable))
    # tokens_parallel(cnn, norm_params, sequence_dir, args.irgs_classes, 
    #               args.irgs_iter, args.token_option, args.max_length, next(iterable))

    n_cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_cores)
    func = partial(tokens_parallel, cnn, norm_params, sequence_dir, args.irgs_classes, 
                   args.irgs_iter, args.token_option, args.max_length)
    p.map(func, iterable)
    p.close()
    p.join() 

class Load_token_sequence(data.Dataset):
    
    def __init__(self, folder, stage='Tr'):

        super(Load_token_sequence, self).__init__()
        self.file_paths = glob.glob(folder + stage + "/*.pkl")
        print("{} set size: {:0d}".format(stage, self.__len__()))
        
    def __getitem__(self, index):
        
        data = joblib.load(self.file_paths[index])
        im = data["tokens"]
        gt = data["labels"]
        pd = data["pad"]

        return im, gt, pd

    def __len__(self):
        return len(self.file_paths)

# ============  DATA LOADER IMAGES + SEGMENTS  ============
from magic_irgs import IRGS

def irgs_segments_parallel(irgs_classes, irgs_iter, token_option, norm_params, use_landmask, item):
    sample, file, i = item
    image, gts, bckg = sample

    # ============== IRGS
    img = np.uint8(image[:, :, 1])        # using HV band
    # img = np.uint8(image)
    mask = np.uint8(bckg*255)
    irgs_output, boundaries = IRGS(img, irgs_classes, irgs_iter, mask=None)
    if use_landmask:
        irgs_output[mask == 0] = -1
        boundaries[mask == 0] = -1

    # ==============  clusters/superpixels ids
    id = np.unique(irgs_output)
    if token_option == 'superpixels':
        # components, n_tokens = measure.label(boundaries, background=-1, return_num=True, connectivity=2)
        components, n_tokens = measure.label(irgs_output, background=-1, return_num=True, connectivity=2)
        segments = components - 1
        print('Sample %d ------IRGS superpixels------: %d'%(i, n_tokens))
        
    elif token_option == 'clusters':
        n_tokens = len(id)
        segments = irgs_output.copy()
        if id[0] == -1: n_tokens -= 1
        print('Sample %d ------IRGS clusters------: %d'%(i, n_tokens))
    
    # when the tokens are calculated on the fly (end-to-end approach)
    # the ids are changed so that they're exclusive among the sample images
    # within a batch. In that sense, -1 ids need to be changed first to infinite 
    # (the reason for this is restricted to the way the code was implemented)
    segments = segments.astype('float')
    segments[segments==-1] = np.inf

    if file != 0:
        # save samples
        image = norm_params.Normalize(image)
        data_dict = {}
        data_dict["img"] = image
        data_dict["lbl"] = gts
        data_dict["bck"] = bckg
        data_dict["seg"] = segments
        data_dict["n_t"] = n_tokens
        data_dict["bound"] = boundaries
        try:
            joblib.dump(data_dict, file)
            # # just to check im-gt-bc-seg-boundaries alignment
            # Image.fromarray(np.uint8(norm_params.Denormalize(image))[:,:,0]).save(os.path.split(file)[0] + '/hh.png')
            # Image.fromarray(np.uint8(norm_params.Denormalize(image)[:,:,1])).save(os.path.split(file)[0] + '/hv.png')
            # Image.fromarray(np.uint8(gts*255/gts.max())).save(os.path.split(file)[0] + '/gts.png')
            # Image.fromarray(np.uint8(bckg*255)).save(os.path.split(file)[0] + '/bckg.png')
            # segments[segments==np.inf] = -1
            # Image.fromarray(np.uint8(segments*255/(n_tokens+1))).save(os.path.split(file)[0] + '/segments.png')
            # Image.fromarray(255*np.uint8(boundaries==-1)).save(os.path.split(file)[0] + '/boundaries.png')
            # exit()
        except FileNotFoundError:
            print('FileNotFoundError exception handled...')
    
    else: 
        return segments, n_tokens, boundaries

def Extract_segments(samples, norm_params, args, use_landmask=True):

    samples_size = len(samples)
    samples_iter = iter(samples)
    iterable = zip(samples_iter, samples.file_paths, range(samples_size))

    if samples_size > 1:
        Parallel(irgs_segments_parallel, iterable, args.irgs_classes, 
                args.irgs_iter, args.token_option, norm_params, use_landmask)
    else:
        # # test line
        irgs_segments_parallel(args.irgs_classes, args.irgs_iter, 
                            args.token_option, norm_params, use_landmask, next(iterable))

class Load_patches_segments(data.Dataset):
    def __init__(self, file_paths, aug=False):
        super(Load_patches_segments, self).__init__()
        self.file_paths = file_paths
        self.data_augmentation = aug

    def __getitem__(self, index):
        
        data = joblib.load(self.file_paths[index])
        img = data["img"]
        lbl = data["lbl"]
        bck = data["bck"]

        seg = data["seg"]
        n_t = data["n_t"]
        bound = data["bound"]

        # Random data augmentation
        if self.data_augmentation:
            transformed = transform(image=img, masks=[lbl, bck, seg, bound])
            
            if transformed['replay']['applied']:
                img = transformed['image']
                lbl = transformed['masks'][0]
                bck = transformed['masks'][1]
                seg = transformed['masks'][2]
                bound = transformed['masks'][3]

                if transformed['replay']['transforms'][0]['applied']:       # 'ShiftScaleRotate'
                    # map seg from 0 to number of segments and update n_t
                    seg = np.where(seg == np.inf, seg, np.searchsorted(np.unique(seg[seg != np.inf]), seg))
                    n_t = len(np.unique(seg))
                    if (seg==np.inf).any(): n_t -= 1


        # img_ = (img+img.min((0,1)))/(img.max((0,1))-img.min((0,1)))
        # Image.fromarray(np.uint8(255*img_)[:,:,0]).save('hht.png')
        # Image.fromarray(np.uint8(255*img_)[:,:,1]).save('hvt.png')
        # Image.fromarray(np.uint8(lbl*255/lbl.max())).save('gtst.png')
        # Image.fromarray(np.uint8(bck*255)).save('bckgt.png')
        # seg[seg==np.inf] = -1
        # Image.fromarray(np.uint8((seg+1)*255/(seg.max()+1))).save('segmentst.png')
        # Image.fromarray(255*np.uint8(bound==-1)).save('boundariest.png')
        # ic(n_t)
        # exit()

        return img, lbl, bck, seg, n_t, bound

    def __len__(self):
        return len(self.file_paths)