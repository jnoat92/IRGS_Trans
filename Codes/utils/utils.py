import torch
import numpy as np
from icecream import ic
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from skimage.measure import find_contours
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score, f1_score, accuracy_score
import csv
import time
import wandb
def Metrics(y_true, y_pred, method, output_folder, time_=None):
    
    acc = 100 * accuracy_score(y_true, y_pred)
    pre = 100 * precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = 100 * recall_score(y_true, y_pred, average=None, zero_division=0)        # --> Same as accuracy per class
    f1s = 100 * f1_score(y_true, y_pred, average=None, zero_division=0)
    IoU = 100 * jaccard_score(y_true, y_pred, average=None, zero_division=0)

    # CM = confusion_matrix(y_true, y_pred)                   # --> confusion matrix C is such that C[i,j] is 
    #                                                         # equal to the number of observations known to 
    #                                                         # be in group i and predicted to be in group j.
    # acc = 100 * CM.trace() / CM.sum()
    # pre = 100 * np.diag(CM)/CM.sum(axis=0)
    # rec = 100 * np.diag(CM)/CM.sum(axis=1)                      # --> Same as accuracy per class
    # f1s = 100 * 2 * pre * rec / (pre+rec)

    if output_folder is None: return acc, IoU

    # Save results
    classes = np.unique(np.concatenate((y_pred, y_true)))
    print_line = [
                    [method],
                    [method, 'OA       ', acc],
                    [method, 'Classes  '] + [str(k)+4*" " for k in classes] + ['Average'],
                    [method, 'Precision'] + list(pre) + [pre.mean()],
                    [method, 'Recall   '] + list(rec) + [rec.mean()],
                    [method, 'F-Score  '] + list(f1s) + [f1s.mean()],
                    [method, 'IoU      '] + list(IoU) + [IoU.mean()],
                 ]
    with open(output_folder + '/metrics.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for k in print_line: writer.writerow(k)

    with open(output_folder + '/metrics.txt', 'a') as f:
        if time_: 
            f.write(time.strftime('%H:%M:%S', time.gmtime(time_)) + '\n')
        for k in print_line:
            for l in k:
                if isinstance(l, float) or isinstance(l, int):
                    f.write(str(np.round(l, 2)) + '  ')
                else: f.write(l + '  ')
            f.write('\n')
        f.write('\n\n')

    wandb.summary[method.split(' ')[0] + '-OA']         = acc
    wandb.summary[method.split(' ')[0] + '-Precision']  = pre.mean()
    wandb.summary[method.split(' ')[0] + '-Recall']     = rec.mean()
    wandb.summary[method.split(' ')[0] + '-F-Score']    = f1s.mean()
    wandb.summary[method.split(' ')[0] + '-IoU']        = IoU.mean()

class aux_obj(object):
    def __init__(self, init_value=None):
        self.train, self.val, self.test = 3*[init_value]

def Parallel(function, iterable, *args):
    '''
    When the iterable is too long ~> 30000. Try to  distribute it 
    in more than one run. Otherwise it takes too long to setup the 
    parallel process.
    '''
    n_cores = multiprocessing.cpu_count() - 1
    print('Configuring CPU multiprocessing...')
    print('Number of cores: %d'%(n_cores))
    p = multiprocessing.Pool(n_cores)
    func = partial(function, *args)
    x = p.map(func, iterable)
    p.close()
    p.join()

    return x

def Map_labels(labels):
    # map labels to [0, 1, 2,..., n_labels-1]

    id = np.unique(labels)
    aux_id = -1 * np.ones_like(labels)     
    c = 0
    for i in id:                                
        if i != -1:
            aux_id[labels == i] = c
            c += 1
    return aux_id

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):])).item()

def plot_hist(img, bins, lim, name, output_path):
    hist , bins  = np.histogram(img, bins=bins)
    plt.figure(figsize=(6, 4))
    plt.plot(bins[1:], hist/np.prod(img.shape))
    plt.title("{}".format(name))
    plt.xlabel('bins')
    plt.ylabel('count')
    # plt.xlim(lim)
    # plt.ylim([0, 0.011])
    plt.tight_layout()
    plt.savefig(output_path + "/" + name + ".png", dpi=500)
    plt.close()

def subplot_grid(images, file_path, x_label_list, y_label_list, fontsize=None, tick_colors=None, title=None):
    
    if len(images.shape) < 3: 
        images = images[np.newaxis,...]

    N = images.shape[0]
    a = np.floor(np.sqrt(N)).astype('int16')
    b = np.ceil(N / a).astype('int16')

    sz = images.shape[1]*5/4
    f_sz = images.shape[1]*5/4
    fig = plt.figure()                          # figsize=(sz, sz)
    k = 0
    for i in range(a):
        for j in range(b):
            k += 1
            ax = fig.add_subplot(a, b, k)
            hdl = ax.imshow(images[k-1])
            fig.colorbar(hdl, ax=ax)
            if title:
                ax.set_title(title)
            else: ax.set_title('%d'%(k))

            ax.set_yticks(range(images.shape[1]))
            ax.set_yticklabels(y_label_list, fontsize=fontsize)
            for ytick, color in zip(ax.get_yticklabels(), tick_colors):
                ytick.set_color(color)

            ax.xaxis.tick_top()
            ax.set_xticks(range(images.shape[2]))
            ax.set_xticklabels(x_label_list, rotation=90, fontsize=fontsize)
            for xtick, color in zip(ax.get_xticklabels(), tick_colors):
                xtick.set_color(color)

    plt.tight_layout()
    # plt.show()
    plt.savefig(file_path, dpi=500)
    plt.close()

class Image_reconstruction(object):

    '''
    This class performes a slide windows for dense predictions.
    If we consider overlap between consecutive patches then we will keep the central part 
    of the patch (stride, stride)

    considering a small overlap is usually useful because dense predections tend 
    to be unaccurate at the border of the image
    '''
    def __init__ (self, inputs, model, output_c_dim, patch_size=256, overlap_percent=0):

        self.inputs = inputs
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
        self.model = model
    
    def Inference(self, tile, use_gpu=1):
        
        '''
        Normalize before calling this method
        '''

        num_rows, num_cols, _ = tile.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
        
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col), (0,0) )
        tile_pad = np.pad(tile, pad_tuple, mode = 'symmetric')
        tile_pad = torch.from_numpy(tile_pad.transpose((2, 0, 1))).float()

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((self.output_c_dim, k1*stride, k2*stride))         # keep central part of the patch
                                                                            # (not exclusive with overlap >= 50%)
        Logits_ave = np.zeros((self.output_c_dim, tile_pad.shape[1], tile_pad.shape[2]))
        count_mat = np.zeros((tile_pad.shape[1], tile_pad.shape[2]))

        if use_gpu: tile_pad = tile_pad.cuda()

        for i in tqdm(range(k1), ncols=50):
            for j in range(k2):
                
                patch = tile_pad[:, i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size)]
                patch = patch.unsqueeze(0)

                logits, _ = self.model(patch)                                                               # shape (batch, channels, H, W)
                Logits_ave[:,i*stride:(i*stride + self.patch_size), 
                             j*stride:(j*stride + self.patch_size)] += logits[0].detach().cpu().numpy()     # shape (channels, H, W)
                count_mat[i*stride:(i*stride + self.patch_size), 
                          j*stride:(j*stride + self.patch_size)] += 1

                probs_ = logits.softmax(1)
                probs[:, i*stride : i*stride+stride, 
                         j*stride : j*stride+stride] = probs_[0, :, overlap//2 : overlap//2 + stride, 
                                                                    overlap//2 : overlap//2 + stride].detach().cpu().numpy()

        Logits_ave /= count_mat
        probs_from_logits_ave = torch.from_numpy(Logits_ave).softmax(0).detach().cpu().numpy()      # this approach removes the blocking effect

        # Taken off the padding
        probs = probs[:, :k1*stride-step_row, :k2*stride-step_col]
        probs_from_logits_ave = probs_from_logits_ave[:, overlap//2: -(overlap//2 + step_row), 
                                                         overlap//2: -(overlap//2 + step_col)]

        return probs, probs_from_logits_ave

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return np.asarray([int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)])

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def time_format(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   return "%02d:%02d:%02d" % (hour, min, sec) 

def get_contours(lbl):
    for lvl in np.unique(lbl):
        level_ctrs = find_contours(lbl, level=lvl)
        for c in level_ctrs:
            try:
                contours = np.concatenate((contours, c), axis=0)
            except:
                contours = c
    return np.uint16(contours)
