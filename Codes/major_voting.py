from icecream import ic
import argparse
from PIL import Image
import numpy as np
from statistics import mode
import os
from skimage import measure

from utils.utils import rgb2gray


# 8-connected pixel connectivity
Neighborhood = np.array([
                        [-1, -1, -1,  0,  0,  1,  1,  1],
                        [-1,  0,  1, -1,  1, -1,  0,  1]
                        ])


def ConComp_binary(mat):

    def BFS():
        while len(queue):
            x, y = queue.pop(0)

            for k in range(8):
                posx = x + Neighborhood[0,k]
                posy = y + Neighborhood[1,k]
                if posx < 0 or posx >= r or \
                   posy < 0 or posy >= c: continue
                if mat[posx, posy] and not comp[posx, posy]:
                    queue.append((posx, posy))
                    comp[posx, posy] = cont

    r, c = mat.shape
    comp = np.zeros((r,c))
    cont = 0

    ij = np.where(mat==1)
    for i, j in zip(ij[0], ij[1]):
        if not comp[i, j]:
            cont += 1
            comp[i, j] = cont
            queue = [(i,j)]
            BFS()
    
    return comp, cont

from numba import jit
import numba
class MV_per_CC(object):
    def __init__ (self):
        print('-- Major voting per superpixel --')

    def do(self, superpixels, n_tokens, labels, background=0):
        print('Total superpixels: ', n_tokens)
        new_labels = -1*np.ones_like(labels)

        for j in range(n_tokens):
            pos = superpixels == j
            lbl = mode(labels[pos])
            new_labels[pos] = lbl
        return new_labels

    def CPU_parallel(self, superpixels, n_tokens, labels, background=0):
        print('CPU parallel...')
        print('Total superpixels: ', n_tokens)

        @jit(nopython=True, parallel=True)
        def MV_per_CC_CPU_parallel():
            n_values = labels.max() + 1
            new_labels = -1*np.ones_like(labels)

            for j in numba.prange(n_tokens):
                counter = np.zeros((n_values))
                aux = -1*np.ones((n_values))

                pos = np.where(superpixels == j)
                for k, l, m in zip(pos[0], pos[1], range(len(pos[0]))):
                    counter[labels[k,l]] += 1
                    if aux[labels[k,l]] == -1: aux[labels[k,l]] = m

                lb_ = np.argsort(counter)
                lb = lb_[-1]
                # Keep the first mode seen (for cases with more than 1 mode)
                for k in lb_[:-1]:
                    if counter[k] == counter[lb] and aux[k] < aux[lb]:
                        lb = k 

                for k, l in zip(pos[0], pos[1]):
                    new_labels[k, l] = lb
            return new_labels
        return MV_per_CC_CPU_parallel()



def MV_per_CC_(superpixels, labels, background=0):

    print('-- Major voting per superpixel --')

    r, c = labels.shape

    components, num = measure.label(superpixels, background=background, return_num=True, connectivity=2)
    print('Total superpixels: ', num)
    new_labels = -1*np.ones_like(labels)

    for j in range(1, num+1):
        pos = components == j
        lbl = mode(labels[pos])
        new_labels[pos] = lbl

    '''
    # ONE HOT ENCODING
    print('-------------- hot encoding --------------')
    one_hot = np.zeros((r, c, len(id)))
    for i in range(len(id)):
        pos = np.where(superpixels==id[i])
        one_hot[pos[0], pos[1], i] = 1

    # MAJOR VOTING PER CONNECTED COMPONENT
    print('-- Major voting per connected component --')
    new_labels = np.zeros_like(labels)
    total_superpixels = 0
    for i in range(len(id)):
        # components, num = ConComp_binary(one_hot[:,:,i])
        components, num = measure.label(one_hot[:,:,i], return_num=True, connectivity=2)
        print('superpixel: ', id[i], num)
        for j in range(1, num+1):
            pos = components == j
            lbl = mode(labels[pos])
            new_labels[pos] = lbl
        total_superpixels += num
    
    print('Total superpixels: ', total_superpixels)
    '''

    return new_labels

def MV_per_Class(classes, labels):
    
    print('-- Major voting per class --')

    class_id = np.unique(classes)
    if class_id[0] == -1: class_id = class_id[1:]

    new_labels = -1*np.ones_like(labels)
    for j in class_id:
        pos = superpixels == j
        lbl = mode(labels[pos])              # Semantic label most repeated into IRGS class
        new_labels[pos] = lbl
    
    return new_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Datasets_dir', type=str, default='../../Dataset/', help='datasets path')
    parser.add_argument('--Dataset_name', type=str, default='21-scenes-less_resolution', help='dataset name')
    parser.add_argument('--Sem_Seg_dir', type=str, default='UNet', help='Semantic Segmentation approach')
    parser.add_argument('--scene', type=str, default='20100721', help='')
    args = parser.parse_args()

    args.Datasets_dir += '/' + args.Dataset_name + '/'
    Superpixels_file = './results_test/' + args.Dataset_name + '/IRGS/' + \
                       args.scene + '/irgs_output.npy'
    ic(Superpixels_file)
    # exit()
    # Prediction - Unet
    Sem_Seg_file = './results_test/'  + args.Dataset_name + '/' + \
                   args.Sem_Seg_dir + '/' + \
                   args.scene + '/' + \
                   'colored_predict_' + args.scene + '.png'
    output_folder = os.path.dirname(Sem_Seg_file) + '/M_voting_IRGSS_Unet/'

    # # Labels
    # Sem_Seg_file = './results_test/'  + args.Dataset_name + '/' + \
    #                args.Sem_Seg_dir + '/' + \
    #                args.scene + '/' + \
    #                'colored_gts_' + args.scene + '.png'
    # output_folder = os.path.dirname(Sem_Seg_file) + '/M_voting_IRGSS_Labels/'

    os.makedirs(output_folder, exist_ok=True)
    print('Scene:', args.scene)
    
    landmask = np.array(Image.open(args.Datasets_dir + args.scene +'/landmask.bmp'))

    superpixels = np.load(Superpixels_file)
    superpixels[landmask==0] = -1

    lbl = rgb2gray(np.array(Image.open(Sem_Seg_file))/255)
    labels = np.zeros_like(lbl)

    class_colors = np.uint8(np.array([[0, 0, 0],           # Background
                                      [255, 204, 239],     # Open water
                                      [204, 0, 255],       # Young ice
                                      [255, 0, 0]          # Multi-year ice
                                     ]))
    for i in range(len(class_colors)):
        labels[lbl==rgb2gray(class_colors[i]/255)] = i
    labels[landmask==0] = 0
    prediction = class_colors[labels.astype(int)]
    Image.fromarray(prediction).save(output_folder +  '/1_prediction_%s.png'%(args.scene))

    if False:
        superpixels = np.array([[1, 1, 1, 2, 2, 2],
                                [1, 1, 1, 2, 1, 1],
                                [1, 3, 1, 0, 0, 1],
                                [1, 3, 1, 5, 0, 1],
                                [1, 1, 1, 5, 0, 0],
                                [1, 1, 4, 1, 1, 0],
                                [1, 4, 4, 4, 1, 1]]
                               )
        labels = np.array([     [1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 1, 0],
                                [0, 0, 1, 0, 0, 1],
                                [0, 0, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 0],
                                [1, 0, 1, 1, 1, 1]]
                               )
    
    # MAJOR VOTING PER CONNECTED COMPONENT
    m_v_per_CC = MV_per_CC_(superpixels, labels, background=-1)
    major_voting_per_CC = class_colors[m_v_per_CC.astype(int)]
    Image.fromarray(major_voting_per_CC).save(output_folder +  '/4_major_voting_per_CC_%s.png'%(args.scene))

    # MAJOR VOTING PER CLASS
    m_v_per_Class = MV_per_Class(superpixels, labels)
    major_voting_per_Class = class_colors[m_v_per_Class.astype(int)]
    Image.fromarray(major_voting_per_Class).save(output_folder +  '/6_major_voting_per_Class_%s.png'%(args.scene))

    a = m_v_per_CC[landmask!=0]
    b = m_v_per_Class[landmask!=0]
    print("Similarity: %.2f %%"%(100 * np.sum(a==b)/len(a)))


    r, c = labels.shape
    print('------------- drawing edges --------------')
    edges = np.ones((r, c)).astype(np.int8)
    for x in range(r):
        for y in range(c):
            for k in range(8):
                posx = x + Neighborhood[0,k]
                posy = y + Neighborhood[1,k]
                if posx < 0 or posx >= r or posy < 0 or posy >= c: continue
                if superpixels[x,y] != superpixels[posx,posy]:
                    edges[x,y], edges[posx,posy] = 0, 0
    
    prediction_and_edges = class_colors[labels.astype(int) * edges]
    Image.fromarray(prediction_and_edges).save(output_folder +  '/2_and_edges_%s.png'%(args.scene))
    major_voting_per_CC_and_edges = major_voting_per_CC * edges[...,np.newaxis]
    Image.fromarray(np.uint8(major_voting_per_CC_and_edges)).save(output_folder +  '/3_major_voting_per_CC_and_edges_%s.png'%(args.scene))
    major_voting_per_Class_and_edges = major_voting_per_Class * edges[...,np.newaxis]
    Image.fromarray(np.uint8(major_voting_per_Class_and_edges)).save(output_folder +  '/5_major_voting_per_Class_and_edges_%s.png'%(args.scene))
