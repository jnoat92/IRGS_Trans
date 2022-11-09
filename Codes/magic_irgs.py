#%%
'''
No@
'''
from matplotlib import cm
import numpy as np
import os, argparse
from utils.dataloader import RadarSAT2_Dataset
from PIL import Image
from icecream import ic

from magic_py.magic_rag import magic_rag
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import Map_labels

def IRGS(img, n_classes, n_iter, mask=None):
    # --- RUN IRGS --- #
    rag = None
    if mask is None:
        rag = magic_rag(img, msk=None, N_class=n_classes, verbose=True)
    else:
        rag = magic_rag(img, msk=mask, N_class=n_classes, verbose=True)

    print("Initializing k-means with", n_classes, "classes")
    rag.initialize_kmeans()

    print("Performing", str(n_iter), "IRGS iterations...")
    # for j in tqdm(range(n_iter), ncols=50):
    for j in range(n_iter):
        # rag.irgs_step(beta1=beta1, current_iter=i+1)
        rag.irgs_step(current_iter=j+1)
    
    irgs_output = rag.result_image
    # irgs_output = rag.result_image_with_boundaries
    # boundaries = np.int16(rag.bmp == -2) # Not consistent with rag.result_image_with_boundaries
    boundaries = np.int16(rag.result_image_with_boundaries != -2)

    boundaries[boundaries == 0] = -1
    boundaries[irgs_output < 0] = -1
    irgs_output[irgs_output < 0] = -1           # background and boundaries.
                                                # IRGS returns an aditional class with 
                                                # label -2 for landmask and boundaries\
    irgs_output = Map_labels(irgs_output)
    
    return irgs_output, boundaries


def Hier_IRGS(img, n_classes, n_iter, mask=None):
    
    def recurs(l, mask):
        irgs_out, bound = IRGS(img, n_classes[l], n_iter, mask=mask)
        irgs_out += 1
        if l == len(n_classes)-1: return irgs_out, bound

        id = np.unique(irgs_out)
        if id[0] == 0: id = id[1:]
        for i in id:
            pos = irgs_out == i
            
            new_mask = 255 * np.uint8(pos)
            new_irgs_out, new_bound = recurs(l+1, new_mask)
            
            irgs_out[pos] = irgs_out[pos] * 10**(len(n_classes)-l-1) + new_irgs_out[pos]
            bound[pos] = new_bound[pos]
        
        return irgs_out, bound

    irgs_output, boundaries = recurs(0, mask)
    # map clusters labels to [0, 1, 2,..., n_clusters-1] 
    # | n_clusters <= irgs_classes
    id = np.unique(irgs_output)
    aux_id = -1 * np.ones_like(irgs_output)     
    c = 0
    for i in id:                                
        if i != 0:
            aux_id[irgs_output == i] = c
            c += 1
    irgs_output = aux_id

    return irgs_output, boundaries


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Datasets_dir', type=str, default='../../Dataset/', help='datasets path')
    parser.add_argument('--Dataset_name', type=str, default='21-scenes-less_resolution', help='dataset name')
    parser.add_argument('--scenes', type=str, default='20100712', help='list of scenes separates by "_"')

    parser.add_argument('--n_classes', type=int, default=100, help='Number of classes')
    parser.add_argument('--n_iter', type=int, default=120, help='Number of iterations')
    parser.add_argument('--data_info_dir', type=str, default='./data_info/')
    parser.add_argument('--save_path', type=str, default='./results_test/', help='path to save inference segmentation')

    args = parser.parse_args()

    args.Datasets_dir += '/' + args.Dataset_name + '/'
    args.scenes = args.scenes.split('_')
    args.data_info_dir += '/' + args.Dataset_name + '/'

    for i in args.scenes:

        data =  RadarSAT2_Dataset(args, name = i, phase="test")
        img = np.uint8(data.image[:,:,1])
        mask = np.uint8(data.background*255)

        # irgs_output, boundaries = Hier_IRGS(img, [2, 2, 2, 2], args.n_iter, mask=mask)
        irgs_output, boundaries = IRGS(img, args.n_classes, args.n_iter, mask=mask)

        output_folder = os.path.join(args.save_path, args.Dataset_name, 'IRGS', i)
        os.makedirs(output_folder, exist_ok=True)
        np.save(output_folder + '/irgs_output.npy', irgs_output)

        irgs_output_colored = np.uint8(255*cm.jet(irgs_output/args.n_classes))[:,:,:3]
        plt.imshow(irgs_output_colored)
        plt.show()
        irgs_output_colored[irgs_output==-1] = 0
        Image.fromarray(irgs_output_colored).save(output_folder + '/irgs_output_colored.png')

        boundaries[irgs_output==-1] = 0
        Image.fromarray(np.uint8(255*boundaries)).save(output_folder + '/boundaries.png')
