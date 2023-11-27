import argparse
import os
from utils.utils import boolean_string
# from operator import truediv

def Arguments_train():
    parser = argparse.ArgumentParser()
    # ================ TUNNED HYPER-PARAMETERS ==========
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.00013, help='learning rate')
    parser.add_argument('--alpha', type=float, default=2.62, help='weight associated to Transformer Loss in end_to_end approach')
    # ===================================================


    # ================ MULTIPLE GPUs SETUP ==========
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--num_workers', default=1, type=int, help='')    
    parser.add_argument('--nodes', default=1, type=int, help='')    
    # ===================================================


    # ================ IRGS-TRANS CONFIG ==========
    parser.add_argument('--mode', type=str, default='end_to_end', choices=['end_to_end', 'multi_stage'], help='.......')
    parser.add_argument('--stage', type=str, default='cnn', choices=['cnn', 'transformer', 'end_to_end'], help='specifiy STAGE on multi-stage mode')
    parser.add_argument('--loss_term', type=str, default='end_to_end', choices=['end_to_end', 'transformer'], help='Specify loss terms for end_to_end approach')
    parser.add_argument('--max_length', type=int, default=400, help='Maximum sequence length')
    parser.add_argument('--mix_images', type=boolean_string, default=True, help='Mix tokens from different images in the batch')
    parser.add_argument('--random_tokens', type=boolean_string, default=True, help='random/oredered tokens')

    # IRGS
    parser.add_argument('--irgs_classes', type=int, default=50, help='Number of classes considered on IRGS')
    parser.add_argument('--irgs_iter', type=int, default=120, help='Number of iterations on IRGS')
    parser.add_argument('--token_option', type=str, default='superpixels', choices=['superpixels', 'clusters'], help='.......')

    # TRANSFORMER
    parser.add_argument('--token_size', type=int, default=16, help='sub-image/word size (square) for transformer input')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads on self-attention module')
    parser.add_argument('--trans_depth', type=int, default=8, help='Number of transformer blocks')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embeding depth')
    # ===================================================


    # ================ DATA/RESULTS CONFIG ==========
    parser.add_argument('--save_samples', type=boolean_string, default=False, help='If True, the samples will be saved in the HD to be loaded during training (It requires less RAM) \
                                                                          If False, the samples are extracted directly from the scene (faster)')
    parser.add_argument('--Datasets_dir', type=str, default='../../Dataset/', help='datasets path')
    parser.add_argument('--Dataset_name', type=str, default='21-scenes-less_resolution', help='dataset name')
    parser.add_argument('--train_path', type=str, default='20100816', help='list of scenes separates by "_"')
    parser.add_argument('--test_path', type=str, default='20100721', help='list of scenes separates by "_"')
    parser.add_argument('--in_chans', type=int, default=2, help='Number of bands')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')

    parser.add_argument('--patch_size', type=int, default=224, help='input image size (square)')
    parser.add_argument('--patch_overlap', type=float, default=0.05, help='Overlap between patches')
    
    parser.add_argument('--data_info_dir', type=str, default='../data_info/')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/')
    parser.add_argument('--model_name', type=str, default='model')
    # ===================================================

    parser.add_argument('--sweep', type=boolean_string, default=False, help='hyperparameter tunning mode')
    parser.add_argument('--epochs', type=int, default=50, help='epoch number')
    parser.add_argument('--samples_per_epoch', type=int, default=1000, help='number of samples for training each epoch')
    parser.add_argument('--patience', type=int, default=15, help='number of epochs after no improvements (stop criteria)')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--CNN', type=str, default='resnet18', help='CNN backbone')

    args = parser.parse_args()

    args.Datasets_dir = os.path.join(args.Datasets_dir, args.Dataset_name)
    args.train_path = args.train_path.split('_')
    args.test_path = args.test_path.split('_')
    args.data_info_dir = os.path.join(args.data_info_dir, args.Dataset_name)

    return args

def Arguments_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=boolean_string, default=True, help='bool flag')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='')    


    # ================ IRGS-TRANS CONFIG ==========
    parser.add_argument('--mode', type=str, default='end_to_end', choices=['end_to_end', 'multi_stage'], help='.......')
    parser.add_argument('--stage', type=str, default='cnn', choices=['cnn', 'transformer', 'end_to_end'], help='stage for multi-stage approach')
    parser.add_argument('--loss_term', type=str, default='transformer', choices=['cnn', 'end_to_end', 'transformer'], help='Specify loss terms for end_to_end approach')
    parser.add_argument('--max_length', type=int, default=400, help='Maximum sequence length')
    parser.add_argument('--mix_images', type=boolean_string, default=False, choices=['False'], help='Mix tokens from different images in the batch')
    parser.add_argument('--random_tokens', type=boolean_string, default=False, choices=['False'], help='random/oredered tokens')

    # IRGS
    parser.add_argument('--irgs_classes', type=int, default=15, help='Number of classes considered on IRGS')
    parser.add_argument('--irgs_iter', type=int, default=120, help='Number of iterations on IRGS')
    parser.add_argument('--token_option', type=str, default='superpixels', choices=['superpixels', 'clusters'], help='.......')

    # TRANSFORMER
    parser.add_argument('--token_size', type=int, default=16, help='sub-image/word size (square) for transformer input')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads on self-attention module')
    parser.add_argument('--trans_depth', type=int, default=8, help='Number of transformer blocks')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embeding depth')
    # ===================================================


    # ================ DATA/RESULTS CONFIG ==========
    parser.add_argument('--save_samples', type=boolean_string, default=False, help='If True, the samples will be saved in the HD to be loaded during training (It requires less RAM) \
                                                                          If False, the samples are extracted directly from the scene (faster)')
    parser.add_argument('--Datasets_dir', type=str, default='../../Dataset/', help='datasets path')
    parser.add_argument('--Dataset_name', type=str, default='21-scenes-less_resolution', help='dataset name')
    parser.add_argument('--test_path', type=str, default='20110725', help='list of scenes separates by "_"')
    parser.add_argument('--in_chans', type=int, default=2, help='Number of classes')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')

    parser.add_argument('--patch_size', type=int, default=5000, help='input image size (square)')
    parser.add_argument('--patch_overlap', type=float, default=0.1, help='Overlap between patches')

    parser.add_argument('--save_path', type=str, default='../results_test/', help='path to save inference segmentation')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/')
    parser.add_argument('--model_name', type=str, default='model')
    # ===================================================

    
    parser.add_argument('--CNN', type=str, default='UNet', help='CNN backbone')

    args = parser.parse_args()
    args.Datasets_dir += '/' + args.Dataset_name + '/'
    args.test_path = args.test_path.split('_')

    return args