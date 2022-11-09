"""
No@
"""
import os
from icecream import ic
import numpy as np
import wandb

import argparse
def Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')    
    parser.add_argument('--nodes', default=1, type=int, help='')    
    args = parser.parse_args()
    return args
args = Arguments()
multi_gpu_config = [' --{} {} '.format(i, j) for i, j in zip(args.__dict__.keys(), args.__dict__.values())]
multi_gpu_config = ''.join(multi_gpu_config)
    

Schedule = []
Schedule = ["wandb login e478171c2941cc8ddc5a71663e36f613042dfc6e"]
Schedule = ["wandb offline"]


# 21 scenes experiments
scenes = ['20100418', '20100426', '20100510', '20100524', '20100605', '20100623', '20100629', 
          '20100712', '20100721', '20100730', '20100807', '20100816', '20100907', '20100909', 
          '20101003', '20101021', '20101027', '20101114', '20101120', '20101206', '20101214']

new_test_scenes = ["20100704", "20101014", "20101017", "20101025", "20110530", 
                   "20110613", "20110627", "20110709", "20110710", 
                   "20110720", "20110725", "20111006", "20111013", "20111029"]


# IRGS Exp
# Schedule.append("python magic_irgs.py   --scene " + '_'.join(scenes))

# ------------ 21-Scenes Leave one not + 14 test scenes ------------------------------------------------------- #
# id = [i for i in range(len(scenes))]
# id = [7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17] + [0, 1, 12, 18, 19, 20]
id = [8]
for i in id:
    test_scene = scenes[i]
    # test_scene = '_'.join(scenes[i:i+1] + new_test_scenes)
    train_scene = scenes[:i] + scenes[i+1:]
    train_scene = '_'.join(train_scene)

# # ------------ 21-Scenes (2010) for training /  (2011) scenes for testing ------------------------------------- #
# new_test_scenes_2011 = ["20110530", "20110613", "20110627", "20110709", "20110710", 
#                         "20110720", "20110725", "20111006", "20111013", "20111029"]

# for i in id:
# # for i in range(1):
#     test_scene = '_'.join(new_test_scenes_2011)
#     train_scene = '_'.join(scenes)

# # # ------------ 35-Scenes Cross-Validation 20/15 --------------------------------------------------------------- #
# for i in range(10):
#     sc = scenes + new_test_scenes
#     np.random.seed(i)
#     np.random.shuffle(sc)
    
#     test_scene = '_'.join(sc[20:])
#     train_scene = '_'.join(sc[:20])

    Schedule.append("python train_IRGS_trans_EndToEnd.py" + multi_gpu_config + "\
                                                --sweep True \
                                                --save_samples True \
                                                --batch_size 8 \
                                                --patch_size 128 \
                                                --patch_overlap 0.3 \
                                                \
                                                --mode end_to_end \
                                                --loss_term end_to_end \
                                                --mix_images True \
                                                --random_tokens True \
                                                \
                                                --irgs_classes 100 \
                                                --irgs_iter 120 \
                                                --token_option superpixels \
                                                --max_length 400 \
                                                \
                                                --train_path " + train_scene + " \
                                                --test_path " + test_scene + " \
                                                --model_name " + "model_{}".format(str(i)))

    # Schedule.append("python test_IRGS_trans_EndToEnd.py \
    #                                             --use_gpu True \
    #                                             --batch_size 128 \
    #                                             --patch_size 128 \
    #                                             --patch_overlap 0.9 \
    #                                             \
    #                                             --mode end_to_end \
    #                                             --loss_term transformer \
    #                                             --stage transformer \
    #                                             \
    #                                             --irgs_classes 10 \
    #                                             --irgs_iter 120 \
    #                                             --token_option superpixels \
    #                                             --max_length 400 \
    #                                             \
    #                                             --test_path " + test_scene + " \
    #                                             --model_name " + "model_{}".format(str(i)))

    # Schedule.append("python test_IRGS_trans_EndToEnd.py \
    #                                             --use_gpu True \
    #                                             --batch_size 128 \
    #                                             --patch_size 128 \
    #                                             --patch_overlap 0.9 \
    #                                             \
    #                                             --mode end_to_end \
    #                                             --loss_term end_to_end \
    #                                             --stage transformer \
    #                                             \
    #                                             --irgs_classes 10 \
    #                                             --irgs_iter 120 \
    #                                             --token_option superpixels \
    #                                             --max_length 400 \
    #                                             \
    #                                             --test_path " + test_scene + " \
    #                                             --model_name " + "model_{}".format(str(i)))

    # Schedule.append("python test_IRGS_trans_EndToEnd.py \
    #                                             --use_gpu True \
    #                                             --batch_size 128 \
    #                                             --patch_size 128 \
    #                                             --patch_overlap 0.9 \
    #                                             \
    #                                             --mode end_to_end \
    #                                             --loss_term end_to_end \
    #                                             --stage cnn \
    #                                             \
    #                                             --irgs_classes 10 \
    #                                             --irgs_iter 120 \
    #                                             --token_option superpixels \
    #                                             --max_length 400 \
    #                                             \
    #                                             --test_path " + test_scene + " \
    #                                             --model_name " + "model_{}".format(str(i)))

    # Schedule.append("python test_IRGS_trans_EndToEnd.py \
    #                                             --use_gpu True \
    #                                             --batch_size 128 \
    #                                             --patch_size 128 \
    #                                             --patch_overlap 0.9 \
    #                                             \
    #                                             --mode multi_stage \
    #                                             --loss_term end_to_end \
    #                                             --stage cnn \
    #                                             \
    #                                             --irgs_classes 10 \
    #                                             --irgs_iter 120 \
    #                                             --token_option superpixels \
    #                                             --max_length 400 \
    #                                             \
    #                                             --test_path " + test_scene + " \
    #                                             --model_name " + "model_{}".format(str(i)))

    # Schedule.append("python test_IRGS_trans_EndToEnd.py \
    #                                             --use_gpu True \
    #                                             --batch_size 128 \
    #                                             --patch_size 128 \
    #                                             --patch_overlap 0.9 \
    #                                             \
    #                                             --mode multi_stage \
    #                                             --loss_term end_to_end \
    #                                             --stage transformer \
    #                                             \
    #                                             --irgs_classes 10 \
    #                                             --irgs_iter 120 \
    #                                             --token_option superpixels \
    #                                             --max_length 400 \
    #                                             \
    #                                             --test_path " + test_scene + " \
    #                                             --model_name " + "model_{}".format(str(i)))


# # ------------------------------------------------------------------------------------------
    # # # # CNN
    # # --token_size 1 --num_heads 6 --embed_dim  48 \
    # Schedule.append("python train_isic.py    \
    #                                         --patch_size 256 \
    #                                         --patch_overlap 0.6 \
    #                                         --lr 1e-4 \
    #                                         \
    #                                         --batch_size 16 \
    #                                         --train_path " + train_scene + " --model_name " + "model_{}".format(str(i)))

    # Schedule.append("python test_isic.py     \
    #                                         --patch_size 256 \
    #                                         --patch_overlap 0.9 \
    #                                         --use_gpu 1 \
    #                                         --test_path " + test_scene + " --model_name " + "model_{}".format(str(i)))


#     # # # # # # IRGS Transformer superpixels
#     Schedule.append("python train_IRGS_trans.py \
#                                                 --patch_size 128 \
#                                                 --patch_overlap 0.2 \
#                                                 --batch_size 8 \
#                                                 --irgs_classes 100 \
#                                                 --max_length 400 \
#                                                 --token_option superpixels \
#                                                 \
#                                                 --irgs_iter 120 \
#                                                 --num_heads 6 \
#                                                 --embed_dim  384 \
#                                                 \
#                                                 --train_path " + train_scene + " \
#                                                 --test_path " + test_scene + " \
#                                                 --model_name " + "model_{}".format(str(i)))
#     Schedule.append("python test_IRGS_trans.py  \
#                                                 --patch_size 3000 \
#                                                 --patch_overlap 0.9 \
#                                                 --irgs_classes 15 \
#                                                 --token_option superpixels \
#                                                 \
#                                                 --irgs_iter 120 \
#                                                 --num_heads 6 \
#                                                 --embed_dim  384 \
#                                                 \
#                                                 --use_gpu 1 \
#                                                 --test_path " + test_scene + " --model_name " + "model_{}".format(str(i)))


#     # # # # # IRGS Transformer clusters
#     Schedule.append("python train_IRGS_trans.py \
#                                                 --patch_size 128 \
#                                                 --patch_overlap 0.2 \
#                                                 --batch_size 8 \
#                                                 --irgs_classes 100 \
#                                                 --max_length 100 \
#                                                 --token_option clusters \
#                                                 \
#                                                 --irgs_iter 120 \
#                                                 --num_heads 6 \
#                                                 --embed_dim  384 \
#                                                 \
#                                                 --train_path " + train_scene + " \
#                                                 --test_path " + test_scene + " \
#                                                 --model_name " + "model_{}".format(str(i)))
#     Schedule.append("python test_IRGS_trans.py  \
#                                                 --patch_size 3000 \
#                                                 --patch_overlap 0.9 \
#                                                 --irgs_classes 15 \
#                                                 --token_option clusters \
#                                                 \
#                                                 --irgs_iter 120 \
#                                                 --num_heads 6 \
#                                                 --embed_dim  384 \
#                                                 \
#                                                 --use_gpu 1 \
#                                                 --test_path " + test_scene + " --model_name " + "model_{}".format(str(i)))


    # # Schedule.append("python major_voting.py   --scene " + test_scene)

for i in range(len(Schedule)):
    os.system(Schedule[i])
