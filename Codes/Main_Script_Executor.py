"""
No@
"""

# 21 scenes 
scenes = ['20100418', '20100426', '20100510', '20100524', '20100605', '20100623', '20100629', 
          '20100712', '20100721', '20100730', '20100807', '20100816', '20100907', '20100909', 
          '20101003', '20101021', '20101027', '20101114', '20101120', '20101206', '20101214']

new_test_scenes = ["20100704", "20101014", "20101017", "20101025", "20110530", 
                   "20110613", "20110627", "20110709", "20110710", 
                   "20110720", "20110725", "20111006", "20111013", "20111029"]

# 35 scenes 
scenes = scenes + new_test_scenes

import os
from icecream import ic
import numpy as np
import wandb

import argparse
def Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default=8, type=int, help='')
    parser.add_argument('--train', default=1, type=int, help='')
    parser.add_argument('--exp', default=1, type=int, help='')

    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')    
    parser.add_argument('--nodes', default=1, type=int, help='')    
    args = parser.parse_args()
    return args
args = Arguments()
multi_gpu_config = []
for i, j in zip(args.__dict__.keys(), args.__dict__.values()):
    if i not in ['model_id', 'train', 'exp']:
        multi_gpu_config.append(' --{} {} '.format(i, j))
multi_gpu_config = ''.join(multi_gpu_config)


Schedule = []
Schedule = ["wandb login e478171c2941cc8ddc5a71663e36f613042dfc6e"]
Schedule.append("wandb offline")

# # # ------------ PREPARE DATA
# # scenes = scenes[:(len(scenes)//2)]
# scenes = scenes[(len(scenes)//2):]
# os.system("python prepare_data.py   --save_samples True \
#                                     --train_path " + '_'.join(scenes) + " \
#                                     --test_path "  + '_'.join(scenes)
#           )
# exit()


# IRGS Exp
# Schedule.append("python magic_irgs.py   --scene " + '_'.join(scenes))

# ------------ 21-Scenes Leave one not + 14 test scenes ------------------------------------------------------- #
# id = [i for i in range(len(scenes))]
# id = [7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17] + [0, 1, 12, 18, 19, 20]
# id = [8]
# for model_id in id:
# -----
if True:
    model_id = args.model_id

    test_scene = scenes[model_id]
    # test_scene = '_'.join(scenes[model_id:model_id+1] + new_test_scenes)
    train_scene = scenes[:model_id] + scenes[model_id+1:]
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

# # # ------------ Experiments --------------------------------------------------------------- #
    if args.train:
        # # TRAIN - MULTI - STAGE
        if args.exp == 0:
            Schedule.append("python train_IRGS_trans_EndToEnd.py" + multi_gpu_config + "\
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --mode multi_stage \
                                                        --stage cnn \
                                                        \
                                                        --train_path " + train_scene + " \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )

        elif args.exp == 1:
            Schedule.append("python train_IRGS_trans_EndToEnd.py" + multi_gpu_config + "\
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --mode multi_stage \
                                                        --stage transformer \
                                                        --mix_images True \
                                                        --random_tokens True \
                                                        \
                                                        --train_path " + train_scene + " \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )
        elif args.exp == 4:
            Schedule.append("python train_IRGS_trans_EndToEnd.py" + multi_gpu_config + "\
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --mode multi_stage \
                                                        --stage end_to_end \
                                                        --mix_images True \
                                                        --random_tokens True \
                                                        \
                                                        --train_path " + train_scene + " \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )
        
        # TRAIN - END - TO - END
        elif args.exp == 2:
            Schedule.append("python train_IRGS_trans_EndToEnd.py" + multi_gpu_config + "\
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --mode end_to_end \
                                                        --loss_term end_to_end \
                                                        --mix_images True \
                                                        --random_tokens True \
                                                        --train_path " + train_scene + " \
                                                        --test_path " + test_scene + " \
                                                        \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )

        elif args.exp == 3:
            Schedule.append("python train_IRGS_trans_EndToEnd.py" + multi_gpu_config + "\
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --mode end_to_end \
                                                        --loss_term transformer \
                                                        --mix_images True \
                                                        --random_tokens True \
                                                        --train_path " + train_scene + " \
                                                        --test_path " + test_scene + " \
                                                        \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )
            
    else:

        # TEST - MULTI - STAGE
        if args.exp == 0:
            Schedule.append("python test_IRGS_trans_EndToEnd.py \
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --use_gpu True \
                                                        \
                                                        --mode multi_stage \
                                                        --stage cnn \
                                                        --irgs_classes 15 \
                                                        \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )

        elif args.exp == 1:
            Schedule.append("python test_IRGS_trans_EndToEnd.py \
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --use_gpu True \
                                                        \
                                                        --mode multi_stage \
                                                        --stage transformer \
                                                        --irgs_classes 15 \
                                                        \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )
        elif args.exp == 4:
            Schedule.append("python test_IRGS_trans_EndToEnd.py \
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --use_gpu True \
                                                        \
                                                        --mode multi_stage \
                                                        --loss_term end_to_end\
                                                        --stage cnn \
                                                        --irgs_classes 15 \
                                                        \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )
            Schedule.append("python test_IRGS_trans_EndToEnd.py \
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --use_gpu True \
                                                        \
                                                        --mode multi_stage \
                                                        --loss_term end_to_end\
                                                        --stage transformer \
                                                        --irgs_classes 15 \
                                                        \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )

        # TEST - END - TO - END
        elif args.exp == 2:
            Schedule.append("python test_IRGS_trans_EndToEnd.py \
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --use_gpu True \
                                                        \
                                                        --mode end_to_end \
                                                        --loss_term end_to_end \
                                                        --stage cnn \
                                                        --irgs_classes 15 \
                                                        \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )
            Schedule.append("python test_IRGS_trans_EndToEnd.py \
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --use_gpu True \
                                                        \
                                                        --mode end_to_end \
                                                        --loss_term end_to_end \
                                                        --stage transformer \
                                                        --irgs_classes 15 \
                                                        \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )
        elif args.exp == 3:
            Schedule.append("python test_IRGS_trans_EndToEnd.py \
                                                        --num_workers " + str(args.num_workers) + " \
                                                        --use_gpu True \
                                                        \
                                                        --mode end_to_end \
                                                        --loss_term transformer \
                                                        --stage transformer \
                                                        --irgs_classes 15 \
                                                        \
                                                        --test_path " + test_scene + " \
                                                        --model_name " + "model_{}".format(str(model_id))
                            )



    # # Schedule.append("python major_voting.py   --scene " + test_scene)
#%%
if __name__ == '__main__':
    for i in range(len(Schedule)):
        os.system(Schedule[i])
