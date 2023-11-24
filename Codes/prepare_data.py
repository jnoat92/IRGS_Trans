from config import Arguments_train
from utils.dataloader import Data_proc, Calculate_norm_params, Extract_segments
from utils.utils import aux_obj, time_format
import time
import wandb
from icecream import ic

if __name__ == '__main__':

    wandb.init(project='prepare_data')

    args = Arguments_train()

    dataset = aux_obj()

    # =========== TRAINING
    start_time = time.time()
    dataset.train, dataset.val, _ = Data_proc(args, set_='train', sliding_window=True, aug=False, padding=False)
    # =========== TEST
    _,      _,       dataset.test = Data_proc(args, set_='test', aug=False)
    print('clipping time:', time_format(time.time() - start_time))

    ic(len(dataset.train))
    ic(len(dataset.val))
    ic(len(dataset.test))

    exit()

    # =========== Image segmentation using IRGS
    norm_params = Calculate_norm_params(args)
    start_time = time.time()

    dataset.train.file_paths.extend(dataset.val.file_paths)
    dataset.train.file_paths.extend(dataset.test.file_paths)
    ic(len(dataset.train))
    Extract_segments(dataset.train, norm_params, args)
    # Extract_segments(dataset.val, norm_params, args)
    # Extract_segments(dataset.test, norm_params, args)
    print('IRGS parallel time:', time_format(time.time() - start_time))
