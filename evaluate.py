"""

"""


# Built-in
import os

# Libs
import albumentations as A
from albumentations.pytorch import ToTensor

# Own modules
from mrs_utils import misc_utils
from network import network_io, network_utils


# Settings
GPU = 1
MODEL_DIR = r'/hdd6/Models/mrs/inria/ecresnet101_dcpspnet_dsinria_lre0.001_lrd0.01_ep80_bs5_ds50_dr0p1'
LOAD_EPOCH = 80
DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data/inria'
PATCHS_SIZE = (512, 512)


def main():
    device, _ = misc_utils.set_gpu(GPU)

    # init model
    args = network_io.load_config(MODEL_DIR)
    model = network_io.create_model(args)
    if LOAD_EPOCH:
        args['trainer']['epochs'] = LOAD_EPOCH
    ckpt_dir = os.path.join(MODEL_DIR, 'epoch-{}.pth.tar'.format(args['trainer']['epochs']))
    network_utils.load(model, ckpt_dir)
    print('Loaded from {}'.format(ckpt_dir))
    model.to(device)
    model.eval()

    # eval on dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensor(sigmoid=False),
    ])
    save_dir = os.path.join(r'/hdd/Results/mrs/inria', os.path.basename(network_utils.unique_model_name(args)))
    evaluator = network_utils.Evaluator('inria', DATA_DIR, tsfm_valid, device)
    evaluator.evaluate(model, PATCHS_SIZE, 2*model.lbl_margin,
                       pred_dir=save_dir, report_dir=save_dir)


if __name__ == '__main__':
    main()
