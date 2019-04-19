"""

"""


# Built-in
import time
import argparse

# Libs
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

# Own modules
from model import unet
from data import data_loader
from mrs_utils import misc_utils
from mrs_utils import xval_utils


DATA_FILE = r'/hdd/mrs/inria/file_list.txt'
INPUT_SIZE = 224
BATCH_SIZE = 8
GPU = 1
ENCODER_NAME = 'res101'
N_CLASS = 2
INIT_LR = 1e-4
MILESTONES = [20, 30]
DROP_RATE = 0.1
EPOCHS = 40
SAVE_DIR = r'/home/lab/Documents/bohao/code/mrs/model/model2.pt'
LOG_DIR = r'/home/lab/Documents/bohao/code/mrs/model/log/log2'
SAVE_EPOCH = 1


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default=DATA_FILE, help='path to the dataset file')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE, help='input size of the patches')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size in training')
    parser.add_argument('--gpu', type=int, default=GPU, help='which gpu to use')
    parser.add_argument('--encoder-name', type=str, default=ENCODER_NAME, help='which encoder to use for extractor, '
                                                                               'see model/model.py for more details')
    parser.add_argument('--n-class', type=int, default=N_CLASS, help='#classes in the output')
    parser.add_argument('--init-lr', type=float, default=INIT_LR, help='initial learning rate')
    parser.add_argument('--milestones', type=list, default=MILESTONES, help='milestones for multi step lr drop')
    parser.add_argument('--drop-rate', type=float, default=DROP_RATE, help='drop rate at each milestone in scheduler')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='num of epochs to train')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR, help='path to save the model')
    parser.add_argument('--log-dir', type=str, default=LOG_DIR, help='directory to save tensorboard summaries')
    parser.add_argument('--save-epoch', type=int, default=SAVE_EPOCH, help='model will be saved every #epochs')

    flags = parser.parse_args()
    return flags


def main(flags):
    # prepare data reader
    transforms = {
        'train': data_loader.JointCompose([
            data_loader.JointFlip(),
            data_loader.JointRotate(),
            data_loader.JointToTensor(),
        ]),
        'valid': data_loader.JointCompose([
            data_loader.JointToTensor(),
        ]),
    }
    transforms_ftr = {
        'train': torchvision.transforms.Compose({
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        }),
        'valid': torchvision.transforms.Compose({
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        }),
    }
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    file_list = misc_utils.load_file(flags.data_file)
    city_id_list = xval_utils.get_inria_city_id(file_list)
    file_list_train, file_list_valid = xval_utils.split_by_id(file_list, city_id_list, list(range(6)))
    reader = {'train':data_loader.RemoteSensingDataset(file_list=file_list_train, input_size=flags.input_size,
                                                       transform=transforms['train'], transform_ftr=transforms_ftr['train']),
              'valid':data_loader.RemoteSensingDataset(file_list=file_list_valid, input_size=flags.input_size,
                                                       transform=transforms['valid'], transform_ftr=transforms_ftr['valid'])}
    reader = {x: data.DataLoader(reader[x], batch_size=flags.batch_size, shuffle=True, num_workers=flags.batch_size,
                                 drop_last=True)
              for x in ['train', 'valid']}

    # build the model
    device = misc_utils.set_gpu(flags.gpu)
    model = unet.Unet(flags.encoder_name, flags.n_class).to(device)

    # make optimizers
    optm = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 0.1*flags.init_lr},
        {'params': model.decoder.parameters(), 'lr': flags.init_lr}
    ], lr=flags.init_lr)
    # Decay LR by a factor of drop_rate at each milestone
    scheduler = optim.lr_scheduler.MultiStepLR(optm, milestones=flags.milestones, gamma=flags.drop_rate)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # train the model
    start_time = time.time()
    model.train_model(device=device, epochs=flags.epochs, optm=optm, criterion=criterion, scheduler=scheduler,
                      reader=reader, save_dir=flags.save_dir, summary_path=flags.log_dir, rev_transform=inv_normalize,
                      save_epoch=flags.save_epoch)
    duration = time.time() - start_time
    print('Total time: {} hours'.format(duration/60/60))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
