"""

"""

import os
from mrs_utils import misc_utils


class Args:
    def __init__(self):
        self.gpu = '0'
        self.epochs = 100
        self.save_epoch = 20
        self.resume_epoch = 0
        self.decay_step = [80]
        self.batch_size = 5
        self.sfn = 64
        self.save_root = r'/hdd6/Models/mrs'
        self.ds_name = 'inria'
        self.encoder_name = 'vgg16'
        self.decoder_name = 'unet'
        self.criterion_name = 'xent,iou'
        self.random_seed = 1
        self.learn_rate_encoder = 1e-4
        self.learn_rate_decoder = 1e-4
        self.decay_rate = 0.1
        self.finetune_dir = None
        self.bp_loss_idx = 0
        self.input_size = (512, 512)

        decay_str = '_'.join([str(ds) for ds in self.decay_step])
        dr_str = str(self.decay_rate).replace('.', 'p')
        self.save_dir = 'ec{}_dc{}_ds{}_lre{}_lrd{}_ep{}_bs{}_sfn{}_ds{}_dr{}'.\
            format(self.encoder_name, self.decoder_name, self.ds_name, self.learn_rate_encoder, self.learn_rate_decoder,
                   self.epochs, self.batch_size, self.sfn, decay_str, dr_str)
        self.save_dir = os.path.join(self.save_root, self.save_dir)

        self.ds_name = misc_utils.stem_string(self.ds_name)
        if self.ds_name == 'inria':
            self.num_classes = 2
            self.data_dir = r'/hdd/mrs/inria/ps512_pd0_ol/patches'
            self.train_file = r'/hdd/mrs/inria/ps512_pd0_ol/file_list_train.txt'
            self.valid_file = r'/hdd/mrs/inria/ps512_pd0_ol/file_list_valid.txt'
        else:
            raise NotImplementedError('Dataset: {} is not supported'.format(self.ds_name))
