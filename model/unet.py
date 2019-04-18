"""

"""


# Built-in
import copy
import time

# Libs
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# Own modules
from mrs_utils import misc_utils


class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.
    ...
    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.
        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.
        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = torch.argmax(outputs, dim=1)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return iou.mean()


class Unet(nn.Module):
    def __init__(self, encoder_name, n_class):
        super(Unet, self).__init__()
        self.encoder_name = misc_utils.stem_string(encoder_name)
        if self.encoder_name == 'res101':
            encoder = list(torchvision.models.resnet101(pretrained=True).children())
            self.conv1 = nn.Sequential(*encoder[:5])
            self.conv2 = encoder[5]
            self.conv3 = encoder[6]
            self.conv4 = encoder[7]
            self.up1 = up(2048, 1024)
            self.up2 = up(1024, 512)
            self.up3 = up(512, 256)
            self.final_conv1 = nn.Conv2d(256, 128, 3, padding=1)
            self.final_conv2 = nn.Conv2d(128, 128, 3, padding=1)
            self.classify = nn.Conv2d(128, n_class, 1)
        else:
            raise NotImplementedError('Encoder name {} not recognized'.format(self.encoder_name))

    def forward(self, x):
        if self.encoder_name == 'res101':
            s1 = self.conv1(x)
            s2 = self.conv2(s1)
            s3 = self.conv3(s2)
            s4 = self.conv4(s3)
            x = self.up1.forward(s4, s3)
            x = self.up2.forward(x, s2)
            x = self.up3.forward(x, s1)
            x = F.interpolate(x, scale_factor=4, mode='bilinear')
            x = self.final_conv1(x)
            x = self.final_conv2(x)
            x = self.classify(x)
            return x
        else:
            raise NotImplementedError('Encoder name {} not recognized'.format(self.encoder_name))

    @staticmethod
    def mask_to_rgb(tensor_mask):
        tensor_mask = tensor_mask.cpu()
        tensor_mask = torch.stack((tensor_mask, tensor_mask, tensor_mask), dim=0)
        return (255 * tensor_mask).float()

    def train_model(self, device, epochs, optm, criterion, scheduler, reader, save_dir,
          summary_path, rev_transform, save_epoch=5, verb_step=100):
        # TODO fix tensorboard image summary
        # TODO learning rate for different part
        writer = SummaryWriter(summary_path)
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 0.0
        step = 0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            for phase in ['train', 'valid']:
                start_time = time.time()
                running_loss = 0.0
                running_iou = 0.0
                if phase == 'train':
                    scheduler.step()
                    self.train()  # Set model to training mode
                else:
                    self.eval()  # Set model to evaluate mode

                reader_cnt = 1
                ftr, lbl, pred = None, None, None
                for ftr, lbl in reader[phase]:
                    reader_cnt += 1

                    if phase == 'train':
                        step += 1

                    ftr = ftr.to(device)
                    lbl = torch.squeeze(lbl, dim=1).long().to(device)
                    pred = self.forward(ftr)

                    # zero the parameter gradients
                    optm.zero_grad()

                    #loss = cross_entropy2d(input=pred, target=lbl)
                    #loss = criterion(F.log_softmax(pred, 1), lbl)
                    loss = criterion(pred, lbl)
                    iou = iou_pytorch(pred, lbl)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optm.step()

                    running_loss = running_loss * (reader_cnt - 1) / reader_cnt + loss.item() / reader_cnt
                    running_iou = running_iou * (reader_cnt - 1) / reader_cnt + iou.item() / reader_cnt
                    if phase == 'train' and reader_cnt % verb_step == 0:
                        writer.add_scalar('loss_train', loss.item(), step)
                        writer.add_scalar('iou_train', iou.item(), step)
                        elapsed = time.time() - start_time
                        print('Epoch {}, {} Step:{} Loss: {:.4f}, IoU: {:.4f}, Duration: {:.0f}m {:.0f}s'.format(
                            epoch, phase, reader_cnt, loss.item(), iou.item(), elapsed // 60, elapsed % 60))

                if phase == 'valid':
                    for img_cnt in range(ftr.shape[0]):
                        lbl_img = self.mask_to_rgb(lbl[img_cnt])
                        pred_img = self.mask_to_rgb(torch.argmax(pred[img_cnt].cpu(), dim=0))
                        tb_img = torchvision.utils.make_grid(
                            [rev_transform(ftr[img_cnt].cpu()), lbl_img, pred_img])
                        writer.add_image('image_valid_{}'.format(img_cnt), tb_img, epoch)
                    writer.add_scalar('loss_valid', running_loss, epoch)
                    writer.add_scalar('iou_valid', running_iou, epoch)
                    elapsed = time.time() - start_time
                    print('Epoch {}, {} Step:{} Loss: {:.4f}, IoU: {:.4f}, Duration: {:.0f}m {:.0f}s'.format(
                        epoch, phase, reader_cnt, running_loss, running_iou, elapsed // 60, elapsed % 60))
                    if running_loss < best_loss:
                        # deep copy the model
                        best_loss = running_loss
                        best_model_wts = copy.deepcopy(self.state_dict())

            if epoch % save_epoch == 0:
                torch.save(best_model_wts, save_dir)
        self.load_state_dict(best_model_wts)



if __name__ == '__main__':
    unet = Unet('res101', 2)
    sample = torch.rand((5, 3, 224, 224))
    device = misc_utils.set_gpu(1)
    sample.to(device)
    unet.forward(sample)
