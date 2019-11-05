# [In Progress] Local modification from original architectures

## SqueezeNet

  Changed from `stride=1` to `stride=2` in the final_conv layer in compatible with `scale_factor=2` when upsampling in unet.
