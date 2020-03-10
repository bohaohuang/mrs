"""

"""


# Built-in

# Libs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Own modules
from mrs_utils import misc_utils


class HistMatcher(object):
    def __init__(self, source_imgs):
        self.source_imgs = source_imgs
        self.source_hist = self.get_histogram(source_imgs, True)

    @staticmethod
    def get_histogram(img_files, progress=False):
        hist = np.zeros((3, 256))
        if progress:
            pbar = tqdm(img_files)
        else:
            pbar = img_files
        for img_file in pbar:
            if isinstance(img_file, str):
                img = misc_utils.load_file(img_file)
            else:
                img = img_file
            for channel in range(3):
                img_hist, _ = np.histogram(img[:, :, channel].flatten(), bins=np.arange(0, 257))
                hist[channel, :] += img_hist
        return hist

    @staticmethod
    def match_image(dist_t, dist_s, img_s):
        bins = np.arange(dist_s.shape[1] + 1)
        im_res = np.zeros_like(img_s)
        for d in range(dist_s.shape[0]):
            im_hist_s = dist_s[d, :] / np.sum(dist_s[d, :])
            im_hist_t = dist_t[d, :] / np.sum(dist_t[d, :])

            cdfsrc = im_hist_s.cumsum()
            cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)
            cdftint = im_hist_t.cumsum()
            cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)

            im2 = np.interp(img_s[:, :, d].flatten(), bins[:-1], cdfsrc)
            im3 = np.interp(im2, cdftint, bins[:-1])
            im_res[:, :, d] = im3.reshape((img_s.shape[0], img_s.shape[1]))
        return im_res

    def match_target_images(self, target_imgs):
        target_hist = self.get_histogram(target_imgs)
        for target_img_file in target_imgs:
            img = misc_utils.load_file(target_img_file)
            yield self.match_image(self.source_hist, target_hist, img)

    def vis_transform_pair(self, target_img_files):
        def plot_hist(hist, smooth=False):
            import scipy.signal
            color_list = ['r', 'g', 'b']
            for c in range(3):
                if smooth:
                    plt.plot(scipy.signal.savgol_filter(hist[c, :], 11, 2), color_list[c])
                else:
                    plt.plot(hist[c, :], color_list[c])

        rand_img = misc_utils.load_file(np.random.choice(self.source_imgs, 1)[0])
        target_img = misc_utils.load_file(np.random.choice(target_img_files, 1)[0])
        target_hist = self.get_histogram(target_img_files)
        match_img = self.match_image(self.source_hist, target_hist, target_img)

        plt.figure(figsize=(15, 8))
        plt.subplot(231)
        plt.imshow(rand_img)
        plt.axis('off')
        plt.subplot(234)
        plot_hist(self.source_hist)
        plt.subplot(232)
        plt.imshow(target_img)
        plt.axis('off')
        plt.subplot(235)
        plot_hist(target_hist)
        plt.subplot(233)
        plt.imshow(match_img)
        plt.axis('off')
        plt.subplot(236)
        plot_hist(self.get_histogram([match_img]), smooth=True)
        plt.tight_layout()
        plt.show()


def main():
    rgb_files = misc_utils.get_files([r'/media/ei-edl01/data/uab_datasets/Austin/data', 'Original_Tiles'], '*_RGB.tif')
    target_files = misc_utils.get_files(['/media/ei-edl01/user/bh163/lbnl/dataset'], '*.jpg')
    hm = HistMatcher(rgb_files)
    hm.vis_transform_pair(target_files)


if __name__ == '__main__':
    main()
