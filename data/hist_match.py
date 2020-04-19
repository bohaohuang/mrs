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
    """
    Match the histogram between two datasets
    """
    def __init__(self, source_imgs):
        self.source_imgs = source_imgs
        self.source_hist = self.get_histogram(source_imgs, True)

    @staticmethod
    def get_histogram(img_files, progress=False):
        """
        Get the histogram of given list of images
        :param img_files: list of images, could be file names or numpy arrays
        :param progress: if True, will show a progress bar
        :return: a numpy array of size (3, 256) where each row represents histogram of certain color channel
        """
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
        """
        Adjust the given image so that its histogram matches the target distribution
        :param dist_t: the target histogram distribution
        :param dist_s: the source histogram distribution
        :param img_s: the source image
        :return: the adjusted image
        """
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

    def match_target_images(self, target_imgs, individual=False):
        """
        Match the given list of target images
        :param target_imgs: list of image files, could be file names or numpy arrays
        :param individual: if True, compute histogram of each target image respectively
        :return: a generator that yields adjusted image one each time
        """
        if not individual:
            target_hist = self.get_histogram(target_imgs)
        for target_img_file in target_imgs:
            if individual:
                target_hist = self.get_histogram([target_img_file])

                '''import scipy.signal
                color_list = ['r', 'g', 'b']
                for c in range(3):
                    plt.plot(target_hist[c, :], color_list[c])
                plt.show()'''

            if isinstance(target_img_file, str):
                img = misc_utils.load_file(target_img_file)
            else:
                img = target_img_file
            yield self.match_image(self.source_hist, target_hist, img)

    def vis_transform_pair(self, target_img_files):
        """
        Visualize a pair of sample
        :param target_img_files: list of target image files, a random of them will be chosen to display
        """
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


if __name__ == '__main__':
    pass
