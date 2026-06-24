# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------
from PIL import Image, ImageOps
import copy
import torch
import random
import numpy as np
import torchvision


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, file_info=None):
        for t in self.transforms:
            if file_info is not None and isinstance(t, WERAugment):
                image, label = t(image, label, file_info)
            else:
                image = t(image)
        return image, label


class WERAugment(object):
    def __init__(self, boundary_path):
        self.boundary_dict = np.load(boundary_path, allow_pickle=True).item()
        self.K = 3

    def __call__(self, video, label, file_info):
        ind = np.arange(len(video)).tolist()
        if file_info not in self.boundary_dict.keys():
            return video, label
        binfo = copy.deepcopy(self.boundary_dict[file_info])
        binfo = [0] + binfo + [len(video)]
        k = np.random.randint(min(self.K, len(label) - 1))
        for i in range(k):
            ind, label, binfo = self.one_operation(ind, label, binfo)
        ret_video = [video[i] for i in ind]
        return ret_video, label

    def one_operation(self, *inputs):
        prob = np.random.random()
        if prob < 0.3:
            return self.delete(*inputs)
        elif 0.3 <= prob < 0.7:
            return self.substitute(*inputs)
        else:
            return self.insert(*inputs)

    @staticmethod
    def delete(ind, label, binfo):
        del_wd = np.random.randint(len(label))
        ind = ind[:binfo[del_wd]] + ind[binfo[del_wd + 1]:]
        duration = binfo[del_wd + 1] - binfo[del_wd]
        del label[del_wd]
        binfo = [i for i in binfo[:del_wd]] + [i - duration for i in binfo[del_wd + 1:]]
        return ind, label, binfo

    @staticmethod
    def insert(ind, label, binfo):
        ins_wd = np.random.randint(len(label))
        ins_pos = np.random.choice(binfo)
        ins_lab_pos = binfo.index(ins_pos)

        ind = ind[:ins_pos] + ind[binfo[ins_wd]:binfo[ins_wd + 1]] + ind[ins_pos:]
        duration = binfo[ins_wd + 1] - binfo[ins_wd]
        label = label[:ins_lab_pos] + [label[ins_wd]] + label[ins_lab_pos:]
        binfo = binfo[:ins_lab_pos] + [binfo[ins_lab_pos - 1] + duration] + [i + duration for i in binfo[ins_lab_pos:]]
        return ind, label, binfo

    @staticmethod
    def substitute(ind, label, binfo):
        sub_wd = np.random.randint(len(label))
        tar_wd = np.random.randint(len(label))

        ind = ind[:binfo[tar_wd]] + ind[binfo[sub_wd]:binfo[sub_wd + 1]] + ind[binfo[tar_wd + 1]:]
        label[tar_wd] = label[sub_wd]
        delta_duration = binfo[sub_wd + 1] - binfo[sub_wd] - (binfo[tar_wd + 1] - binfo[tar_wd])
        binfo = binfo[:tar_wd + 1] + [i + delta_duration for i in binfo[tar_wd + 1:]]
        return ind, label, binfo


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BICUBIC


    def __call__(self, img_group):
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in
                         crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret


class Stack(object):

    def __init__(self, roll=True):
        self.roll = roll

    def __call__(self, img_group):

        if self.roll:
            return np.concatenate([np.array(x).transpose(2, 0, 1) for x in img_group], axis=0)
        else:
            return np.concatenate(img_group, axis=0)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).contiguous()
        return img.float().div(255.) if self.div else img.float()


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        tensor = tensor.view((-1, 3) + tensor.size()[-2:])

        return tensor


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2, frame_interval=1):
        self.min_len = 32
        self.max_len = int(np.ceil(230/frame_interval))
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):

        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]

