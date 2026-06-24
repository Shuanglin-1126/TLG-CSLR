import os
import sys
import pdb
import glob
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
from utils.video_augmentation import *

sys.path.append("..")
global kernel_sizes
kernel_sizes = ['K5', "P2", 'K5', "P2"]


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', num_gloss=-1, mode="train",
                 transform_mode=True, datatype='video', drop_ratio=1.,
                 frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.dataset = dataset
        self.input_size = input_size
        # global kernel_sizes
        # kernel_sizes = kernel_size
        self.frame_interval = frame_interval  # not implemented for read_features()
        self.image_scale = image_scale  # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"/data/che_xiao/my_project/CSLR/dataset/preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        self.data_aug = self.transform()

    def __getitem__(self, idx):
        input_data, label, fi = self.read_video(idx)
        images, label = self.data_aug(input_data, label)
        # images = images.view((-1, 3) + images.size()[-2:])
        return images, torch.LongTensor(label), self.inputs_list[idx]

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix2014-T' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'][:-6] + r'/1' + fi['folder'][-6:])
        elif 'phoenix2014' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-210x260px/" + fi['folder'])
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [Image.open(img_path).convert('RGB') for img_path in img_list], label_list, fi


    def transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # TemporalRescale(0.2, self.frame_interval),
        if self.transform_mode == "train":
            return Compose([
                GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(),
                Stack(),
                ToTorchFormatTensor(),
                GroupNormalize(mean=mean, std=std),
                TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            return Compose([
                GroupScale(self.input_size),
                GroupCenterCrop(self.input_size),
                Stack(),
                ToTorchFormatTensor(),
                GroupNormalize(mean=mean, std=std),
            ])


    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))

        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride
                left_pad += int((int(ks[1]) - 1) / 2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor(
                [np.ceil(len(vid) / total_stride) * total_stride + 2 * left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        prefix=None,
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()