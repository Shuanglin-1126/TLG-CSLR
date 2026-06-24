from utils.video_augmentation import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import yaml
import importlib
from PIL import Image
import os
import cv2


device = 'cuda:0'

def load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
    # 移除 'module.' 前缀
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model.to(device)

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def transform(input_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return Compose([
        GroupScale(input_size),
        GroupCenterCrop(input_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std),
    ])


def visualize_attention_map(image, attention, idx):
    """
    将注意力热力图叠加到原始图像上。

    Args:
        image (np.ndarray): 原始图像，OpenCV格式 (H, W, C)。
        attention_map (np.ndarray): 大小为 (num_patches) 的注意力权重数组。
        patch_size (int): ViT模型的图像块大小。

    Returns:
        np.ndarray: 叠加了热力图的图像。
    """

    # image = img_resize(image).permute(1,2,0).numpy()

    # 将一维的注意力权重重新形状为二维网格
    attention_map = np.zeros((14, 14))
    attention *= 10
    for i, val in enumerate(idx):
        h, w = val // 14, val % 14
        attention_map[h, w] = attention[i]

    # 放大热力图以匹配原始图像大小
    attention_map_resized = cv2.resize(attention_map, (224, 224))

    # 使用 matplotlib 生成一个颜色映射的热力图
    cmap = plt.get_cmap('jet')
    heatmap = cmap(attention_map_resized)

    # 将热力图转换为OpenCV格式 (0-255)
    heatmap_colored = (heatmap[..., :3] * 255).astype(np.float32)

    # 将热力图与原始图像叠加
    alpha = 0.3
    image = image.permute(1,2,0).numpy() * 255
    overlay_image = cv2.addWeighted(image, alpha, heatmap_colored, 1-alpha, 0)

    return overlay_image.astype(np.uint8)


if __name__ == '__main__':

    sparser = utils.get_parser()
    p = sparser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)

    arg = sparser.parse_args()

    model_class = import_class(arg.model)
    model = model_class(
        **arg.model_args,
        loss_weights=arg.loss_weights,
    )
    model = load_model(
        model,
        r'/data/che_xiao/my_project/AdaptSign-main/output/phoenix/vitb16/_best_model.pt')

    video_path = r'/data3/SLRdata/datasets/SLDataset/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/01October_2012_Monday_heute_default-0/1/'
    img_list = os.listdir(video_path)
    img_list = [os.path.join(video_path, i) for i in img_list]
    imgs = [Image.open(img_path).convert('RGB') for img_path in img_list]
    data_aug = transform(224)
    imgs, _ = data_aug(imgs, None)
    out = model(imgs.unsqueeze(0).to(device))
    attn = out['attn'][-1][:, 0, 8:].squeeze().detach().cpu().numpy()
    idx = out['idx'].squeeze().cpu().numpy()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    imgs = (imgs * std + mean).cpu()
    frame_count = 1
    for idx_, attn_, image in zip(idx, attn, imgs[1:]):
        new_img = visualize_attention_map(image, attn_, idx_)
        cv2.imwrite(f"/data/che_xiao/my_project/AdaptSign-main/output/key_pathch_attn/0{frame_count:03d}.jpg", new_img)
        frame_count += 1




