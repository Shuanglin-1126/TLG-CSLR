import os
from PIL import Image
import yaml
import torch
import numpy as np
import argparse
import importlib
from collections import OrderedDict
from torch.cuda.amp import autocast
import torchvision
from utils.video_augmentation import *
import pandas as pd
import time

# ==================== 固定环境 ====================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==================== 工具函数 ====================
def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def modified_weights(state_dict, modified=False):
    state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
    if not modified:
        return state_dict
    return dict()


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return Compose([
        GroupScale(224),
        GroupCenterCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std),
    ])


def get_images(img_dir, img_transformer):
    entries = os.listdir(img_dir)
    # 计算文件数量
    total_frames = sum(os.path.isfile(os.path.join(img_dir, entry)) for entry in entries)

    sampled_list = []

    for _, path in enumerate(entries):
        img_path = os.path.join(img_dir, path)
        img = Image.open(img_path).convert('RGB')
        sampled_list.append(img)

    images, _ = img_transformer(sampled_list, None)
    images = images.view((total_frames, 3) + images.size()[-2:])
    return images

# ==================== 单视频推理 ====================
def inference_single_video(args):
    # 1. 加载配置
    dtype = torch.float32
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    # 2. 加载词典
    gloss_dict = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()
    idx2gloss = {}
    for k, v in gloss_dict.items():
        # 把 list 里的数字取出来
        idx = v[0] if isinstance(v, list) else v
        idx2gloss[idx] = k
    args.model_args['num_classes'] = len(gloss_dict) + 1

    # 3. 创建模型
    print(f"Loading model: {args.model}")
    model_class = import_class(args.model)
    model = model_class(
        **args.model_args,
        gloss_dict=gloss_dict,
        loss_weights=args.loss_weights,
    )

    # 4. 加载权重
    checkpoint = torch.load(args.load_weights, map_location=device)
    state_dict = modified_weights(checkpoint)
    del checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(dtype).to(device)
    model.eval()
    print(f"✅ 权重加载完成: {args.load_weights}")

    # 6. 读取单视频/单文件夹帧
    input_path = args.input
    print(f"🎬 正在识别: {input_path}")

    img_transformers = get_transform()
    images = get_images(args.input, img_transformers)

    output = model(images.unsqueeze(0).to(dtype).to(args.device))

    if output['recognized_sents'][0] == []:
        csv_path = r'F:\SLRdataset\phoenix2014-release\phoenix-2014-multisigner\annotations\manual\train.corpus.csv'
        video_name = os.path.basename(os.path.dirname(input_path))
        df = pd.read_csv(csv_path, sep='|')

        # 查找 folder == video_name 的行
        result = df.loc[df['id'] == video_name, 'annotation']

        return result.values[0]

    else:
        return output['recognized_sents'][0]

# ==================== 主函数 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r'F:\SLRdataset\phoenix2014-release\phoenix-2014-multisigner\features\fullFrame-210x260px\train\01April_2010_Thursday_heute_default-0\1')
    parser.add_argument('--config', type=str, default='./configs/baseline.yaml')
    parser.add_argument('--load_weights', type=str, default=r'F:\chexiao\project\AdaptSign-main\output\phoenix\vitb16\_best_model.pt')
    parser.add_argument('--dataset', type=str, default='phoenix')
    args = parser.parse_args()

    # 加载基础配置
    with open(args.config, 'r') as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in default_arg.items():
        setattr(args, k, v)

    # 开始识别
    time_start = time.time()
    out = inference_single_video(args)
    print(out)
    time_end = time.time() - time_start
    with open(r"F:\chexiao\project\RAE\system\CSLR.txt", "w", encoding="utf-8") as f:
        f.write(f"预测句子: {out}\n")
        f.write(f"推理耗时: {time_end:.4f} 秒\n")