from typing import Any, Union, List
import torch.nn as nn
import torch
from .model_lora import build_model, LoRALinear


__all__ = ["load"]
_MODELS = {
    'ViT-B/16': r'/data/che_xiao/my_project/AdaptSign-main/checkpoint/ViT_base_16.bin',
}


def merge_lora_weights(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # 计算 delta_W = A^T @ B^T * scaling
            delta_weight = (module.lora.lora_A.T @ module.lora.lora_B.T) * module.lora.scaling
            # 合并到原始权重
            module.weight.data += delta_weight.T
            print(f"✅ Merged LoRA into {name}")
            # 删除 lora 参数（可选）
            del module.lora


def inject_lora_into_vit(
        model: nn.Module,
        rank: int = 8,
        lora_alpha: int = 8,
        target_modules: list = None
):
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']

    for name, module in model.named_modules():
        # 跳过顶层模块（避免替换整个 ViT）
        if name == '':
            continue

        parent_name = name.split('.')[:-1]
        child_name = name.split('.')[-1]

        if isinstance(module, nn.Linear) and any(t in child_name for t in target_modules):
            #print(f"🔧 Replacing {name} with LoRALinear: {module.in_features} -> {module.out_features}")

            # 创建 LoRALinear
            lora_linear = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                lora_alpha=lora_alpha,
                bias=module.bias is not None
            )

            # 复制原始权重和偏置
            lora_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_linear.bias.data = module.bias.data.clone()

            # 替换模块
            parent_module = model
            if parent_name:
                for part in parent_name:
                    parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, lora_linear)

    return model


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _MODELS[name]
    else:
        raise RuntimeError(f"Model {name} not found")

    with open(model_path, 'rb') as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")

    model = build_model(state_dict).to(device)
    if str(device) == "cpu":
        model.float()

    target_modules = ['q_proj', 'v_proj', 'fc1', 'fc2']
    model = inject_lora_into_vit(model, rank=8, lora_alpha=8, target_modules=target_modules)
    return model

