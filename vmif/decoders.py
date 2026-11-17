import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import MODELS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party'))

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGENET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
INCEPTION = (torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))


def _state(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}


def _norm(x, stats):
    m, s = stats
    return (x - m.to(x.device)) / s.to(x.device)


def to_tensor(batch):
    return torch.from_numpy(np.stack(batch)).to(DEV).permute(0, 3, 1, 2).float() / 255.


def load_gigatime():
    from gigatime_archs import gigatime
    m = gigatime(num_classes=23)
    m.load_state_dict(_state(MODELS['gigatime']), strict=True)
    return m.to(DEV).eval()


class _ConvLoRA(torch.nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base_layer = base
        self.lora_A = torch.nn.ModuleDict({'default': torch.nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride, bias=False)})
        self.lora_B = torch.nn.ModuleDict({'default': torch.nn.Conv2d(r, base.out_channels, 1, bias=False)})
        self.scaling = alpha / r

    def forward(self, x):
        return self.base_layer(x) + self.lora_B['default'](self.lora_A['default'](x)) * self.scaling


def load_flash():
    from gigatime_flash_model import GigaTIMEFlash
    m = GigaTIMEFlash(num_classes=23)
    pe = m.encoder.base_model.model.patch_embed
    pe.proj = _ConvLoRA(pe.proj)
    target, loaded = m.state_dict(), {}
    for key, value in _state(MODELS['flash']).items():
        for c in (key, key.replace('.base_layer.', '.'), key.replace('encoder.', 'encoder.base_model.model.', 1)):
            if c in target and target[c].shape == value.shape:
                loaded[c] = value
                break
    assert len(loaded) == len(target)
    m.load_state_dict(loaded, strict=True)
    return m.to(DEV).eval()


def load_hex():
    importlib.import_module('musk.modeling')
    from hex_architecture import CustomModel
    m = CustomModel(visual_output_dim=1024, num_outputs=48)
    assert not m.load_state_dict(torch.load(MODELS['hex'], map_location='cpu', weights_only=False), strict=False).unexpected_keys
    return m.to(DEV).eval()


def load_musk():
    importlib.import_module('musk.modeling')
    from musk import utils
    from timm import create_model
    m = create_model('musk_large_patch16_384', vocab_size=64010)
    utils.load_model_and_may_interpolate('hf_hub:xiangjx/musk', m, 'model|module', '')
    return m.to(DEV).eval()


def load_uni():
    import timm
    m = timm.create_model('hf-hub:MahmoodLab/UNI', pretrained=True, init_values=1e-5, dynamic_img_size=True)
    return m.to(DEV).eval()


def windows(x, w=256):
    b = x.shape[0]
    return x.unfold(2, w, w).unfold(3, w, w).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, w, w), b


@torch.no_grad()
def gigatime_mean(model, x):
    return torch.sigmoid(model(_norm(x, IMAGENET))).mean((2, 3))


@torch.no_grad()
def hex_pred(model, x):
    x = F.interpolate(x, size=(384, 384), mode='bilinear', antialias=True, align_corners=False)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
        return model(_norm(x, INCEPTION))[0].float()


@torch.no_grad()
def musk_feat(model, x):
    x = F.interpolate(x, size=(384, 384), mode='bilinear', antialias=True, align_corners=False)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
        return model(image=_norm(x, INCEPTION), with_head=False, out_norm=False)[0].float()


@torch.no_grad()
def uni_feat(model, x):
    x = F.interpolate(x, size=(224, 224), mode='bilinear', antialias=True, align_corners=False)
    return model(_norm(x, IMAGENET))
