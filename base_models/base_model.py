import torch
from torch import nn
from WideResnet import WideResNet

from utils import split_up_model
from Resnet import ResNet18, PreActResNet18, ResNetCifar

class BaseModel(torch.nn.Module):
    """
    Change the model structure to perform the adaptation "AdaContrast" for other datasets
    """

    def __init__(self, model, arch_name):
        super().__init__()
        if isinstance(model, WideResNet):
            self.nChannels = model.nChannels
        self.arch_name = arch_name
        if arch_name != 'vit':
            self.encoder, self.fc = split_up_model(model)
            if isinstance(self.fc, nn.Sequential):
                for module in self.fc.modules():
                    if isinstance(module, nn.Linear):
                        self._num_classes = module.out_features
                        self._output_dim = module.in_features
            elif isinstance(self.fc, nn.Linear):
                self._num_classes = self.fc.out_features
                self._output_dim = self.fc.in_features
            else:
                raise ValueError("Unable to detect output dimensions")
        else:
            self.model = model
            self._num_classes = model.head.out_features
            self._output_dim = model.head.in_features

    def forward(self, x, return_feats=False):
        # --- CNN-based models ---
        if self.arch_name not in ['vit']:
            feat = self.encoder(x)

            # WideResNet 特殊处理
            if self.arch_name == 'WideResNet':
                feat = torch.nn.functional.avg_pool2d(feat, 8)
                feat = feat.view(-1, self.nChannels)

            # ✅ 其余 CNN 模型（如 ResNet），做自适应池化到 (1, 1)
            elif feat.dim() == 4:
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = torch.flatten(feat, 1)  # (B, C)

            logits = self.fc(feat)

        # --- ViT models ---
        else:
            feat = self.model.forward_features(x)
            feat = self.model.fc_norm(feat[:, 0])
            logits = self.model.head(feat)

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dim(self):
        return self._output_dim
