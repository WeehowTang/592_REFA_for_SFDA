import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

# ========== Dataset ==========
class Office31Dataset(Dataset):
    def __init__(self, root_dir, size=None, preprocess=None):
        """
        root_dir: 图片根目录
        size: 如果没有传入 preprocess，则可指定图片大小
        preprocess: CLIP 自带的 transform，可选
        """
        self.imagefolder = ImageFolder(root=root_dir)
        self.classes = self.imagefolder.classes

        if preprocess is not None:
            self.transform = preprocess
        else:

            assert size is not None, "size must be specified if preprocess is None"
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5], [0.5, 0.5])
            ])

    def __len__(self):
        return len(self.imagefolder)

    def __getitem__(self, idx):
        image_path, label_idx = self.imagefolder.imgs[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)  # 使用选择的 transform
        label_name = self.classes[label_idx]
        return image_tensor, label_name, idx
