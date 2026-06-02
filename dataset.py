import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import binary_dilation
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm


class RescueNetDataset(Dataset):
    """
    Dataset class for RescueNet semantic segmentation.

    Each sample returns:
    - image: normalized RGB tensor [3, 384, 384]
    - classification: class labels [384, 384]
    - boundary: edge map derived from mask [1, 96, 96]
    """

    CLASS_NAMES = {
        0: 'Background',
        1: 'Water',
        2: 'Building_No_Damage',
        3: 'Building_Minor_Damage',
        4: 'Building_Major_Damage',
        5: 'Building_Total_Destruction',
        6: 'Vehicle',
        7: 'Road-Clear',
        8: 'Road-Blocked',
        9: 'Tree',
        10: 'Pool'
    }

    NUM_CLASSES = 11

    def __init__(
        self,
        root,
        split='train',
        image_size=(512, 512),
        augment=True,
        dilation_radius=2,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size

        # only apply augmentation during training
        self.augment = augment and (split == 'train')

        self.dilation_radius = dilation_radius

        # dataset folder structure
        self.img_dir = os.path.join(root, split, f'{split}-org-img')
        self.mask_dir = os.path.join(root, split, f'{split}-label-img')

        # list all files
        self.img_files = sorted(os.listdir(self.img_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        # basic sanity check
        assert len(self.img_files) == len(self.mask_files), \
            f"Mismatch: {len(self.img_files)} images but {len(self.mask_files)} masks"

        print(f"[RescueNetDataset] split={split:5s} | "
              f"samples={len(self.img_files)} | "
              f"augment={self.augment}")

    def __len__(self):
        return len(self.img_files)

    def _load(self, idx):
        # read image and mask from disk
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')

        mask = Image.open(mask_path)
        if mask.mode != 'P':
            # ensure mask is in a single-channel format
            mask = mask.convert('L')

        # resize both to target size
        image = image.resize(self.image_size, Image.LANCZOS)
        mask = mask.resize(self.image_size, Image.NEAREST)

        return image, mask

    def _random_crop(self, image, mask, crop_scale=(0.5, 1.0)):
        # randomly crop a region and resize back
        w, h = image.size
        scale = random.uniform(*crop_scale)

        crop_w = int(w * scale)
        crop_h = int(h * scale)

        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)

        image = image.crop((x, y, x + crop_w, y + crop_h))
        mask = mask.crop((x, y, x + crop_w, y + crop_h))

        image = image.resize((w, h), Image.LANCZOS)
        mask = mask.resize((w, h), Image.NEAREST)

        return image, mask

    def _augment(self, image, mask):
        # start with random crop
        image, mask = self._random_crop(image, mask)

        # horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # vertical flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # random rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            rotation = {
                1: Image.ROTATE_90,
                2: Image.ROTATE_180,
                3: Image.ROTATE_270
            }[k]
            image = image.transpose(rotation)
            mask = mask.transpose(rotation)

        # slight blur to simulate real-world capture
        if random.random() > 0.4:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # occasionally remove color (forces reliance on structure)
        if random.random() > 0.85:
            image = ImageEnhance.Color(image).enhance(0.0)

        # sharpen edges a bit
        if random.random() > 0.5:
            image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 2.0))

        # brightness adjustment
        if random.random() > 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))

        # contrast adjustment
        if random.random() > 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))

        # color jitter
        if random.random() > 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))

        return image, mask

    def _image_to_tensor(self, image):
        # convert PIL image to normalized tensor
        img_np = np.array(image)
        img_tensor = torch.from_numpy(img_np)

        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.float() / 255.0

        # ImageNet normalization (helps convergence even when training from scratch)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img_tensor = (img_tensor - mean) / std
        return img_tensor

    def _mask_to_classification(self, mask):
        # convert mask image to class index tensor
        mask_np = np.array(mask)
        return torch.from_numpy(mask_np).long()

    def _classification_to_boundary(self, classification_tensor):
        # downsample mask to match boundary resolution
        mask_small = classification_tensor.unsqueeze(0).unsqueeze(0).float()
        mask_small = F.interpolate(mask_small, scale_factor=0.25, mode='nearest')

        mask_np = mask_small.squeeze().numpy().astype(np.int32)

        # detect edges by checking neighboring differences
        diff_x = np.pad(np.abs(np.diff(mask_np, axis=1)), ((0, 0), (0, 1)), mode='constant')
        diff_y = np.pad(np.abs(np.diff(mask_np, axis=0)), ((0, 1), (0, 0)), mode='constant')

        boundary = ((diff_x > 0) | (diff_y > 0)).astype(np.float32)

        # optionally thicken boundaries
        r = self.dilation_radius // 4
        if r > 0:
            structure = np.ones((r * 2 + 1, r * 2 + 1), dtype=bool)
            boundary = binary_dilation(boundary, structure=structure).astype(np.float32)

        return torch.from_numpy(boundary).unsqueeze(0)

    def __getitem__(self, idx):
        # load data
        image, mask = self._load(idx)

        # apply augmentation if training
        if self.augment:
            image, mask = self._augment(image, mask)

        # convert to tensors
        image_tensor = self._image_to_tensor(image)
        classification_tensor = self._mask_to_classification(mask)

        # derive boundary map from mask
        boundary_tensor = self._classification_to_boundary(classification_tensor)

        return {
            'image': image_tensor,
            'classification': classification_tensor,
            'boundary': boundary_tensor,
            'filename': self.img_files[idx],
        }


def compute_sample_weights_fast(dataset, device):
    """
    Assign higher weights to samples that contain important classes.
    Helps the model focus more on rare or critical categories.
    """

    weights = []
    class_458 = torch.tensor([4, 5, 8], device=device)

    for mask_file in tqdm(dataset.mask_files, desc="Computing sample weights"):
        mask_path = os.path.join(dataset.mask_dir, mask_file)

        # load mask and move to device
        mask = Image.open(mask_path)
        mask = mask.resize(dataset.image_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).to(device)

        # check presence of important classes
        contains_458 = torch.isin(mask, class_458).any()
        contains_9 = (mask == 9).any()
        contains_6 = (mask == 6).any()
        contains_10 = (mask == 10).any()

        weight = 1.0

        # assign importance-based weights
        if contains_458:
            weight += 4.0
        if contains_9:
            weight += 2.5
        if contains_6:
            weight += 2.0
        if contains_10:
            weight += 1.5

        weights.append(weight)

    return weights