import bioio
import itertools
import torch
import torchvision
import pandas as pd
import numpy as np
import skimage
import concurrent.futures

from deep_phase.utils import data_operations

def RandomNoise(sigma):
    def gauss_noise_tensor(img):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        
        out = img + sigma * torch.randn_like(img)
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out
    return gauss_noise_tensor


class CellImageDataset(torch.utils.data.Dataset):
    """Given pandas dataframe with coordinates,
    paths, and classes, produce dataset from all rows."""

    def __init__(
            self,
            dataframe: pd.DataFrame,
            rgb_map: np.array,
            crop_size: int,
            categories,
            rotation,
            noise,
            augmentation='randaug',
            device=torch.device('cpu'),
            series_as_time=False,
        ):

        self.df = self._validate_dataframe(dataframe)
        self.device = device

        if isinstance(categories, dict):
            cats = list(itertools.chain(*categories.values()))
            sub_to_super = {}
            keys = categories.keys()
            for superclass, subclasses in categories.items():
                for subclass in subclasses:
                    sub_to_super[subclass] = superclass
        else:
            cats = categories
            keys = categories
            sub_to_super = {c: c for c in cats}

        label_raw = pd.Categorical(
            self.df['category'],
            categories=cats,
        )

        self.sub_label_map = label_raw.categories
        self.sub_label = torch.from_numpy(
            label_raw.codes.astype("long")
        ).to(device)

        self.df['superclass'] = self.df['category'].map(sub_to_super)
        super_label_raw = pd.Categorical(
            self.df['superclass'],
            categories=list(keys),
        )

        self.sup_label_map = super_label_raw.categories
        self.sup_label = torch.from_numpy(
            super_label_raw.codes.astype("long")
        ).to(device)

        self.images = self._build_dataset(rgb_map, crop_size, series_as_time)

        if augmentation == 'randaug':
            self.transform_train = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(224),
                    torchvision.transforms.RandAugment(),
                    torchvision.transforms.ConvertImageDtype(torch.float32),
                    torchvision.transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

        else:
            transforms = [
                torchvision.transforms.ConvertImageDtype(torch.float32),
            ]
            if rotation:
                transforms += [
                    torchvision.transforms.RandomRotation(rotation),
                ]
            if noise:
                transforms += [
                    RandomNoise(noise),
                ]

            transforms += [
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]

            self.transform_train = torchvision.transforms.Compose(transforms)

        self.transform_eval = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ConvertImageDtype(torch.float32),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self.setting = 'train'

    @staticmethod
    def from_config(config_yaml, dataframe=None):
        config = data_operations.parse_log(config_yaml)
        rgb_map = data_operations.get_rgb_map(config['rgb_map'], config['channels'])

        if not dataframe:
            dataframe = pd.read_csv(config['eval_name'])

        result = CellImageDataset(
            dataframe,
            rgb_map,
            crop_size=config['crop_size'],
            categories=config['training_classes'],
            rotation=config['rotation'],
            noise=config['noise'],
            augmentation=config['augmentation'],
        )

        result.eval()
        return result

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        transform = self.transform_eval
        if self.setting == 'train':
            transform = self.transform_train
        return (transform(self.images[idx].to(self.device)),
                self.sub_label[idx],
                self.sup_label[idx])

    def get_image(self, idx):
        transform = torchvision.transforms.CenterCrop(224)
        return np.moveaxis(transform(self.images[idx]).numpy(), 0, -1)

    def train(self):
        self.setting = 'train'

    def eval(self):
        self.setting = 'eval'

    def get_cat_name(self, category):
        return self.sub_label_map[category]

    def _validate_dataframe(self, dataframe):
        # check for expected columns
        expected = set([
                           'local_path',
                           'category',
                           'series',
                           'center_x',
                           'center_y',
                       ])

        assert len(expected) == len(expected & set(dataframe.columns)), (
            "Input dataframe is missing an expected column, "
            f"expected values are {expected}"
        )

        return dataframe.sort_values(by=["local_path", "series"]).reset_index(drop=True)

    def _build_dataset(self, rgb_map, crop_size, series_as_time):
        # keep data on cpu, move to gpu when needed
        device = torch.device('cpu')
        result = torch.empty(
            (len(self.df), 3, 256, 256),
            dtype=torch.float32,
            device=device, requires_grad=False)
        rgb_map = torch.as_tensor(rgb_map, device=device, dtype=torch.float32)

        # create a relative grid for slicing (from -crop_size to crop_size-1)
        y_offset, x_offset = np.indices((2*crop_size, 2*crop_size)) - crop_size

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    _load_image, dat_index, dat,
                    series_as_time, device, x_offset,
                    y_offset, rgb_map)
                for dat_index, dat in self.df.groupby(["local_path", "series"])
            ]

            # Collect results as threads complete
            for future in concurrent.futures.as_completed(futures):
                idx, stack = future.result()
                result[idx] = stack

        return (result * 255).to(torch.uint8)


def _load_image(dat_index, dat, series_as_time, device, x_offset, y_offset, rgb_map):
    path, series = dat_index
    resize = torchvision.transforms.Resize(256, antialias=True)
    bio_img = bioio.BioImage(path)
    if series_as_time:
        print(series)
        image = torch.as_tensor(
            bio_img.get_image_data('CZYX', T=int(series)).squeeze().astype(np.float32),
            device=device,
        )
    else:
        bio_img.set_scene(int(series))
        image = torch.as_tensor(
            bio_img.get_image_data().squeeze().astype(np.float32),
            device=device,
        )

    if image.ndim == 2:
        image = torch.unsqueeze(image, 0)
    channels = image.shape[0]

    dat_np = dat[['center_x', 'center_y']].to_numpy()
    x_centers = dat_np[:, 0]
    y_centers = dat_np[:, 1]

    # broadcast to create absolute coordinates for all crops
    y_indices = y_centers[:, None, None] + y_offset[None, :, :]
    x_indices = x_centers[:, None, None] + x_offset[None, :, :]

    img_stack = image[:, y_indices, x_indices].type(torch.float32)
    img_stack = img_stack.to(device).permute(1, 0, 2, 3)

    min_vals = img_stack.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_vals = img_stack.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-6  # handle empty images
    img_stack = (img_stack - min_vals) / ranges

    # matrix multiply the rgb_map with image stack to convert channels to 3
    # n x c: rbg_map new and old channel numbers
    # bcwh: batch, old channel, width, height
    # bnwh: batch, new chanenl, width, height
    img_stack = torch.einsum("nc, bcwh -> bnwh", rgb_map, img_stack)
    img_stack = resize(img_stack)
    return dat.index, img_stack
