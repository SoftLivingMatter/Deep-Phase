import bioio
import itertools
import torch
import torchvision
import pandas as pd
import numpy as np
import skimage

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        transform = self.transform_eval
        if self.setting == 'train':
            transform = self.transform_train
        return (transform(self.images[idx].to(self.device)),
                self.sub_label[idx],
                self.sup_label[idx])

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
        starting_ind = 0
        resize = torchvision.transforms.Resize(256, antialias=True)
        rgb_map = torch.as_tensor(rgb_map, device=device, dtype=torch.float32)
        for name, dat in self.df.groupby(["local_path", "series"]):
            path, series = name

            bio_img = bioio.BioImage(path)
            if series_as_time:
                print(series)
                image = torch.as_tensor(
                    bio_img.get_image_data('CZYX', T=int(series)).squeeze().astype(np.int32),
                    device=device,
                )
            else:
                bio_img.set_scene(int(series))
                image = torch.as_tensor(
                    bio_img.get_image_data().squeeze().astype(np.int32),
                    device=device,
                )
            channels = image.shape[0]

            for i, (_, row) in enumerate(
                dat[["center_x", "center_y"]].iterrows(),
                start=starting_ind,
            ):
                x, y = row.to_numpy()
                # image order is TCZYX, e.g. y then x
                if image.ndim == 3:
                    img = image[
                        :,
                        y - crop_size : y + crop_size,
                        x - crop_size : x + crop_size,
                    ].type(torch.float32)
                    # scale min/max
                    min_val = img.view(channels, -1).min(axis=1)[0]
                    max_val = img.view(channels, -1).max(axis=1)[0]
                    img = (
                            (img - min_val[:, None, None])
                            / (max_val[:, None, None] - min_val[:, None, None])
                    )
                    # apply the rgb map by converting image to channels x pixel
                    # matrix, perform matmul, then reshape to 3 x row x col
                    result[i] = resize(
                        (rgb_map @ img.view(channels, -1)).view(3, *img.shape[1:])
                    )
                else:  # single channel image
                    img = image[
                        y - crop_size : y + crop_size,
                        x - crop_size : x + crop_size,
                    ].type(torch.float32)
                    # scale min/max
                    min_val = img.min()
                    max_val = img.max()
                    img = (
                            (img - min_val)
                            / (max_val - min_val)
                    )
                    # apply the rgb map by converting image to channels x pixel
                    # matrix, perform matmul, then reshape to 3 x row x col
                    result[i] = resize(
                        (rgb_map @ img.view(1, -1)).view(3, *img.shape)
                    )

            starting_ind += len(dat)
        return (result * 255).to(torch.uint8)
