"""Module for working with DSEC dataset"""

import lightning as L
from dsec_det.dataset import DSECDet
from dsec_det.label import CLASSES
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
import yaml
import numpy as np
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Union, Dict, Tuple, Generator
from torchvision.transforms.functional import adjust_gamma
from model.modules.current_encoder import ConstantCurrentLIFEncoderCell
import torch.distributed as dist
import os


def _collate_fn(batch):
    """Combines samples into a batch taking into account the time dimension"""
    events = torch.stack([sample[0] for sample in batch], dim=1)
    images = torch.stack([sample[2] for sample in batch])
    src_images = torch.stack([sample[3] for sample in batch])
    pad_val = max([sample[1].shape[1] for sample in batch])
    targets = torch.full((batch[0][1].shape[0], len(batch), pad_val, 5), -1, dtype=torch.float32)
    for idx, sample in enumerate(batch):
        trg = sample[1]
        targets[:, idx, : trg.shape[1]] = trg
    return {"events": events, "targets": targets, "images": images, "src_image": src_images}


class DSECDataModule(L.LightningDataModule):
    """Module for working with DSEC dataset"""

    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 4,
        num_workers: int = 4,
        time_step_us: int = 1000,
        iter: bool = False,
        encode: bool = False,
        resize: Optional[Union[int, List[int]]] = None,
    ):
        """
        :param root: Root directory where the DSEC dataset is stored.
        :type root: str
        :param batch_size: Number of samples per batch to load.
        :type batch_size: int
        :param num_workers: Number of subprocesses to use for data loading.
        :type num_workers: int
        :param time_step_us: Duration of each time step in microseconds.
        :type time_step_us: int
        :param iter: If True, use iterable dataset for distributed/multi-worker loading.
        :type iter: bool
        :param encode: If True, apply current-based encoding to images.
        :type encode: bool
        :param resize: Target size for resizing input data (height, width) or None for no resizing.
        :type resize: Optional[Union[int, List[int]]]
        """
        super().__init__()
        self.save_hyperparameters()
        self.remap = {
            "pedestrian": "pedestrian",
            "rider": None,
            "car": "car",
            "bus": "car",
            "truck": "car",
            "bicycle": None,
            "motorcycle": None,
            "train": None,
        }

    def get_labels(self) -> List[str]:
        return list(filter(lambda x: x is not None, self.remap.values()))

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._get_dataset("train")
        if stage in ("fit", "validate"):
            self.val_dataset = self._get_dataset("val")
        if stage in ("test", "predict"):
            self.test_dataset = self._get_dataset("test")

    def _get_dataset(self, split: str):
        seed = (
            int(os.environ.get("PL_GLOBAL_SEED"))
            if os.environ.get("PL_GLOBAL_SEED") is not None
            else None
        )
        dataset = DSECDataset(
            self.hparams.root,
            split,
            time_step_us=self.hparams.time_step_us,
            resize=self.hparams.resize,
            remap=self.remap,
            seed=seed,
            encode=self.hparams.encode,
        )
        if self.hparams.iter:
            return DSECDatasetIter(dataset)
        return dataset

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, False)

    def predict_dataloader(self):
        return self._get_dataloader(self.test_dataset, False)

    def _get_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=_collate_fn,
            shuffle=None if self.hparams.iter else shuffle,
        )


class ImageGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], requires_grad=False
        )
        self.sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]], requires_grad=False
        )

    def forward(self, img: torch.Tensor):
        c = img.size(0)
        img = img.unsqueeze(0)
        grad_x = F.conv2d(img, self.sobel_x.expand(c, 1, -1, -1), padding=1, groups=c).squeeze(0)
        grad_y = F.conv2d(img, self.sobel_y.expand(c, 1, -1, -1), padding=1, groups=c).squeeze(0)
        return torch.sqrt(grad_x**2 + grad_y**2)


class Gamma(nn.Module):
    def __init__(self, gamma: float, gain: float = 1):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("gain", torch.tensor(gain))

    def forward(self, img: torch.Tensor):
        return adjust_gamma(img, self.gamma, self.gain)


class CurrentEncode(nn.Module):
    def __init__(self, steps: int):
        super().__init__()
        self.steps = steps
        self.encoder = ConstantCurrentLIFEncoderCell()

    def forward(self, input: torch.Tensor):
        c, h, w = input.shape
        events = torch.zeros((self.steps, c, h, w), dtype=torch.float32, device=input.device)
        with torch.no_grad():
            state = None
            for idx in range(self.steps):
                events[idx], state = self.encoder(input, state)
        return events


class DSECDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        time_step_us: int = 1000,
        resize: Optional[Union[int, List[int]]] = None,
        remap: Optional[Dict[str, Optional[str]]] = None,
        seed: Optional[int] = None,
        encode: bool = False,
    ):
        super().__init__()
        with open("utils/dsec_split.yaml", "r") as file:
            split_config = yaml.safe_load(file)
        self.dataset = DSECDet(
            Path(root),
            split,
            sync="back",
            debug=False,
            split_config=split_config,
            interpolate_labels=True,
            interp_step_us=time_step_us,
        )

        self.time_step_us = time_step_us
        self.duration = 50000
        self.num_steps = self.duration // self.time_step_us

        self.resize = resize
        self._prepare_resize()

        self.base_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(self.img_resize, max_size=self.dataset.width if resize is None else None),
            ]
        )

        self.encode = encode
        if self.encode:
            self.image_transforms = v2.Compose(
                [
                    v2.Grayscale(),
                    Gamma(0.6),
                    ImageGradient(),
                    v2.Normalize(mean=[0.05], std=[0.4]),
                    CurrentEncode(self.num_steps),
                ]
            )
        else:
            self.image_transforms = v2.Normalize(mean=[0.25, 0.24, 0.23], std=[0.235, 0.215, 0.20])

        self._remap_to_lut(remap)

        self.index_list = np.arange(len(self.dataset))
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.index_list)

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
        idx = self.index_list[idx]

        data = self.dataset[idx]

        events = self._preprocess_events(data["events"])

        pad_img = torch.zeros([3, events.size(-2), events.size(-1)], dtype=torch.float32)
        src_img = self.base_transform(data["image"])
        pad_img[:, 0 : src_img.size(-2), 0 : src_img.size(-1)] = src_img
        prep_img = self.image_transforms(pad_img)

        if self.encode:
            events = torch.cat((events, prep_img), dim=1)

        tracks = self._preprocess_detections(data["tracks"])

        return events, tracks, prep_img, pad_img

    def __len__(self):
        return len(self.dataset)

    def _preprocess_events(self, src: np.ndarray) -> torch.Tensor:
        events = torch.zeros(
            [self.num_steps, 2, self.target_height, self.target_width],
            dtype=torch.float32,
        )
        time_stamps = (src["t"] - src["t"][0]) // self.time_step_us
        time_stamps = time_stamps.clip(0, self.num_steps - 1)
        events[
            time_stamps,
            src["p"].astype(np.uint16),
            src["y"],
            src["x"],
        ] = 1
        return self.events_resize(events) if self.resize is not None else events

    def _preprocess_detections(self, labels: np.ndarray):
        """
        Input labels format:
        ('t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence', 'track_id')
        Output labels format:
        ('t', 'class_id', lux, luy, rdx, rdy)
        """
        if not labels.size:
            return torch.full((self.num_steps, 1, 5), -1, dtype=torch.float32)

        ts_list = np.unique(labels[:]["t"])

        classes = self.remap_lut[labels[:]["class_id"]]
        classes[(labels[:]["w"] ** 2 + labels[:]["h"] ** 2) < (40**2)] = -1
        mask = classes >= 0
        classes = classes[mask]
        labels = labels[mask]

        if not labels.size:
            return torch.full((self.num_steps, 1, 5), -1, dtype=torch.float32)

        labels[:]["class_id"] = classes

        _, counts = np.unique(labels[:]["t"], return_counts=True)
        out = np.full((self.num_steps, max(counts), 5), -1, dtype=np.float32)

        for idx, ts in enumerate(ts_list[1:51]):
            ts_labels = labels[labels[:]["t"] == ts]
            out[idx, : ts_labels.shape[0]] = np.array(
                [
                    ts_labels[:]["class_id"],
                    ts_labels[:]["x"] / self.target_width,
                    ts_labels[:]["y"] / self.target_height,
                    (ts_labels[:]["x"] + ts_labels[:]["w"]) / self.target_width,
                    (ts_labels[:]["y"] + ts_labels[:]["h"]) / self.target_height,
                ],
                dtype=np.float32,
            ).T
        return torch.from_numpy(out)

    def _remap_to_lut(self, remap: Optional[Dict[str, Optional[str]]]):
        self.remap_lut = np.arange(len(CLASSES))
        if remap is None:
            self.class_list = CLASSES
            return
        self.class_list = list(filter(lambda x: x is not None, remap.values()))
        for key, value in zip(remap.keys(), remap.values()):
            self.remap_lut[CLASSES.index(key)] = (
                self.class_list.index(value) if value is not None else -1
            )

    def _prepare_resize(self) -> None:
        if self.resize is None:
            self.target_width = self.dataset.width
            self.target_height = self.dataset.height
            self.img_resize = self.resize
            return

        self.events_resize = v2.Resize(
            self.resize,
            interpolation=v2.InterpolationMode.NEAREST,
        )

        target_ratio = self.resize[1] / self.resize[0]
        self.target_width = max(self.dataset.width, int(self.dataset.height * target_ratio))
        self.target_height = int(self.target_width / target_ratio)

        if self.target_height < self.dataset.height:
            self.target_height = max(self.dataset.height, int(self.dataset.width / target_ratio))
            self.target_width = int(self.target_height * target_ratio)

        self.img_resize = (
            int(self.resize[0] * self.dataset.height / self.target_height),
            int(self.resize[1] * self.dataset.width / self.target_width),
        )


class DSECDatasetIter(IterableDataset):
    def __init__(self, dataset: DSECDataset):
        super().__init__()
        self.dataset = dataset
        self.index_list = np.arange(len(self.dataset))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        total_workers = worker_info.num_workers
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id
        len_per_worker = len(self.index_list) // total_workers
        sequence_start = len_per_worker * global_worker_id
        indexes = self.index_list[sequence_start : sequence_start + len_per_worker]
        return iter(self.samples_generator(indexes))

    def samples_generator(
        self, indexes: np.ndarray
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        while True:
            for index in indexes:
                batch = self.dataset[index]
                if not (batch[1][:, :, 0] >= 0).count_nonzero():
                    continue
                yield batch
