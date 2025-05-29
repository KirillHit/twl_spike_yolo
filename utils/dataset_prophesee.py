"""Module for working with Prophesee datasets"""

import os
import glob
import numpy as np
from typing import Tuple, Optional, List, Union
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from torchvision.transforms import v2
from utils.prophesee_toolbox.src.io.psee_loader import PSEELoader


def _collate_data(batch):
    """Combines samples into a batch taking into account the time dimension"""
    events = torch.stack([sample[0] for sample in batch], dim=1)
    targets = pad_sequence(
        [sample[1] for sample in batch],
        batch_first=True,
        padding_value=-1,
    )
    return {"events": events, "targets": targets}


class PropheseeDataModule(L.LightningDataModule):
    """Module for working with GEN1 and 1MPX dataset"""

    def __init__(
        self,
        root: str = "./data",
        dataset: str = "gen1",
        batch_size: int = 4,
        num_workers: int = 1,
        num_steps: int = 42,
        time_step_us: int = 4000,
        resize: Optional[Union[List[int], int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def get_labels(self) -> Tuple[str]:
        """Returns a list of class names"""
        return ("car", "person")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._prepare_dataset("train")
        if stage in ("fit", "validate"):
            self.val_dataset = self._prepare_dataset("val")
        if stage in ("test", "predict"):
            self.test_dataset = self._prepare_dataset("test")

    def _prepare_dataset(self, split: str) -> Dataset:
        return PropheseeDataset(
            dataset=self.hparams.dataset,
            root=self.hparams.root,
            split=split,
            num_steps=self.hparams.num_steps,
            time_step_us=self.hparams.time_step_us,
            resize=self.hparams.resize,
            seed=int(os.environ.get("PL_GLOBAL_SEED")),
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool) -> StatefulDataLoader:
        return StatefulDataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=_collate_data,
            shuffle=shuffle,
        )


class PropheseeDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        root: str,
        split: str,
        num_steps: int = 32,
        time_step_us: int = 2000,
        resize: Optional[Union[List[int], int]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dataset, self.root, self.split = dataset, root, split
        self.num_steps, self.time_step_us = num_steps, time_step_us
        self.resize = resize

        self.height = 240
        self.width = 304

        self._prepare_resize()
        self._get_files_list()
        self._prepare_properties()
        self._preprocess_labels()

        self.index_list = np.arange(self.labels_len)
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.index_list)

    def __len__(self):
        return self.labels_len

    def __getitem__(self, idx: int):
        idx = self.index_list[idx]

        file_idx, label_idx = self._idx_remap(idx)

        start_idx = self.labels_starts_maps[file_idx][label_idx]
        count_labels = self.labels_counts_maps[file_idx][label_idx]
        labels = self.labels_list[file_idx][start_idx : start_idx + count_labels]
        labels_time = int(labels[0, 0].item())
        labels = labels[:, 1:]

        events = self._load_events(self.data_files[file_idx], labels_time)

        events = events if self.resize is None else self.transform(events)

        return events, labels

    def _get_files_list(self) -> None:
        data_dir = os.path.join(self.root, self.dataset, self.split)
        self.labels_files = glob.glob(data_dir + "/*.npy")
        self.data_files = [p.replace("_bbox.npy", "_td.dat") for p in self.labels_files]
        if not len(self.labels_files):
            raise RuntimeError(
                f"Directory '{data_dir}' does not contain data or data is invalid! I'm expecting: "
                f"./data/{self.dataset}/{data_dir}/*_bbox.npy (and *_td.dat). "
                "The datasets can be downloaded from these links: "
                "https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/ or "
                "https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/"
            )

    def _prepare_properties(self) -> None:
        labels = np.load(self.labels_files[0])
        # For some reason, the labels of the datasets gen1 and 1mpx are different
        self.ts_name = list(labels.dtype.fields.keys())[0]

    def _prepare_resize(self) -> None:
        if self.resize is None:
            self.target_width = self.width
            self.target_height = self.height
            return

        self.transform = v2.Resize(
            self.resize,
            interpolation=v2.InterpolationMode.NEAREST,
        )

        target_ratio = self.resize[1] / self.resize[0]
        self.target_width = max(self.width, int(self.height * target_ratio))
        self.target_height = int(self.target_width / target_ratio)

        if self.target_height < self.height:
            self.target_height = max(self.height, int(self.width / target_ratio))
            self.target_width = int(self.target_height * target_ratio)

    def _preprocess_labels(self) -> None:
        self.labels_counts_maps = []
        self.labels_starts_maps = []
        self.labels_list = []
        self.labels_len = 0
        for labels_path in self.labels_files:
            labels = self._labels_prepare(np.load(labels_path))
            labels = labels[labels[:, 0] > (self.time_step_us * self.num_steps)]
            _, labels_index, labels_count = np.unique(
                labels[:, 0], return_index=True, return_counts=True
            )
            self.labels_list.append(torch.from_numpy(labels))
            self.labels_counts_maps.append(labels_count)
            self.labels_starts_maps.append(labels_index)
            self.labels_len += len(labels_count)

    def _labels_prepare(self, labels: np.ndarray) -> np.ndarray:
        """
        :param labels: Labels in numpy format
            ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
        :type labels: np.ndarray
        :return: Labels in format (ts [ms], class id, xlu, ylu, xrd, yrd)
        :rtype: np.ndarray
        """
        prep_labels = np.array(
            [
                labels[:][self.ts_name],
                labels[:]["class_id"],
                labels[:]["x"] / self.target_width,
                labels[:]["y"] / self.target_height,
                (labels[:]["x"] + labels[:]["w"]) / self.target_width,
                (labels[:]["y"] + labels[:]["h"]) / self.target_height,
            ],
            dtype=np.float32,
        ).T
        return prep_labels[prep_labels[:, 1] >= 0]

    def _idx_remap(self, idx: int) -> Tuple[int, int]:
        file_idx = 0
        for counts in self.labels_counts_maps:
            file_len = len(counts)
            if idx < file_len:
                break
            idx -= file_len
            file_idx += 1
        return file_idx, idx

    def _load_events(self, file: str, labels_time: int) -> torch.Tensor:
        loader = PSEELoader(file)

        start_time = labels_time - self.num_steps * self.time_step_us
        if start_time > 0:
            loader.seek_time(start_time)
        events_list = loader.load_delta_t(self.num_steps * self.time_step_us)

        time_stamps = (events_list[:]["t"] - start_time) // self.time_step_us

        # For some reason in 1mpx there are events that go beyond the frame boundaries
        events_list[:]["x"] = events_list[:]["x"].clip(0, self.width - 1)
        time_stamps = time_stamps.clip(0, self.num_steps - 1)

        # Return events format (ts, c [0-negative, 1-positive], h, w)
        events_out = torch.zeros(
            [self.num_steps, 2, self.target_height, self.target_width],
            dtype=torch.float32,
        )

        events_out[
            time_stamps[:],
            events_list[:]["p"].astype(np.uint32),
            events_list[:]["y"].astype(np.uint32),
            events_list[:]["x"].astype(np.uint32),
        ] = 1

        return events_out
