import torch
import os
import glob
import numpy as np
from typing import Tuple, Optional, List, Union, Generator
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2
from utils.prophesee_toolbox.src.io.psee_loader import PSEELoader
import model as md
from utils.plotter import Plotter
import cv2
from tqdm import tqdm
import torchmetrics.detection


class PropheseeRuntimeDataset(IterableDataset):
    def __init__(
        self,
        dataset: str,
        root: str,
        split: str,
        time_step_us: int = 2000,
        time_shift: int = 0,
        resize: Optional[Union[List[int], int]] = None,
    ):
        super().__init__()
        self.dataset, self.root, self.split = dataset, root, split
        self.time_step_us, self.time_shift = time_step_us, time_shift
        self.resize = resize

        match self.dataset:
            case "gen1":
                self.height = 240
                self.width = 304
            case "1mpx":
                self.height = 720
                self.width = 1280
                self._remap_1mpx()
            case _:
                raise ValueError(f'The dataset parameter cannot be "{self.dataset}"!')

        self._prepare_resize()
        self._get_files_list()
        self._prepare_properties()
        self._preprocess_labels()

    def __iter__(self):
        return iter(self.samples_generator())

    def samples_generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        for data_file, labels_list in zip(self.data_files, self.labels_list):
            loader = PSEELoader(data_file)
            while not loader.done:
                start_time = loader.current_time
                events = self._load_events(loader)
                events = events if self.resize is None else self.transform(events)
                end_time = loader.current_time
                labels = self._load_labels(labels_list, start_time, end_time)
                yield events, labels
        raise StopIteration

    def _load_labels(self, labels_list: List[torch.Tensor], start_time: int, end_time: int):
        mask = (labels_list[:, 0] > start_time) & (labels_list[:, 0] <= end_time)
        return labels_list[mask, 1:]

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
        self.approx_len = (60_000_000 // self.time_step_us) * len(self.data_files)
        
    def __len__(self):
        return self.approx_len

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
            self.labels_list.append(torch.from_numpy(labels))

    def _labels_prepare(self, labels: np.ndarray) -> np.ndarray:
        """
        :param labels: Labels in numpy format
            ('ts [us]', 'x', 'y', 'w', 'h', 'class_id', 'confidence', 'track_id')
        :type labels: np.ndarray
        :return: Labels in format (ts [ms], class id, xlu, ylu, xrd, yrd)
        :rtype: np.ndarray
        """
        remapped_cls = (
            self.remap_lut[labels[:]["class_id"]]
            if self.dataset == "1mpx"
            else labels[:]["class_id"]
        )
        prep_labels = np.array(
            [
                labels[:][self.ts_name],
                remapped_cls,
                labels[:]["x"] / self.target_width,
                labels[:]["y"] / self.target_height,
                (labels[:]["x"] + labels[:]["w"]) / self.target_width,
                (labels[:]["y"] + labels[:]["h"]) / self.target_height,
            ],
            dtype=np.float32,
        ).T
        return prep_labels[prep_labels[:, 1] >= 0]

    def _load_events(self, loader: PSEELoader) -> torch.Tensor:
        events_list = loader.load_delta_t(self.time_step_us)

        # For some reason in 1mpx there are events that go beyond the frame boundaries
        events_list[:]["x"] = events_list[:]["x"].clip(0, self.width - 1)

        # Return events format (ts, c [0-negative, 1-positive], h, w)
        events_out = torch.zeros(
            [2, self.target_height, self.target_width],
            dtype=torch.float32,
        )

        events_out[
            events_list[:]["p"].astype(np.uint32),
            events_list[:]["y"].astype(np.uint32),
            events_list[:]["x"].astype(np.uint32),
        ] = 1

        return events_out

    def _remap_1mpx(self):
        remap = {
            "pedestrians": "person",
            "two wheelers": "person",
            "cars": "car",
            "trucks": "car",
            "buses": "car",
            "signs": None,
            "traffic lights": None,
        }
        target = ("car", "person")
        self.remap_lut = np.arange(len(remap.keys()))
        for key, value in zip(remap.keys(), remap.values()):
            self.remap_lut[list(remap.keys()).index(key)] = (
                target.index(value) if value is not None else -1
            )


def map_estimate(
    metric: torchmetrics.detection.MeanAveragePrecision,
    preds: torch.Tensor,
    labels: torch.Tensor,
):
    map_preds = []
    map_target = []
    map_preds.append(
        {
            "boxes": preds[:, 2:],
            "scores": preds[:, 1],
            "labels": preds[:, 0].type(torch.IntTensor),
        }
    )
    map_target.append(
        {
            "boxes": labels[:, 1:],
            "labels": labels[:, 0].type(torch.IntTensor),
        }
    )
    metric.update(map_preds, map_target)


if __name__ == "__main__":
    model = md.Yolo.load_from_checkpoint(
        ".neptune/yolo8m_gen1/SODA-442/checkpoints/epoch=2-step=25002.ckpt"
    )
    model.to("cuda")
    model.eval()

    dataset = PropheseeRuntimeDataset(
        "gen1", "/media/ubuntu/neuro/datasets/data", "test", 8000, 0, [256, 320]
    )
    dataloader = DataLoader(dataset)
    dataloader_iter = iter(dataloader)

    plotter = Plotter(0.9)

    map_metric = torchmetrics.detection.MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", backend="faster_coco_eval"
    )

    state = None
    try:
        for events, labels in tqdm(dataloader_iter, total=len(dataset)):
            events = events[0].to("cuda")
            labels = labels[0].to("cuda")
            with torch.no_grad():
                preds, state = model.predict(events, state)
            if labels.numel():
                map_estimate(map_metric, preds, labels)
            """ 
            img = plotter.apply(plotter.preprocess_events(events), preds, labels)
            cv2.imshow("Res", img)
            if cv2.waitKey(1) == ord("q"):
                break """
    except KeyboardInterrupt:
        print("Stop running!")
    result = map_metric.compute()
    print(result)
