"""
Basic object detector class
"""

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from lightning.pytorch.loggers import NeptuneLogger
import torchmetrics.detection
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
from typing import Tuple, Optional, List, Dict, Any
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from utils.roi import RoI
from utils.plotter import Plotter
import utils.box as box
import model.tools.generator as gen


class Detector(L.LightningModule):
    """Basic object detector class

    Implements the basic functions for calculating losses,
    training the network and generating predictions. The network model is passed
    as a parameter when initializing.

    .. warning::
        This class can only be used as a base class for inheritance.
    """

    def __init__(
        self,
        num_classes: int,
        loss_ratio: Optional[float] = 0.04,
        time_window: int = 16,
        iou_threshold: float = 0.4,
        learning_rate: float = 0.001,
        clip_grad: int = -1,
        dt: float = 0.001,
        state_storage: bool = False,
        init_weights: bool = True,
        in_channels: int = 2,
        sizes: List[List[int]] = [[20, 30, 40], [60, 90, 120], [150, 200, 250]],
        aspect_ratios: List[List[float]] = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]],
        load_model: str = "",
        plotter: Optional[Plotter] = None,
    ):
        """
        :param num_classes: Number of classes.
        :type num_classes: int
        :param loss_ratio: The ratio of the loss for non-detection to the loss for false positives.
            The higher this parameter, the more guesses the network generates.
            This is necessary to keep the network active. Defaults to 0.04.
        :type loss_ratio: int, optional
        :param time_window: The size of the time window at the beginning of the sequence,
            which can be truncated to a random length. This ensures randomization of the length of
            training sequences and the ability of the network to work with streaming information.
            Defaults to 0.
        :type time_window: int, optional
        :param iou_threshold: Minimum acceptable iou. Defaults to 0.4.
        :type iou_threshold: float, optional
        :param learning_rate: Learning rate. Defaults to 0.001.
        :type learning_rate: float, optional
        :param clip_grad: Number of frames at the end of the sequence used to take the gradient.
            If less than zero, the gradient is taken over the entire sequence. Defaults to -1.
        :type clip_grad: int, optional
        :param dt: Time step to use in integration. Defaults to 0.001.
        :type dt: float, optional
        :param state_storage: If true preserves preserves all intermediate states of spiking neurons.
            Necessary for analyzing the network operation. Defaults to False.
        :type state_storage: bool, optional
        :param init_weights: If true, apply the weight initialization function. Defaults to True.
        :type init_weights: bool, optional
        :param in_channels: Number of input channels
        :type in_channels: int, optional
        :param sizes: List of anchor box sizes. Must contain three nested lists, each containing any
            number of sizes. The first list corresponds to the feature map with the highest resolution.
        :type sizes: List[List[int]], optional
        :param aspect_ratios: List of aspect ratios of anchor boxes. ratio = height / width.
            Same format as sizes.
        :type aspect_ratios: List[List[float]], optional
        :param load_model: The name of the model whose weights will be loaded.
        :type load_model: str, optional
        :param plotter: Class for displaying results. Needed for the prediction step.
            Expects to receive a utils.Plotter object. Defaults to None.
        :type plotter: Plotter, optional
        """
        super().__init__()
        self.save_hyperparameters(ignore=["plotter"])
        self.plotter = plotter

        self.anchor_generator = AnchorGenerator(self.hparams.sizes, self.hparams.aspect_ratios)
        self.num_anchors = self.anchor_generator.num_anchors_per_location()
        self.num_class_out = [n * (self.hparams.num_classes + 1) for n in self.num_anchors]
        self.num_box_out = [n * 4 for n in self.num_anchors]

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.L1Loss(reduction="none")

        self.roi_blk = RoI(self.hparams.iou_threshold)
        self.map_metric = torchmetrics.detection.MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", backend="faster_coco_eval"
        )

        # Storages for network results. Сhild classes should use these to store their outputs.
        self.storage_box = gen.Storage(auto_reset=False)
        self.storage_cls = gen.Storage(auto_reset=False)
        self.storage_feature = gen.Storage(auto_reset=False)

        # Models for handling static images and event data. Should be defined in child classes.
        self.image_model: gen.ModelGenerator | None = None
        self.net: gen.ModelGenerator | None = None

    def load(self):
        """Loads model weights from HuggingFace Hub if a path is provided"""
        if not self.hparams.load_model:
            return
        path = hf_hub_download("KirillHit/twl_spike_yolo", self.hparams.load_model + ".safetensors")
        self.load_state_dict(sf.load_file(path))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max", patience=5),
                "monitor": "map_50",
                "interval": "step",
                "frequency": self.trainer.val_check_interval,
                "strict": False,
            },
        }

    def events_forward(
        self, events: torch.Tensor, state: gen.ListState | None = None
    ) -> gen.ListState | None:
        """Processes a single time step of event data"""
        self.storage_box.reset()
        self.storage_cls.reset()
        _, state = self.net(events, state)
        self._create_anchors(events)
        return state

    def image_forward(self, image: torch.Tensor) -> None:
        """Processes a static image"""
        self.storage_box.reset()
        self.storage_cls.reset()
        self.image_model(image)
        self._create_anchors(image)

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Main forward method

        Handles both image and event data, processes the batch, and returns predictions.
        """
        if "images" in batch and self.image_model is not None:
            self.image_forward(batch["images"])

        if self.net is not None:
            events = batch["events"]
            state = None
            duration = events.shape[0]
            grad_time = duration - self.hparams.clip_grad if self.hparams.clip_grad >= 0 else 0
            for idx, ts in enumerate(events):
                # Enable gradients only for the last frames if clip_grad is set
                with torch.set_grad_enabled(self.training and (idx >= grad_time)):
                    state = self.events_forward(ts, state)

        return self._get_predictions()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._prepare_batch(batch)
        preds = self.forward(batch)
        loss = self._loss(preds, batch["targets"])
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=self.trainer.datamodule.hparams.batch_size
        )
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._prepare_batch(batch)
        preds = self.forward(batch)
        loss = self._loss(preds, batch["targets"])
        self.log("test_loss", loss, batch_size=self.trainer.datamodule.hparams.batch_size)
        self._map_estimate(preds, batch["targets"])
        return loss

    def on_test_epoch_end(self):
        self._map_compute()

    def on_validation_epoch_start(self):
        self.plotter.labels = self.trainer.datamodule.get_labels()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._prepare_batch(batch)
        preds = self.forward(batch)
        loss = self._loss(preds, batch["targets"])
        self.log("val_loss", loss, batch_size=self.trainer.datamodule.hparams.batch_size)
        self._map_estimate(preds, batch["targets"])
        if not batch_idx:
            background = batch.get("src_image", batch.get("images", None))
            img = self._plot_static(batch["events"], background, self.predict(), batch["targets"])
            if isinstance(self.logger, NeptuneLogger):
                self.logger.experiment["training/images"].append(
                    to_pil_image(
                        img,
                    ),
                    description=f"Step: {self.global_step}",
                )
            else:
                self.logger.experiment.add_image("images", img, self.global_step)
        return loss

    def on_validation_epoch_end(self):
        self._map_compute()

    def on_predict_epoch_start(self):
        self.plotter.labels = self.trainer.datamodule.get_labels()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        batch = self._prepare_batch(batch)
        if "images" in batch and self.image_model is not None:
            self.image_forward(batch["images"])

        background = batch.get("src_image", batch.get("images", None))

        if self.net is not None:
            preds = []
            state = None
            for ts in batch["events"]:
                state = self.events_forward(ts, state)
                preds.append(self.predict())
            preds = list(zip(*preds))
            batch_size = batch["events"].size(1)

            for idx in range(batch_size):
                back = background[idx] if background is not None else None
                video = self._plot_events(
                    batch["events"][:, idx], back, preds[idx], batch["targets"][idx]
                )
                self.plotter(video, self.trainer.datamodule.hparams.time_step_us // 1000)
        else:
            preds = self.predict()
            res = self._plot_static(None, background, preds, batch["targets"])
            numpy_image = (res.numpy()).astype(np.uint8)
            cv2_image = np.transpose(numpy_image, (1, 2, 0))
            self.plotter([cv2_image], 0)

    def predict(self) -> torch.Tensor:
        """Returns processed predictions from storage, applying NMS"""
        if not len(self.storage_cls.storage):
            return torch.empty(0)
        anchors, cls, bbox = self._get_predictions()
        preds = box.multibox_detection(F.softmax(cls, dim=2), bbox, anchors).cpu()
        prep_pred = [preds_b[preds_b[:, 0] >= 0] for preds_b in preds]
        return prep_pred

    def _plot_events(
        self,
        events: torch.Tensor,
        background: Optional[torch.Tensor],
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> List[np.ndarray]:
        """Creates a video visualization for event-based predictions"""
        video = []
        for ts_idx, ts in enumerate(events):
            out_preds = (
                preds[ts_idx] if ts_idx >= (events.size(0) - self.hparams.time_window) else None
            )
            img = self.plotter.preprocess(ts, background)
            video.append(self.plotter.apply(img, out_preds, None))
        video.append(self.plotter.apply(img, preds[-1], labels))
        return video

    def _plot_static(
        self,
        events: Optional[torch.Tensor],
        background: Optional[torch.Tensor],
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Creates a static image visualization for predictions"""
        batch_size = labels.size(0) if events is None else events.size(1)
        out = []
        for idx in range(batch_size):
            img = (
                self.plotter.preprocess_image(background[idx])
                if events is None
                else self.plotter.preprocess(events[-1, idx], background[idx])
            )
            out.append(
                torch.from_numpy(self.plotter.apply(img, preds[idx], labels[idx]))
                .permute((2, 0, 1))
                .flip(0)
            )
        res_events = make_grid(out, pad_value=255)
        return res_events

    def _get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collects and reshapes class and box predictions from storage"""
        cls_preds = self.storage_cls.get()
        bbox_preds = self.storage_box.get()

        cls_preds = self._concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.hparams.num_classes + 1)
        bbox_preds = self._concat_preds(bbox_preds)
        bbox_preds = bbox_preds.reshape(bbox_preds.shape[0], -1, 4)

        return (self.anchors, cls_preds, bbox_preds)

    def _flatten_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Transforms the tensor so that each pixel retains channels values and smooths each batch"""
        return torch.flatten(torch.permute(pred, (0, 2, 3, 1)), start_dim=1)

    def _concat_preds(self, preds: List[torch.Tensor]) -> torch.Tensor:
        """Concatenating Predictions for Multiple Scales"""
        return torch.cat([self._flatten_pred(p) for p in preds], dim=1)

    def _create_anchors(self, X: torch.Tensor) -> None:
        """Creates anchors if not already present, using current feature maps"""
        if hasattr(self, "anchors") or not len(self.storage_feature.storage):
            return
        features_map = self.storage_feature.get()
        h, w = X.shape[-2:]
        self.anchors = self.anchor_generator(ImageList(X, [X.shape[-2:]]), features_map)[0]
        self.anchors[:, 0] = self.anchors[:, 0] / w
        self.anchors[:, 1] = self.anchors[:, 1] / h
        self.anchors[:, 2] = self.anchors[:, 2] / w
        self.anchors[:, 3] = self.anchors[:, 3] / h

        self.storage_feature.reset()
        self.storage_feature.auto_reset = True

    def _loss(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the total loss (classification and bounding box regression)"""
        anchors, cls_preds, bbox_preds = preds
        bbox_offset, bbox_mask, class_labels = self.roi_blk(anchors, labels)
        _, _, num_classes = cls_preds.shape

        cls = self.cls_loss.forward(cls_preds.reshape(-1, num_classes), class_labels.reshape(-1))
        bbox = self.box_loss.forward(bbox_preds * bbox_mask, bbox_offset * bbox_mask)

        if self.hparams.loss_ratio is None:
            return cls.mean() + bbox.mean()

        mask = class_labels.reshape(-1) > 0
        gt_loss = cls[mask].mean()
        background_loss = cls[~mask].mean()

        return (
            gt_loss * self.hparams.loss_ratio
            + background_loss * (1 - self.hparams.loss_ratio)
            + bbox.mean()
        )

    def _map_compute(self):
        """Computes and logs mAP metrics at the end of an epoch"""
        result = self.map_metric.compute()
        self.log_dict(
            {
                k: result[k]
                for k in result.keys()
                if k
                in [
                    "map",
                    "map_50",
                    "mar_1",
                    "mar_10",
                    "mar_100",
                ]
            },
        )
        self.map_metric.reset()

    def _map_estimate(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ):
        """Updates the mAP metric with current predictions and targets"""
        anchors, cls_preds, bbox_preds = preds
        prep_pred = box.multibox_detection(F.softmax(cls_preds, dim=2), bbox_preds, anchors)
        map_preds = []
        map_target = []
        for batch, label in zip(prep_pred, labels):
            masked_preds = batch[batch[:, 0] >= 0]
            map_preds.append(
                {
                    "boxes": masked_preds[:, 2:],
                    "scores": masked_preds[:, 1],
                    "labels": masked_preds[:, 0].type(torch.IntTensor),
                }
            )
            masked_label = label[label[:, 0] >= 0]
            map_target.append(
                {
                    "boxes": masked_label[:, 1:],
                    "labels": masked_label[:, 0].type(torch.IntTensor),
                }
            )
        self.map_metric.update(map_preds, map_target)

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the batch for processing by truncating the time window if necessary"""
        events, targets = batch["events"], batch["targets"]
        diff = (
            torch.randint(0, self.hparams.time_window, (1,), requires_grad=False)
            if self.hparams.time_window
            else 0
        )
        if len(targets.shape) == 3:
            batch["events"] = events[diff:]
            return batch
        ts = events.shape[0] - diff
        batch["events"] = events[:ts]
        batch["targets"] = targets[ts - 1].squeeze(0) if self.net is not None else targets[0]
        return batch
