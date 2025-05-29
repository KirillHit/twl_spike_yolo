"""Tool for displaying event videos, predictions and boxes"""

import cv2
from PIL import Image
import numpy as np
import torch
from typing import List, Optional
import os
import matplotlib.colors as mcolors


class Plotter:
    """Tool for displaying event videos, predictions and boxes"""

    def __init__(
        self,
        threshold: float = 0.8,
        show_video: bool = True,
        save_video: bool = False,
        file_path: str = "log",
        file_name: str = "out",
    ):
        """
        :param threshold: Threshold value for displaying box. Defaults to 0.8.
        :type threshold: float, optional
        :param show_video: If true, shows video in window. Defaults to True.
        :type show_video: bool, optional
        :param save_video: If true, saves the video to a file. Defaults to False.
        :type save_video: bool, optional
        :param file_path: Folder for saved video. Defaults to "log".
        :type file_path: str, optional
        :param file_name: Save file name. Defaults to "out".
        :type file_name: str, optional
        """
        self.threshold = threshold
        self.show_video = show_video
        self.save_video = save_video
        self.file_path = file_path
        self.file_name = file_name
        self.colors = [
            list(reversed([int(c * 255) for c in mcolors.to_rgb(color)]))
            for color in mcolors.TABLEAU_COLORS
        ]
        self.labels = None
        self.gif_idx = 0

    def __call__(self, video: List[np.ndarray], interval: int = 50) -> None:
        """Displays frames obtained by the apply method and saves them

        :param video: List of frames
        :type video: List[np.ndarray]
        :param interval: Time between frames in milliseconds
        :type interval: int
        """
        if self.show_video:
            self._show_video(video, interval)
        if self.save_video:
            self._save_video(video, interval)

    def preprocess(self, events: torch.Tensor, image: Optional[torch.Tensor] = None):
        background = None if image is None else self.preprocess_image(image)
        return self.preprocess_events(events, background)

    def preprocess_image(self, img: torch.Tensor) -> np.ndarray:
        numpy_image = (img.cpu().numpy() * 255).astype(np.uint8)
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        return cv2_image

    def preprocess_events(
        self, events: torch.Tensor, background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        c, h, w = events.shape
        res_img = np.zeros((h, w, 3), dtype=np.uint8)
        if background is not None:
            res_img[: background.shape[0], : background.shape[1]] = background
        plt_image = events.permute(1, 2, 0).cpu()
        if c == 3:
            res_img[plt_image[..., 2] > 0] = [0, 150, 255]
        res_img[plt_image[..., 0] > 0] = [255, 150, 0]
        res_img[plt_image[..., 1] > 0, 2] = 255
        return res_img

    def apply(
        self,
        image: np.ndarray,
        predictions: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Prepares a frame from an event camera for display
        and overlays prediction and target boxes on it

        :param image: Background image
        :type image: np.ndarray
        :param predictions: Tensor shape [anchor, 6],
            one label contains (class, iou, xlu, ylu, xrd, yrd).
        :type predictions: Optional[torch.Tensor]
        :param target: Ground Truth. Tensor shape [count_box, 5],
            one label contains (class id, xlu, ylu, xrd, yrd)
        :type target: Optional[torch.Tensor]
        :return: Returns an image that can be processed by opencv
        :rtype: np.ndarray
        """
        process_image = image.copy()
        h, w, c = process_image.shape
        if target is not None and target.numel():
            target = self._prepare_targets(target, h, w)
            self._draw_target_boxes(process_image, target)
        if predictions is not None and predictions.numel():
            predictions = self._prepare_preds(predictions, h, w)
            self._draw_preds_box(process_image, predictions)
        return process_image

    def _prepare_preds(
        self, preds: Optional[torch.Tensor], height: int, width: int
    ) -> Optional[torch.Tensor]:
        preds = preds[(preds[:, 0] >= 0) & (preds[:, 1] >= self.threshold)]
        preds[:, [2, 4]] = preds[:, [2, 4]] * width
        preds[:, [3, 5]] = preds[:, [3, 5]] * height
        preds[..., 1] *= 100
        return preds.int().cpu()

    def _prepare_targets(
        self, target: Optional[torch.Tensor], height: int, width: int
    ) -> Optional[torch.Tensor]:
        target = target[target[:, 0] >= 0]
        target[:, [1, 3]] = target[:, [1, 3]] * width
        target[:, [2, 4]] = target[:, [2, 4]] * height
        return target.int().cpu()

    def _draw_preds_box(self, image: np.ndarray, preds: Optional[torch.Tensor]) -> None:
        for box in preds:
            start_point = (box[2].item(), box[3].item())
            end_point = (box[4].item(), box[5].item())
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color=self.colors[int(box[0].item()) % len(self.colors)],
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text="%.2f %s"
                % (
                    box[1].item() / 100,
                    self.labels[box[0].item()] if self.labels is not None else "",
                ),
                org=(box[2].item(), box[3].item() - 4),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                thickness=1,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA,
            )

    def _draw_target_boxes(self, image: np.ndarray, target: Optional[torch.Tensor]) -> None:
        for box in target:
            start_point = (box[1].item(), box[2].item())
            end_point = (box[3].item(), box[4].item())
            cv2.rectangle(
                image,
                start_point,
                end_point,
                color=self.colors[int(box[0].item()) % len(self.colors)],
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def _show_video(self, video: List[np.ndarray], interval: int) -> None:
        key = ord("r")
        while key == ord("r") or (key != ord("q") and cv2.waitKey() != ord("q")):
            for img in video:
                cv2.imshow("Res", img)
                key = cv2.waitKey(interval)
                if key != -1:
                    break
        cv2.destroyWindow("Res")

    def _save_video(self, video: List[np.ndarray], interval: int) -> None:
        h, w, _ = video[0].shape
        os.makedirs(self.file_path, exist_ok=True)
        step = int((1000 / interval) / 20)
        imgs = [Image.fromarray(video[idx]) for idx in range(0, len(video), step)]
        # duration is the number of milliseconds between frames; this is 40 frames per second
        imgs[0].save(
            os.path.join(self.file_path, self.file_name + self.gif_idx + ".gif"),
            save_all=True,
            append_images=imgs[1:],
            duration=50,
        )
        self.gif_idx += 1
