"""
Network configuration similar to yolo8, adapted for multimodal input
"""

import torch
from typing import List
from model.detector import Detector
import model.tools.generator as gen
import model.tools.layers as l


class MultimodalYolo(Detector):
    """Generates a model similar to yolo8, adapted for multimodal input

    This class provides a network architecture that processes both static images (via a dedicated 
    ANN branch) and event-based data, allowing for flexible and efficient multimodal fusion.
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        :param model: model type - n, s, m, l, x
        :type model: str
        :raises KeyError: Invalid model type.
        """
        super().__init__(*args, **kwargs)
        model_types = {
            "n": (1 / 3, 0.25, 2.0),
            "s": (1 / 3, 0.50, 2.0),
            "m": (2 / 3, 0.75, 1.5),
            "l": (1.0, 1.00, 1.0),
            "x": (1.0, 1.25, 1.0),
        }
        # d: depth_multiple, w: width_multiple, r: ratio
        self.d, self.w, self.r = model_types[model]
        self.image_feature = gen.Storage(auto_reset=False)
        self.image_model = gen.ModelGenerator(self.image_cfgs(), 3, self.hparams.init_weights)
        self.net = gen.ModelGenerator(self.get_cfgs(), self.hparams.in_channels, self.hparams.init_weights)
        self.load()
    
    def image_forward(self, image: torch.Tensor) -> None:
        self.image_feature.reset()
        self.image_model(image)

    def image_cfgs(self) -> List[gen.LayerGen]:
        return [
            *self._conv(int(64 * self.w), 3, 2, lif=False),
            *self._conv(int(128 * self.w), 3, 2, lif=False),
            *self._c2f(int(128 * self.w), int(3 * self.d), lif=False),
            l.Store(self.image_feature),
        ]

    def get_cfgs(self) -> List[gen.LayerGen]:
        storage_cnn = gen.Storage()
        storage_4 = gen.Storage()
        storage_6 = gen.Storage()
        storage_9 = gen.Storage()
        storage_12 = gen.Storage()
        return [
            *self._conv(int(64 * self.w), 3, 2),
            *self._conv(int(128 * self.w), 3, 2),
            *self._c2f(int(128 * self.w), int(3 * self.d)),
            l.Store(storage_cnn),
            l.Get(self.image_feature),
            l.CurrentEncoder(),
            l.Store(storage_cnn),
            l.Dense(storage_cnn),
            *self._conv(int(256 * self.w), 3, 2),
            *self._c2f(int(256 * self.w), int(6 * self.d)),
            l.Store(storage_4),
            *self._conv(int(512 * self.w), 3, 2),
            *self._c2f(int(512 * self.w), int(6 * self.d)),
            l.Store(storage_6),
            *self._conv(int(512 * self.w * self.r), 3, 2),
            *self._c2f(int(512 * self.w * self.r), int(3 * self.d)),
            l.Store(storage_9),
            l.Up(),
            l.Store(storage_6),
            l.Dense(storage_6),
            *self._c2f(int(512 * self.w), int(3 * self.d), False),
            l.Store(storage_12),
            l.Up(),
            l.Store(storage_4),
            l.Dense(storage_4),
            *self._c2f(int(256 * self.w), int(3 * self.d), False),
            l.Store(self.storage_feature),
            *self._conv(int(256 * self.w), 3, 2),
            l.Store(storage_12),
            l.Dense(storage_12),
            *self._c2f(int(512 * self.w), int(3 * self.d), False),
            l.Store(self.storage_feature),
            *self._conv(int(512 * self.w), 3, 2),
            l.Store(storage_9),
            l.Dense(storage_9),
            *self._c2f(int(512 * self.w * self.r), int(3 * self.d), False),
            l.Store(self.storage_feature),
            *self._detect(self.storage_feature, 0),
            *self._detect(self.storage_feature, 1),
            *self._detect(self.storage_feature, 2),
        ]

    def _detect(
        self,
        storage_detect: gen.Storage,
        idx: int,
    ) -> List[gen.LayerGen]:
        storage = gen.Storage()
        return (
            l.Get(storage_detect, idx),
            l.Store(storage),
            *self._conv(),
            *self._conv(),
            *self._decode(self.num_box_out[idx]),
            l.Store(self.storage_box),
            l.Get(storage),
            *self._conv(),
            *self._conv(),
            *self._decode(self.num_class_out[idx]),
            l.Store(self.storage_cls),
        )

    def _decode(self, out_channels: int):
        return (
            l.Conv(kernel_size=1),
            l.Norm(),
            l.LI(dt=self.hparams.dt, state_storage=self.hparams.state_storage),
            l.Tanh(),
            l.Conv(out_channels, kernel_size=1),
        )

    def _conv(self, out_channels: int = None, kernel: int = 3, stride: int = 1, lif=True):
        return (
            l.Conv(out_channels, stride=stride, kernel_size=kernel, bias=not lif),
            l.Norm(bias=not lif),
            l.LIF(dt=self.hparams.dt, state_storage=self.hparams.state_storage)
            if lif
            else l.SiLU(),
        )

    def _bottleneck(self, shortcut: bool = True, lif=True):
        net = (*self._conv(lif=lif), *self._conv(lif=lif))
        if shortcut:
            storage = gen.Storage()
            return [l.Store(storage), *net, l.Store(storage), l.Residual(storage)]
        else:
            return net

    def _c2f(self, out_channels: int, n: int, shortcut: bool = True, lif=True):
        in_storage = gen.Storage()
        dense_storage = gen.Storage()
        net = []
        for _ in range(n):
            net += [*self._bottleneck(shortcut, lif=lif), l.Store(dense_storage)]
        return (
            l.Store(in_storage),
            l.Conv(int(out_channels / 2), 1, bias=not lif),
            l.Store(dense_storage),
            l.Get(in_storage),
            l.Conv(int(out_channels / 2), 1, bias=not lif),
            l.Store(dense_storage),
            *net,
            l.Dense(dense_storage),
            l.Conv(out_channels, 1, bias=not lif),
        )
