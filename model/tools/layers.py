"""
Layer Generators
"""

from torch import nn
from typing import Tuple, Optional
import norse.torch as snn
from norse.torch.module.snn import SNNCell
from model.modules.synapse import SynapseCell
from model.modules.sum_pool import SumPool2d
from model.modules.lstm import ConvLSTM
from model.modules.sli import SLICell
import model.tools.generator as gen
from model.modules.current_encoder import ConstantCurrentLIFEncoderCell


class Store(gen.LayerGen):
    """Stores a tensor in the specified storage and passes it on"""

    def __init__(self, storage: gen.Storage):
        self.storage = storage

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        self.storage.add_input(in_channels)
        return self.storage, in_channels


class Get(gen.LayerGen):
    """Get tensor from storage"""

    def __init__(self, storage: gen.Storage, idx: int = 0):
        self.storage, self.idx = storage, idx
        self.storage.add_requests()

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        shape = self.storage.shape()
        if len(shape) <= self.idx:
            raise RuntimeError("Attempt to access a non-existent tensor in storage")
        return gen.StorageGetter(self.storage, self.idx), shape[self.idx]


class Residual(gen.LayerGen):
    """Summarizes data in storage

    Needed to create residual networks.
    """

    def __init__(self, storage: gen.Storage):
        self.storage = storage
        self.storage.add_requests()

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        shape = self.storage.shape()
        if shape.count(shape[0]) != len(shape):
            raise RuntimeError(
                "The residual network received tensors of different shapes: " + str(shape)
            )
        return gen.ResidualModule("residual", self.storage), shape[0]


class Dense(gen.LayerGen):
    """Combines data across channels

    Needed to create dense networks.
    """

    def __init__(self, storage: gen.Storage):
        self.storage = storage
        self.storage.add_requests()

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return gen.ResidualModule("dense", self.storage), sum(self.storage.shape())


class Conv(gen.LayerGen):
    """Generator of standard 2d convolution

    Uses :external:class:`torch.nn.Conv2d` module.
    Bias defaults to ``False``, padding is calculated automatically.
    """

    def __init__(self, out_channels: int = None, kernel_size: int = 3, stride: int = 1, bias=False):
        """
        :param out_channels: Number of channels produced by the convolution.
            Defaults to None.
        :type out_channels: int, optional
        :param kernel_size:  Size of the convolving kernel. Defaults to 3.
        :type kernel_size: int, optional
        :param stride: Stride of the convolution. Defaults to 1.
        :type stride: int, optional
        """
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        out = in_channels if self.out_channels is None else self.out_channels
        return nn.Conv2d(
            in_channels,
            out,
            kernel_size=self.kernel_size,
            padding=int(self.kernel_size / 2),
            stride=self.stride,
            bias=self.bias,
        ), out


class Pool(gen.LayerGen):
    """Pooling layer generator

    Uses modules :external:class:`torch.nn.AvgPool2d`,
    :external:class:`torch.nn.MaxPool2d`, :class:`SumPool2d`.
    """

    def __init__(self, type: str, kernel_size: int = 2, stride: Optional[int] = None):
        """
        :param type: Pooling type.

            - ``A`` - :external:class:`torch.nn.AvgPool2d`.
            - ``M`` - :external:class:`torch.nn.MaxPool2d`.
            - ``S`` - :class:`SumPool2d`.
        :type type: str
        :param kernel_size: The size of the window. Defaults to 2.
        :type kernel_size: int, optional
        :param stride: The stride of the window. Default value is kernel_size.
        :type stride: Optional[int], optional
        :raises ValueError: Non-existent pool type.
        """
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        match type:
            case "A":
                self.pool = nn.AvgPool2d
            case "M":
                self.pool = nn.MaxPool2d
            case "S":
                self.pool = SumPool2d
            case _:
                raise ValueError(f'[ERROR]: Non-existent pool type "{type}"!')

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return self.pool(
            self.kernel_size, stride=self.stride, padding=int(self.kernel_size / 2)
        ), in_channels


class Up(gen.LayerGen):
    """Upsample layer generator

    Uses :external:class:`torch.nn.Upsample` module.
    """

    def __init__(self, scale: int = 2, mode: str = "nearest"):
        """
        :param scale:  Multiplier for spatial size. Defaults to 2.
        :type scale: int, optional
        :param mode: The upsampling algorithm: one of 'nearest', 'linear', 'bilinear',
            'bicubic' and 'trilinear'. Defaults to "nearest".
        :type mode: str, optional
        """
        self.scale = scale
        self.mode = mode

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.Upsample(scale_factor=self.scale, mode=self.mode), in_channels


class Norm(gen.LayerGen):
    """Batch Normalization layer generator

    Uses :external:class:`torch.nn.BatchNorm2d` module.
    """

    def __init__(self, bias: bool = False):
        """
        :param bias: If True, adds a learnable bias. Defaults to False.
        :type bias: bool, optional
        """
        self.bias = bias

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        norm_layer = nn.BatchNorm2d(in_channels)
        if not self.bias:
            norm_layer.bias = None
        return norm_layer, in_channels


class LIF(gen.LayerGen):
    """Generator of the layer of LIF neurons

    Uses :external:class:`norse.torch.module.lif.LIFCell` module.
    """

    def __init__(self, dt: float = 0.001, state_storage: bool = False):
        """
        :param dt: Time step to use in integration. Defaults to 0.001.
        :type dt: float, optional
        :param state_storage: If the truth, wraps the module into the :class:`StateStorage` class,
            in which the intermediate states of the neuron are preserved for analysis, defaults to False
        :type state_storage: bool, optional
        """
        self.dt = dt
        self.state_storage = state_storage

    def get(self, in_channels: int) -> Tuple[SNNCell, int]:
        cell = snn.LIFCell(dt=self.dt)
        module = cell if not self.state_storage else gen.StateStorage(cell)
        return module, in_channels


class LI(gen.LayerGen):
    """Generator of the layer of LI neurons

    Uses :external:class:`norse.torch.module.leaky_integrator.LICell` module.
    """

    def __init__(self, dt: float = 0.001, state_storage: bool = False):
        """
        :param dt: Time step to use in integration. Defaults to 0.001.
        :type dt: float, optional
        :param state_storage: If the truth, wraps the module into the :class:`StateStorage` class,
            in which the intermediate states of the neuron are preserved for analysis, defaults to False
        :type state_storage: bool, optional
        """
        self.dt = dt
        self.state_storage = state_storage

    def get(self, in_channels: int) -> Tuple[SNNCell, int]:
        cell = snn.LICell(dt=self.dt)
        module = cell if not self.state_storage else gen.StateStorage(cell)
        return module, in_channels


class CurrentEncoder(gen.LayerGen):
    """Translates scalar data into a spiking form, interpreting it as a presynaptic current

    Uses :external:class:`norse.torch.module.encode.ConstantCurrentLIFEncoder` module.
    """

    def get(self, in_channels: int) -> Tuple[SNNCell, int]:
        return ConstantCurrentLIFEncoderCell(), in_channels


class ReLU(gen.LayerGen):
    """ReLU layer generator

    Uses :external:class:`torch.nn.ReLU` module.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.ReLU(), in_channels


class SiLU(gen.LayerGen):
    """SiLU layer generator

    Uses :external:class:`torch.nn.SiLU` module.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.SiLU(), in_channels


class Tanh(gen.LayerGen):
    """SiLU layer generator

    Uses :external:class:`torch.nn.Tanh` module.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        return nn.Tanh(), in_channels


class LSTM(gen.LayerGen):
    """LSTM layer generator

    Uses :class:`ConvLSTM <models.module.conv_lstm.ConvLSTM>` module.
    """

    def __init__(self, hidden_size: Optional[int] = None):
        """
        :param hidden_size: Number of hidden channels. Defaults to None.
        :type hidden_size: Optional[int], optional
        """
        self.hidden_size = hidden_size

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        h_size = in_channels if self.hidden_size is None else self.hidden_size
        return ConvLSTM(in_channels, h_size), h_size


class Synapse(gen.LayerGen):
    """Generator of the layer of synapse

    Uses :class:`SynapseCell <models.module.synapse.SynapseCell>` module.
    """

    def get(self, in_channels: int) -> Tuple[snn.LICell, int]:
        return SynapseCell(), in_channels


class SLI(gen.LayerGen):
    """Generator of the layer of Saturable LI neurons

    Uses :class:`SLICell <models.module.sli.SLICell>` module.
    """

    def __init__(self, state_storage: bool = False):
        """
        :param state_storage: If the truth, wraps the module into the :class:`StateStorage` class,
            in which the intermediate states of the neuron are preserved for analysis, defaults to False
        :type state_storage: bool, optional
        """
        self.state_storage = state_storage

    def get(self, in_channels: int) -> Tuple[SNNCell, int]:
        module = SLICell() if not self.state_storage else gen.StateStorage(SLICell())
        return module, in_channels
