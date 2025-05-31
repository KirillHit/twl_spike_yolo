"""
Model generation tools

Custom types
------------

List for storing the state of the model

.. data:: ListState
    :type: List[torch.Tensor | None | ListState]
"""

import torch
from torch import nn
from typing import Optional, List, Any, NamedTuple, Tuple
from norse.torch.module.snn import SNNCell, _merge_states
from norse.torch.utils.state import _is_module_stateful


type ListState = List[torch.Tensor | None | ListState]


class LayerGen:
    """Base class for model layer generators

    The ``get`` method must initialize the network module and pass it to the generator.

    .. warning::
        This class can only be used as a base class for inheritance.
    """

    def get(self, in_channels: int) -> Tuple[nn.Module, int]:
        """Initializes and returns the network layer

        :param in_channels: Number of input channels.
        :type in_channels: int
        :return: The generated module and the number of channels that will be after applying
            this layer to a tensor with ``in_channels`` channels.
        :rtype: Tuple[nn.Module, int]
        """
        raise NotImplementedError


class ModelGenerator(nn.Module):
    """Tool for generating and processing a model from a list of layer generators"""

    def __init__(self, cfg: List[LayerGen], in_channels: int, init_weights: bool = True):
        """
        :param cfg: Description of the network configuration.
        :type cfg: List[LayerGen]
        :param in_channels: Number of input channels.
        :type in_channels: int
        :param init_weights: If ``true`` apply weight initialization function.
            Defaults to True.
        :type init_weights: bool, optional
        """
        super().__init__()
        self.net = nn.ModuleList()
        self.state_layers: List[bool] = []
        self.out_channels = in_channels
        for layer_gen in cfg:
            layer, self.out_channels = layer_gen.get(self.out_channels)
            self.net.append(layer)
            self.state_layers.append(_is_module_stateful(layer))

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, X: torch.Tensor, state: ListState | None = None
    ) -> Tuple[torch.Tensor, ListState]:
        state = [None] * len(self.net) if state is None else state
        for idx, (layer, is_state) in enumerate(zip(self.net, self.state_layers)):
            if is_state:
                X, state[idx] = layer(X, state[idx])
            else:
                X = layer(X)
        return X, state


class Storage(nn.Module):
    """Stores the forward pass values"""

    def __init__(self, auto_reset: bool = True):
        """
        :param auto_reset: If true, automatically clears the storage after the last call of
            the model generator. Set to False if you want the value to persist after a forward
            pass of the model. Defaults to True.
        :type auto_reset: bool, optional
        """
        super().__init__()
        self.storage = []
        self.channels = []
        self.requests_threshold = 0
        self.requests_idx = 0
        self.auto_reset = auto_reset

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Store the input tensor and returns it back"""
        self.storage.append(X)
        return X

    def add_requests(self) -> None:
        """Adds one to the receive counter threshold

        A receive is considered a call to the get method. If the threshold is not zero,
        then the storage will be automatically released when the threshold is reached.
        """
        self.requests_threshold += 1

    def add_input(self, channels: int) -> None:
        """Adds information about the input tensor that the storage will expect

        This is only necessary so that the generator can calculate the parameters
        for subsequent layers.

        :param channels: Number of input tensor channels
        :type channels: int
        """
        self.channels.append(channels)

    def shape(self) -> List[int]:
        """Returns a list of channel values for the expected data

        The method does not analyze the current data, but relies on the data received
        via the add_input method.
        """
        return self.channels

    def reset(self) -> None:
        """Resets storage"""
        self.storage = []
        self.requests_idx = 0

    def get(self) -> List[torch.Tensor]:
        """Returns a list of saved tensors"""
        temp = self.storage
        if self.auto_reset and self.requests_threshold:
            self.requests_idx += 1
            if self.requests_idx == self.requests_threshold:
                self.reset()
        return temp


class StorageGetter(nn.Module):
    """Returns the specified stored tensor"""

    def __init__(self, storage: Storage, idx: int = 0):
        super().__init__()
        self.storage, self.idx = storage, idx

    def forward(self, X=None) -> torch.Tensor:
        return self.storage.get()[self.idx]


class ResidualModule(nn.Module):
    """The class combines data stored in the storage by summing or concatenating

    It is used to create dense or residual networks.
    """

    def __init__(self, mode: str, storage: Storage):
        """
        :param mode: Method of combining:

            * "residual" - summarizes data
            * "dense" - combines data across channels
        :type mode: str
        :param storage: Storage to merge
        :type storage: Storage
        """
        super().__init__()
        self.storage = storage
        modes = {"residual": self._residual, "dense": self._dense}
        self.func = modes[mode]

    def forward(self, X=None) -> torch.Tensor:
        return self.func(self.storage.get())

    def _residual(self, input: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(input).sum(dim=0)

    def _dense(self, input: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(input, dim=1)


class StateStorage(torch.nn.Module):
    """
    Class wrapper for neurons with a state of Norse.
    Saves the intermediate states of the stored neurons for further analysis.
    """

    def __init__(self, m: SNNCell):
        """
        :param m: An initialized module with a state from the Norse library,
            which will be called with a direct passage
        :type m: SNNCell
        """
        super().__init__()
        self.module = m
        self.state_list: List[torch.Tensor] = []
        self.out_list: List[torch.Tensor] = []

    def get_state(self) -> NamedTuple:
        """Returns intermediate states of neurons"""
        return _merge_states(self.state_list)

    def get_out(self) -> torch.Tensor:
        """Returns the output of neurons for all time steps"""
        return torch.stack(self.out_list)

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        """
        Conveys the value to his module directly.
        If the network is in non -training mode, retains intermediate states
        """
        if state is None:
            self.state_list.clear()
            self.out_list.clear()
        out, new_state = self.module(input_tensor, state)
        if not self.training:
            self.state_list.append(self._state_to_cpu(new_state))
            self.out_list.append(out.cpu())
        return out, new_state

    def _state_to_cpu(self, states: NamedTuple) -> NamedTuple:
        state_dict = states._asdict()
        cls = states.__class__
        keys = list(state_dict.keys())
        output_dict = {key: getattr(states, key).cpu() for key in keys}
        return cls(**output_dict)
