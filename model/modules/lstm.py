"""
Convolutional LSTM
"""

import torch
from torch import nn
from typing import Tuple, Optional


class ConvLSTM(nn.Module):
    """Convolutional LSTM

    For more details, see https://github.com/ndrplz/ConvLSTM_pytorch/tree/master.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 1,
        bias: bool = False,
    ):
        """
        :param in_channels: Number of input channels.
        :type in_channels: int
        :param hidden_channels: Number of hidden channels.
        :type hidden_channels: int
        :param kernel_size: Size of the convolving kernel. Defaults to 1.
        :type kernel_size: int, optional
        :param bias: If ``True``, adds a learnable bias to the output. Defaults to False.
        :type bias: bool, optional
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def _init_hidden(self, target: torch.Tensor):
        batch, _, h, w = target.shape
        return (
            torch.zeros((batch, self.hidden_channels, h, w), device=target.device),
            torch.zeros((batch, self.hidden_channels, h, w), device=target.device),
        )

    def forward(
        self, X: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param X: Input tensor.  Shape [batch, channel, h, w].
        :type X: torch.Tensor
        :param state: Past state of the cell. Defaults to None.
            It is a list of the form: (hidden state, cell state).
        :type state: Optional[Tuple[torch.Tensor, torch.Tensor]], optional
        :return: List of form: (next hidden state, (next hidden state, next cell state)).
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        """
        hidden_state, cell_state = self._init_hidden(X) if state is None else state
        combined = torch.cat([X, hidden_state], dim=1)
        combined = self.conv(combined)
        input_gate, forget_gate, out_gate, in_node = torch.split(
            combined, self.hidden_channels, dim=1
        )
        I = torch.sigmoid(input_gate)
        F = torch.sigmoid(forget_gate)
        O = torch.sigmoid(out_gate)
        C = torch.tanh(in_node)

        cell_next = F * cell_state + I * C
        hidden_next = O * torch.tanh(cell_next)

        # This form is needed for the model generator to work
        return hidden_next, (hidden_next, cell_next)
