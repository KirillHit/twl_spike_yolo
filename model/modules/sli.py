"""
Model of saturable leaky integrator
"""

import torch
from typing import Tuple, NamedTuple
from norse.torch.module.snn import SNNCell, SNN

__all__ = (
    "SLIState",
    "SLIParameters",
    "SLICell",
)


class SLIState(NamedTuple):
    """State of a saturable-leaky-integrator"""

    v: torch.Tensor
    """membrane voltage"""

    i: torch.Tensor
    """input current"""


class SLIParameters(NamedTuple):
    """Parameters of a saturable leaky integrator"""

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    """inverse synaptic time constant"""

    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    """inverse membrane time constant"""

    v_leak: torch.Tensor = torch.as_tensor(0.0)
    """leak potential"""

    v_st: torch.Tensor = torch.as_tensor(1.0)
    """saturation potential in mV"""


class SLI(SNN):
    """A neuron layer that wraps a saturable-leaky-integrator :class:`SLICell` in time.

    The layer iterates over the  _outer_ dimension of the input.
    """

    def __init__(self, p: SLIParameters = SLIParameters(), **kwargs):
        """
        :param p: parameters of the leaky integrator
        :type p: SLIParameters, optional
        :param dt: integration timestep to use
        :type p: float, optional
        """
        super().__init__(
            activation=sli_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> SLIState:
        state = SLIState(
            v=torch.full(
                input_tensor.shape[1:],  # Assume first dimension is time
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *input_tensor.shape[1:],
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


class SLICell(SNNCell):
    """Cell for a leaky-integrator"""

    def __init__(self, p: SLIParameters = SLIParameters(), **kwargs):
        """
        :param p: parameters of the leaky integrator
        :type p: SLIParameters, optional
        :param dt: integration timestep to use
        :type p: float, optional
        """
        super().__init__(
            activation=sli_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> SLIState:
        state = SLIState(
            v=self.p.v_leak.detach(),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state


def sli_feed_forward_step(
    input_tensor: torch.Tensor,
    state: SLIState,
    p: SLIParameters = SLIParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, SLIState]:
    # compute current jumps
    i_jump = state.i + input_tensor * torch.sigmoid(p.v_st - torch.abs(state.v))
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_new = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    return v_new, SLIState(v_new, i_decayed)
