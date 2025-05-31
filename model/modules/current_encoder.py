"""
Constant current LIF encoder
"""

from typing import Tuple
import torch

from norse.torch.module.snn import SNNCell
from norse.torch.functional.lif import LIFParameters, LIFState, LIFFeedForwardState
from norse.torch.functional.threshold import threshold
from norse.torch.utils.clone import clone_tensor


class ConstantCurrentLIFEncoderCell(SNNCell):
    """Encodes scalar input as a spike train using a leaky integrate-and-fire (LIF) neuron model

    The input is interpreted as a constant presynaptic current. The cell integrates the input
    and emits a spike whenever the membrane potential crosses the threshold.
    """

    def __init__(self, p: LIFParameters = LIFParameters(), **kwargs):
        super().__init__(
            activation=lif_current_encoder_step,
            state_fallback=self.initial_state,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFFeedForwardState:
        state = LIFFeedForwardState(
            v=clone_tensor(self.p.v_leak),
            i=torch.zeros(
                input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


def lif_current_encoder_step(
    input_current: torch.Tensor,
    state: LIFState,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs a single integration step for the constant current LIF encoder

    Integrates the input current, applies the threshold, emits a spike if the threshold is crossed,
    and resets the membrane potential accordingly.
    """
    dv = dt * p.tau_mem_inv * input_current
    v_decayed = state.v + dv
    z = threshold(v_decayed - p.v_th, p.method, p.alpha)

    v_new = v_decayed - z * (v_decayed - p.v_reset)
    return z, LIFFeedForwardState(v=v_new, i=input_current)
