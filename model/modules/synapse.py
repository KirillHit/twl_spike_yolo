"""
Model of synaptic transmission
"""

import torch
from typing import Tuple, NamedTuple
from norse.torch.module.snn import SNNCell

__all__ = (
    "SynapseState",
    "SynapseParameters",
    "SynapseCell",
)


class SynapseState(NamedTuple):
    """State of a synapse"""

    p: torch.Tensor
    """mediator concentration"""


class SynapseParameters(NamedTuple):
    """Parameters of a synapse"""

    tau_med_secretion: torch.Tensor = torch.as_tensor(1.0 / 1e-3)
    """time constant of mediator secretion"""

    tau_med_dissociation: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    """time constant of mediator dissociation"""

    sigma_inhibition: torch.Tensor = torch.as_tensor(0.0)
    """critical value of mediator concentration

    Must be >= 0.5. If equal to 0, synaptic inhibition is not applied.
    """


class SynapseCell(SNNCell):
    """Model of synaptic transmission"""

    def __init__(self, p: SynapseParameters = SynapseParameters(), **kwargs):
        """
        :param p: parameters of the leaky integrator
        :type p: SLIParameters, optional
        :param dt: integration timestep to use
        :type p: float, optional
        """
        super().__init__(
            activation=synapse_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )
        if (p.sigma_inhibition != 0) & (p.sigma_inhibition < 0.5):
            raise ValueError(
                "Valid values for sigma_inhibition are 0 or >= 0.5, but received ",
                p.sigma_inhibition,
            )

    def initial_state(self, input_tensor: torch.Tensor) -> SynapseState:
        state = SynapseState(
            p=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.p.requires_grad = True
        return state


def synapse_feed_forward_step(
    input_tensor: torch.Tensor,
    state: SynapseState,
    p: SynapseParameters = SynapseParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, SynapseState]:
    """Synopsis calculation algorithm

    For more details, see https://andjournal.sgu.ru/sites/andjournal.sgu.ru/files/text-pdf/2022/05/bakhshiev-demcheva_299-310.pdf.
    Paragraph 1.2
    """
    tau = torch.empty_like(
        input_tensor,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    tau_mask = input_tensor > 0
    tau[tau_mask] = p.tau_med_secretion
    tau[~tau_mask] = p.tau_med_dissociation

    dp = (input_tensor - state.p) * tau * dt

    p_new = state.p + dp

    if p.sigma_inhibition.is_nonzero():
        g = 4 * p.sigma_inhibition * (p_new - p.sigma_inhibition * p_new.square())
    else:
        g = p_new
    g = g.clamp(0.0)

    return g, SynapseState(p_new)
