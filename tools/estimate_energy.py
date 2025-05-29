from lightning.pytorch.cli import LightningCLI
import lightning as L
import torch
from torch import nn
from norse.torch import LIFCell, LICell
from model.tools.generator import ModelGenerator
from tabulate import tabulate
from typing import List
from tqdm import tqdm
import model
from utils import PropheseeDataModule, DSECDataModule
from statistics import mean


class EstimateEnergy:
    def __init__(self, e_mac: float = 4.6, e_ac: float = 0.9, delta_time: int = 10, quiet=True):
        """
        :param e_mac: energy cost of MAC operation in pJ, defaults to 4.6
        :type e_mac: float, optional
        :param e_ac: energy cost of AC operation in pJ, defaults to 0.9
        :type e_ac: float, optional
        :param delta_time: The number of the last steps used to estimate energy consumption.
            This is necessary to account for the network's operation only in its steady state.
            Defaults to 10
        :type delta_time: int, optional
        :param quiet: Print detailed information for each example, defaults to false
        :type quiet: bool, optional
        """
        self.e_mac, self.e_ac, self.delta_time, self.quiet = e_mac, e_ac, delta_time, quiet

    def process(self, det: model.Detector, batch: torch.Tensor):
        with torch.no_grad():
            flops_encode = 0
            det.storage_cls.reset()
            det.storage_box.reset()
            if det.image_model is not None:
                det.reset()
                flops_encode = self.estimate_ann(
                    det.image_model, batch["images"][0, None].to(det.device)
                )
            flops_list, activity_list = self.estimate_snn(
                det.net, batch["events"][:, 0, None].to(det.device)
            )
        return self.compare(
            det.net, flops_list, activity_list, flops_encode / batch["events"].size(0)
        )

    def compare(
        self,
        model: ModelGenerator,
        flops_list: List[float],
        activity_list: List[float],
        ann_encode: float,
    ):
        table = []

        snn_flops, ann_flops, decode_flops = 0, 0, 0
        for idx, layer in enumerate(model.net):
            if isinstance(layer, LIFCell):
                activity = activity_list[idx]
                flops = flops_list[idx]
                snn_flops += flops * activity
                ann_flops += flops
                table.append((idx, snn_flops, flops * self.e_ac, activity))
            if isinstance(layer, LICell):
                activity = activity_list[idx]
                flops = flops_list[idx]
                decode = flops_list[idx + 2]
                snn_flops += flops * activity
                decode_flops += decode
                ann_flops += decode
                table.append((idx, snn_flops, flops * self.e_ac + decode * self.e_mac, activity))

        snn_energy = snn_flops * self.e_ac + decode_flops * self.e_mac + ann_encode * self.e_mac
        ann_energy = ann_flops * self.e_mac
        mean_activity = sum(activity_list) / len(table)

        if not self.quiet:
            self.print_info(table, ann_flops, snn_flops, snn_energy, ann_energy, activity)

        return snn_energy, ann_energy, mean_activity

    def print_info(self, layers, ann_flops, snn_flops, snn_energy, ann_energy, activity):
        diff_flops = ann_flops - snn_flops
        diff_energy = ann_energy - snn_energy

        print(
            "\nList of layers:\n",
            tabulate(
                layers,
                headers=["Idx", "FLOPS", "Energy (pJ)", "Activity"],
                tablefmt="simple",
                floatfmt=(".0f", ".0f", ".0f", ".1%"),
            ),
        )

        res_table = [
            ("SNN", snn_flops * 10**-6, snn_energy * 10**-6, activity),
            ("Similar ANN", ann_flops * 10**-6, ann_energy * 10**-6, 1),
            ("Diff", diff_flops * 10**-6, diff_energy * 10**-6),
        ]
        print(
            "\nResults:\n",
            tabulate(
                res_table,
                headers=["Type", "MFLOP", "Energy (uJ)", "Activity"],
                floatfmt=(".0f", ".0f", ".2f", ".1%"),
            ),
        )

        print(
            "\nReduction in consumption compared to ANN:\n",
            tabulate(
                [
                    ["FLOP ", ann_flops / snn_flops],
                    ["Energy ", ann_energy / snn_energy],
                ],
                headers=["Value", "Decrease"],
                floatfmt=(".0f", ".2f"),
            ),
        )

    def estimate_snn(self, model: ModelGenerator, input: torch.Tensor):
        time = input.shape[0]
        state = [None] * len(model.net)
        flops_list = [0] * len(model.net)
        activity_list = [0] * len(model.net)
        for ts, data in enumerate(tqdm(input, leave=False, desc="Processing")):
            for idx, (layer, is_state) in enumerate(zip(model.net, model.state_layers)):
                if is_state:
                    data, state[idx] = layer(data, state[idx])
                else:
                    data = layer(data)
                if ts < (time -  self.delta_time - 1):
                    continue
                if isinstance(layer, LIFCell):
                    activity_list[idx] += self.compute_activity(data)
                    flops_list[idx] = flops_list[idx - 2]
                elif isinstance(layer, LICell):
                    activity_list[idx] = activity_list[idx - 3]
                    flops_list[idx] = flops_list[idx - 2]
                elif isinstance(layer, nn.Conv2d):
                    if not flops_list[idx]:
                        flops_list[idx] = self.compute_flops(data, layer)

        activity_list = [activity /  self.delta_time for activity in activity_list]

        return flops_list, activity_list

    def estimate_ann(self, model: ModelGenerator, data: torch.Tensor):
        flops_list = [0] * len(model.net)
        for idx, (layer, is_state) in enumerate(zip(model.net, model.state_layers)):
            if is_state:
                data, _ = layer(data, None)
            else:
                data = layer(data)
            if isinstance(layer, nn.SiLU):
                flops_list[idx] = flops_list[idx - 2]
            elif isinstance(layer, nn.Conv2d):
                flops_list[idx] = self.compute_flops(data, layer)
        return sum(flops_list)

    def compute_flops(self, tensor: torch.Tensor, conv: nn.Conv2d):
        h, w = tensor.shape[-2:]
        k, c_i, c_o = conv.kernel_size, conv.in_channels, conv.out_channels
        flops = (k[0] ^ 2) * h * w * c_i * c_o
        return flops

    def compute_activity(self, spikes: torch.Tensor):
        b, c, x, y = spikes.shape
        volume = b * c * x * y
        activity = torch.count_nonzero(spikes) / volume
        return activity.item()


if __name__ == "__main__":
    cli = LightningCLI(
        model.Detector,
        L.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        run=False,
    )

    cli.trainer.limit_test_batches = 2
    cli.trainer.test(
        cli.model,
        datamodule=cli.datamodule,
        ckpt_path=".neptune/yolol.ckpt",
    )

    dataloader = cli.datamodule.test_dataloader()

    print("\nStarting to calculate energy efficiency...\n")
    estimator = EstimateEnergy(quiet=True)

    n = 10
    data_iter = iter(dataloader)
    snn_energy_list, ann_energy_list, activity_list = [], [], []
    for _ in tqdm(range(n), desc="Sample"):
        snn_energy, ann_energy, activity = estimator.process(cli.model, next(data_iter))
        snn_energy_list.append(snn_energy)
        ann_energy_list.append(ann_energy)
        activity_list.append(activity)

    print(
        f"\nResults for {n} samples\n",
        tabulate(
            [
                ["SNN (mJ)", mean(snn_energy_list) * 10**-9],
                ["Activity (%)", mean(activity_list) * 100],
                ["Similar ANN (mJ)", mean(ann_energy_list) * 10**-9],
                ["Decrease", mean(ann_energy_list) / mean(snn_energy_list)],
            ],
            floatfmt=(".0f", ".2f"),
        ),
    )
