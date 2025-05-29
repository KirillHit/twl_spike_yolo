from lightning.pytorch.cli import LightningCLI
import lightning as L
import model
from utils import PropheseeDataModule, DSECDataModule


def cli_main():
    LightningCLI(
        model.Detector,
        L.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
