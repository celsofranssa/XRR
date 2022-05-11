import os
import hydra
from omegaconf import OmegaConf

from source.helper.FitHelper import FitHelper


def fit(params):
    fit_helper = FitHelper(params)
    fit_helper.perform_fit()


@hydra.main(config_path="settings/", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "fit" in params.tasks:
        fit(params)


if __name__ == '__main__':
    perform_tasks()
