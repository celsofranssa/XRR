import os
import hydra
from omegaconf import OmegaConf

from source.helper.PreprocessHelper import PreprocessHelper
from source.helper.ULSEEvalHelper import ULSEEvalHelper
from source.helper.ULSEFitHelper import ULSEFitHelper
from source.helper.ULSEPredictHelper import ULSEPredictHelper
from source.helper.XRREvalHelper import XRREvalHelper
from source.helper.XRRFitHelper import XRRFitHelper
from source.helper.XRRPredictHelper import XRRPredictHelper


def preprocess(params):
    if params.model.type == "retriever":
        helper = PreprocessHelper(params)
        helper.perform_preprocess()


def fit(params):
    if params.model.type == "reranker":
        fit_helper = XRRFitHelper(params)
        fit_helper.perform_fit()

    elif params.model.type == "retriever":
        fit_helper = ULSEFitHelper(params)
        fit_helper.perform_fit()


def predict(params):
    if params.model.type == "reranker":
        helper = XRRPredictHelper(params)
        helper.perform_predict()

    elif params.model.type == "retriever":
        helper = ULSEPredictHelper(params)
        helper.perform_predict()

def eval(params):
    if params.model.type == "reranker":
        helper = XRREvalHelper(params)
        helper.perform_eval()

    elif params.model.type == "retriever":
        helper = ULSEEvalHelper(params)
        helper.perform_eval()


@hydra.main(config_path="setting/", config_name="setting.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "preprocess" in params.tasks:
        preprocess(params)

    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)


if __name__ == '__main__':
    perform_tasks()
