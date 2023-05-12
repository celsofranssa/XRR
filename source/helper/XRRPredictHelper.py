import pickle
from collections import Counter

from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer

from source.callback.PredictionWriter import PredictionWriter
from source.callback.RerankerPredictionWriter import RerankerPredictionWriter
from source.datamodule.XRRDataModule import XRRDataModule
from source.model.XRRModel import XRRModel


class XRRPredictHelper:

    def __init__(self, params):
        self.params = params

    def perform_predict(self):
        for fold_idx in self.params.data.folds:
            # data
            rankings = self._get_rankings(fold_idx)
            dm = XRRDataModule(
                params=self.params.data,
                tokenizer=self._get_tokenizer(),
                rankings=rankings,
                fold_idx=fold_idx)

            # model
            model = XRRModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.ckpt"
            )

            self.params.prediction.fold_idx = fold_idx
            # trainer
            trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[RerankerPredictionWriter(self.params.prediction)]
            )

            # predicting
            dm.prepare_data()
            dm.setup("predict")

            print(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.predict(
                model=model,
                datamodule=dm,

            )

    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.params.model.tokenizer.architecture
        )

    def _get_rankings(self, fold_idx):
        rankings = {}
        for retriever in self.params.ranking.retrievers:
            with open(f"{self.params.ranking.dir}{retriever}_{self.params.data.name}.rnk", "rb") as ranking_file:
                retriever_rankings = pickle.load(ranking_file)[fold_idx]
                for cls in ["all", "head", "tail"]:
                    if cls not in rankings:
                        rankings[cls] = {}
                    self.merge(rankings[cls], retriever_rankings[cls])
        a = 1
        return rankings

    def merge(self, rankings, retriever_rankings):
        for text_idx in retriever_rankings.keys():
            labels_scores = retriever_rankings[text_idx]
            d2 = rankings.get(text_idx, {})
            for label_idx, score in d2.items():
                if label_idx in labels_scores:
                    labels_scores[label_idx] = 0.5 * (score + labels_scores[label_idx])
                else:
                    labels_scores[label_idx] = score

            rankings[text_idx] = labels_scores
