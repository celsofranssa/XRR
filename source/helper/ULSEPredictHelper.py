import pickle

from omegaconf import OmegaConf
import pytorch_lightning as pl

from source.callback.PredictionWriter import PredictionWriter
from source.datamodule.ULSEDataModule import ULSEDataModule
from source.model.ULSEModel import ULSEModel


class ULSEPredictHelper:

    def __init__(self, params):
        self.params = params

    def perform_predict(self):
        for fold_idx in self.params.data.folds:
            # datamodule
            vectorizer = self.load_vectorizer(fold_id=fold_idx)
            datamodule = ULSEDataModule(
                self.params.data,
                tokenizer=vectorizer.build_tokenizer(),
                vocabulary=vectorizer.vocabulary_,
                fold_idx=fold_idx)
            # predicting
            datamodule.prepare_data()
            datamodule.setup("predict")

            # model
            model = ULSEModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.ckpt"
            )

            self.params.prediction.fold_idx = fold_idx
            # trainer
            trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[PredictionWriter(self.params.prediction)]
            )



            print(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.predict(
                model=model,
                datamodule=datamodule,

            )

    def load_vectorizer(self, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/vectorizer.pkl", "rb") as vectorizer_file:
            return pickle.load(vectorizer_file)
