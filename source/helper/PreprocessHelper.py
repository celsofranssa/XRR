import pickle

import pandas as pd
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class PreprocessHelper:

    def __init__(self, params):
        self.params = params

    def get_samples(self, fold_id, split):
        ids = self.load_ids(fold_id, split)
        samples_df = pd.DataFrame(self.load_samples())
        samples_df = samples_df[samples_df["idx"].isin(ids)].reset_index(drop=True)
        return samples_df

    def load_ids(self, fold_id, split):
        with open(f"{self.params.data.dir}fold_{fold_id}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def load_samples(self):
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            return pickle.load(samples_file)

    def get_vectorizer(self, texts):
        return TfidfVectorizer(
            analyzer="word", max_features=self.params.data.vocabulary_size
        ).fit(texts)

    def checkpoint_vectorizer(self, vectorizer, fold_id):
        with open(f"{self.params.data.dir}fold_{fold_id}/vectorizer.pkl", "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

    def perform_preprocess(self):
        for fold_idx in self.params.data.folds:
            print(
                f"Preprocess {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            # vectorizer
            train_samples_df = self.get_samples(fold_id=fold_idx, split="train")
            train_samples_df["text"] = train_samples_df.apply(lambda row: f"{row['text']} {' '.join(row['labels'])}",
                                                              axis=1)

            vectorizer = self.get_vectorizer(train_samples_df["text"])
            self.checkpoint_vectorizer(vectorizer, fold_idx)
