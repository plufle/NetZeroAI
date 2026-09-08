import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import constants as C
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

FEATURE_COLS = [f"{ind}_norm" for ind in C.INDICATOR_DIRECTIONS]


class ModelTrainer:
    def initiate_model_training(self, train_df, test_df):
        logging.info("Starting model training (StandardScaler + PCA + KMeans)")
        try:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[FEATURE_COLS])
            X_test = scaler.transform(test_df[FEATURE_COLS])

            k_table = []
            for k in range(2, 8):
                km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(X_train)
                k_table.append(
                    {"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X_train, km.labels_)}
                )
            k_table_df = pd.DataFrame(k_table)
            logging.info(f"Elbow/silhouette table:\n{k_table_df}")

            pca = PCA(n_components=2, random_state=C.RANDOM_STATE)
            pca_train = pca.fit_transform(X_train)
            pca_test = pca.transform(X_test)

            kmeans = KMeans(n_clusters=C.N_CLUSTERS_DEFAULT, n_init=10, random_state=C.RANDOM_STATE)
            train_labels = kmeans.fit_predict(X_train)
            final_silhouette = silhouette_score(X_train, train_labels)

            # Relabel cluster IDs by ascending mean index score, purely for
            # stable, intuitive coloring/legend order in the dashboard. This
            # relabeling does NOT feed back into the qcut-based `tier` column.
            order = (
                train_df.assign(_c=train_labels)
                .groupby("_c")["netzero_index_score"]
                .mean()
                .sort_values()
                .index.tolist()
            )
            relabel = {raw: i for i, raw in enumerate(order)}

            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["pca_1"], train_df["pca_2"] = pca_train[:, 0], pca_train[:, 1]
            test_df["pca_1"], test_df["pca_2"] = pca_test[:, 0], pca_test[:, 1]

            train_df["cluster_id"] = [relabel[c] for c in train_labels]
            test_df["cluster_id"] = [relabel[c] for c in kmeans.predict(X_test)]
            train_df["cluster_label"] = train_df["cluster_id"].apply(lambda i: f"ML Segment {i + 1}")
            test_df["cluster_label"] = test_df["cluster_id"].apply(lambda i: f"ML Segment {i + 1}")

            model_bundle = {"scaler": scaler, "pca": pca, "kmeans": kmeans, "relabel": relabel}
            save_object(C.MODEL_PATH, model_bundle)
            logging.info(f"Saved fitted scaler+PCA+KMeans bundle to {C.MODEL_PATH}")

            return train_df, test_df, k_table_df, final_silhouette, pca.explained_variance_ratio_

        except Exception as e:
            raise CustomException(e, sys)
