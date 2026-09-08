import json
import sys
from datetime import datetime, timezone

from src import constants as C
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run(self):
        try:
            logging.info("=== NetZeroAI train pipeline started ===")

            train_path, test_path = DataIngestion().initiate_data_ingestion()
            train_df, test_df = DataTransformation().initiate_data_transformation(train_path, test_path)
            train_df, test_df, k_table_df, silhouette, explained_var = ModelTrainer().initiate_model_training(
                train_df, test_df
            )

            train_df.to_csv(C.FINAL_TABLE_PATH, index=False)

            manifest = {
                "built_at": datetime.now(timezone.utc).isoformat(),
                "reference_year": C.REFERENCE_YEAR,
                "final_country_count": len(train_df),
                "anchor_country_count": len(test_df),
                "imputed_counts": {
                    col: int(train_df[f"{col}_imputed"].sum()) for col in C.INDICATOR_DIRECTIONS
                },
                "tier_counts": train_df["tier"].value_counts().to_dict(),
                "elbow_silhouette_table": k_table_df.to_dict(orient="records"),
                "silhouette_k4": float(silhouette),
                "pca_explained_variance": explained_var.tolist(),
            }
            with open(C.MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2, default=str)

            logging.info(f"=== Pipeline complete: {len(train_df)} countries scored ===")
            print(f"NetZeroAI pipeline complete: {len(train_df)} countries scored.")
            print(f"Tier counts: {manifest['tier_counts']}")
            print("\nAnchor country check:")
            print(test_df[["country", "netzero_index_score", "tier", "cluster_label"]].to_string(index=False))

            return train_df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run()
