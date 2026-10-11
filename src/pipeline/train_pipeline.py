import json
import sys
from datetime import datetime, timezone

from src import constants as C
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.index_validation import IndexValidation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.fingerprint import artifact_paths, build_fingerprint, file_sha256
from src.logger import logging


def _json_default(obj):
    # numpy scalars -> native Python; anything else -> string
    return obj.item() if hasattr(obj, "item") else str(obj)


class TrainPipeline:
    def run(self):
        try:
            logging.info("=== NetZeroAI train pipeline started ===")
            build = build_fingerprint()       # taken before reading, so it describes the inputs used

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            train_df, test_df = DataTransformation().initiate_data_transformation(train_path, test_path)
            train_df, test_df, k_table_df, silhouette, explained_var = ModelTrainer().initiate_model_training(
                train_df, test_df
            )
            train_df, validation = IndexValidation().initiate_index_validation(train_df)

            train_df.to_csv(C.FINAL_TABLE_PATH, index=False)
            # Hash every artifact last, so a partial rebuild or a swapped file is detectable.
            build["artifact_sha256"] = {name: file_sha256(path) for name, path in artifact_paths().items()}

            imputation_details = [
                {"country": row["country"], "indicator": col, "value": float(row[col]), "source": row[f"{col}_impute_source"]}
                for col in C.INDICATOR_DIRECTIONS
                for _, row in train_df[train_df[f"{col}_imputed"]].iterrows()
            ]
            manifest = {
                "pipeline_version": C.PIPELINE_VERSION,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "build": build,
                "data_audit": ingestion.audit,
                "reference_year": C.REFERENCE_YEAR,
                "final_country_count": len(train_df),
                "anchor_country_count": len(test_df),
                "dimension_weights": C.DIMENSION_WEIGHTS,
                "imputed_counts": {
                    col: int(train_df[f"{col}_imputed"].sum()) for col in C.INDICATOR_DIRECTIONS
                },
                "imputation_details": imputation_details,
                "tier_counts": train_df["tier"].value_counts().to_dict(),
                "elbow_silhouette_table": k_table_df.to_dict(orient="records"),
                "silhouette_k4": float(silhouette),
                "pca_explained_variance": explained_var.tolist(),
                **validation,
            }
            with open(C.MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2, default=_json_default)

            logging.info(f"=== Pipeline complete: {len(train_df)} countries scored ===")
            print(f"NetZeroAI pipeline complete: {len(train_df)} countries scored.")
            print(f"Tier counts: {manifest['tier_counts']}")
            mc = validation["sensitivity"]["monte_carlo"]
            agreement = validation["tier_cluster_agreement"]
            print(
                f"Weight sensitivity: mean Spearman {mc['mean_spearman']:.3f} across {mc['n_draws']} draws; "
                f"{mc['pct_countries_tier_stable_80']:.0f}% of countries keep their tier in >=80% of draws"
            )
            print(
                f"Tier vs cluster: ARI {agreement['adjusted_rand_index']:.3f}, "
                f"exact agreement {agreement['exact_agreement_pct']:.0f}% "
                f"(permutation baseline {agreement['permutation_baseline']['exact_agreement_mean_pct']:.1f}%)"
            )
            print("\nAnchor country check:")
            print(test_df[["country", "netzero_index_score", "tier", "cluster_label"]].to_string(index=False))

            return train_df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run()
