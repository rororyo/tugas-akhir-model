import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionPreprocessor:
    """
    Production Preprocessor - Uses mean/centroid GNN embeddings for unseen customers.
    Keeps variable names to match the original pipeline as closely as possible.
    """

    def __init__(self, artifacts_dir="fraud_detection_artifacts"):
        """Load artifacts"""
        self.artifacts_dir = artifacts_dir

        # Load scaler
        with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        # Load encoders
        with open(os.path.join(artifacts_dir, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)

        # Load thresholds
        with open(os.path.join(artifacts_dir, "thresholds.json"), "r") as f:
            thresholds = json.load(f)
            self.bins = np.array(thresholds["bins"])
            self.high_amount_threshold = thresholds["high_amount_threshold"]
            self.high_quantity_threshold = thresholds["high_quantity_threshold"]
            self.OHE_THRESHOLD = thresholds["OHE_THRESHOLD"]

        # ✅ Load global statistics (if available)
        global_stats_path = os.path.join(artifacts_dir, "global_statistics.json")
        if os.path.exists(global_stats_path):
            with open(global_stats_path, "r") as f:
                self.global_stats = json.load(f)
            print("✅ Using global statistics (no data leakage)")
            self.use_global_stats = True
        else:
            # Fallback to customer aggregations (old approach)
            print("⚠️ global_statistics.json not found - using customer aggregations")
            print("   Consider retraining with the fixed version!")
            self.customer_agg = pd.read_pickle(
                os.path.join(artifacts_dir, "customer_aggregations.pkl")
            )
            self.use_global_stats = False

        # Load feature info
        with open(os.path.join(artifacts_dir, "feature_info.json"), "r") as f:
            feature_info = json.load(f)
            self.feature_names = feature_info["feature_names"]
            self.n_gnn_features = feature_info["n_gnn_features"]
            self.n_tabular_features = feature_info["n_tabular_features"]

        # Attempt to load mean / centroid embedding for GNN features
        self.mean_gnn_embedding = self._load_mean_gnn_embedding()
        # Attempt to load clipping bounds (optional)
        self.gnn_min, self.gnn_max = self._load_gnn_clip_bounds()

        print(f"✓ Loaded artifacts from {artifacts_dir}")
        print(f"  - Features: {len(self.feature_names)} ({self.n_tabular_features} tabular + {self.n_gnn_features} GNN)")
        if self.mean_gnn_embedding is not None:
            print(f"  - ✅ Using MEAN/centroid GNN embedding (shape: {self.mean_gnn_embedding.shape}) for unseen nodes")
        else:
            print(f"  - ⚠️ MEAN GNN embedding not found; will fallback to ZERO embeddings (NOT recommended)")

    def _load_mean_gnn_embedding(self):
        """Try multiple locations/formats to load or compute a mean GNN embedding."""
        # 1) Preferred: mean_gnn_embedding.npy
        mean_npy = os.path.join(self.artifacts_dir, "mean_gnn_embedding.npy")
        if os.path.exists(mean_npy):
            try:
                arr = np.load(mean_npy)
                arr = arr.astype(np.float32)
                if arr.shape[0] == self.n_gnn_features:
                    return arr
                # if 2D and single row, squeeze
                if arr.ndim == 2 and arr.shape[1] == self.n_gnn_features and arr.shape[0] == 1:
                    return arr.squeeze(0).astype(np.float32)
                print("⚠️ mean_gnn_embedding.npy found but shape mismatch. Ignoring.")
            except Exception:
                print("⚠️ Failed to load mean_gnn_embedding.npy. Ignoring.")

        # 2) mean_gnn_embedding.pkl
        mean_pkl = os.path.join(self.artifacts_dir, "mean_gnn_embedding.pkl")
        if os.path.exists(mean_pkl):
            try:
                with open(mean_pkl, "rb") as f:
                    arr = pickle.load(f)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.shape[0] == self.n_gnn_features:
                    return arr
                print("⚠️ mean_gnn_embedding.pkl found but shape mismatch. Ignoring.")
            except Exception:
                print("⚠️ Failed to load mean_gnn_embedding.pkl. Ignoring.")

        # 3) Compute from node embeddings file if available: gnn_node_embeddings.npy / .pkl
        nodes_npy = os.path.join(self.artifacts_dir, "gnn_node_embeddings.npy")
        if os.path.exists(nodes_npy):
            try:
                all_emb = np.load(nodes_npy).astype(np.float32)
                if all_emb.ndim == 2 and all_emb.shape[1] == self.n_gnn_features:
                    return np.mean(all_emb, axis=0).astype(np.float32)
                print("⚠️ gnn_node_embeddings.npy found but shape mismatch. Ignoring.")
            except Exception:
                print("⚠️ Failed to load gnn_node_embeddings.npy. Ignoring.")

        nodes_pkl = os.path.join(self.artifacts_dir, "gnn_node_embeddings.pkl")
        if os.path.exists(nodes_pkl):
            try:
                with open(nodes_pkl, "rb") as f:
                    all_emb = pickle.load(f)
                all_emb = np.asarray(all_emb, dtype=np.float32)
                if all_emb.ndim == 2 and all_emb.shape[1] == self.n_gnn_features:
                    return np.mean(all_emb, axis=0).astype(np.float32)
                print("⚠️ gnn_node_embeddings.pkl found but shape mismatch. Ignoring.")
            except Exception:
                print("⚠️ Failed to load gnn_node_embeddings.pkl. Ignoring.")

        # 4) Nothing found: return None so caller can decide (fallback to zeros if required)
        return None

    def _load_gnn_clip_bounds(self):
        """Optional: load gnn_min.npy and gnn_max.npy for clipping embeddings to training ranges."""
        gnn_min_path = os.path.join(self.artifacts_dir, "gnn_min.npy")
        gnn_max_path = os.path.join(self.artifacts_dir, "gnn_max.npy")
        if os.path.exists(gnn_min_path) and os.path.exists(gnn_max_path):
            try:
                gmin = np.load(gnn_min_path).astype(np.float32)
                gmax = np.load(gnn_max_path).astype(np.float32)
                if gmin.shape[0] == self.n_gnn_features and gmax.shape[0] == self.n_gnn_features:
                    return gmin, gmax
                print("⚠️ gnn_min / gnn_max found but shape mismatch. Ignoring clipping.")
            except Exception:
                print("⚠️ Failed to load gnn_min/gnn_max. Ignoring clipping.")
        return None, None

    def _get_replacement_gnn(self, n_rows: int):
        """
        Return an array of shape (n_rows, n_gnn_features) to use for unseen nodes.
        Uses mean embedding if available; otherwise fallback to zeros (with warning).
        Applies clipping if clip bounds are available.
        """
        if self.mean_gnn_embedding is not None:
            rep = np.tile(self.mean_gnn_embedding.astype(np.float32), (n_rows, 1))
            # optional clipping to training min/max
            if self.gnn_min is not None and self.gnn_max is not None:
                rep = np.clip(rep, self.gnn_min.astype(np.float32), self.gnn_max.astype(np.float32))
            return rep
        else:
            # Fallback — this is the less-desirable option. Keep dtype consistent.
            print("⚠️ WARNING: mean GNN embedding not available. Falling back to ZERO embeddings. Please compute and save mean_gnn_embedding.npy")
            return np.zeros((n_rows, self.n_gnn_features), dtype=np.float32)

    def preprocess(self, df, enabled_features=None):
        """
        Preprocess and append GNN embeddings (mean/centroid for unseen nodes).
        Keeps original variable names and flow as much as possible.
        """
        df = df.copy()

        # Handle enabled_features
        if enabled_features is None:
            enabled_features = {col: True for col in df.columns}

        for col, is_enabled in enabled_features.items():
            if not is_enabled and col in df.columns:
                df[col] = np.nan

        # Convert Transaction Date
        if "Transaction Date" in df.columns:
            df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors='coerce')

        # ============================================
        # FEATURE ENGINEERING
        # ============================================

        if self.use_global_stats:
            # ✅ NEW APPROACH: Use global statistics

            # Fill missing values with global medians
            if "Transaction Amount" not in df.columns or df["Transaction Amount"].isna().all():
                df["Transaction Amount"] = self.global_stats["global_median_amount"]
            else:
                df["Transaction Amount"] = df["Transaction Amount"].fillna(
                    self.global_stats["global_median_amount"]
                )

            if "Quantity" not in df.columns or df["Quantity"].isna().all():
                df["Quantity"] = self.global_stats["global_median_quantity"]
            else:
                df["Quantity"] = df["Quantity"].fillna(
                    self.global_stats["global_median_quantity"]
                )

            if "Customer Age" not in df.columns or df["Customer Age"].isna().all():
                df["Customer Age"] = self.global_stats["global_median_age"]
            else:
                df["Customer Age"] = df["Customer Age"].fillna(
                    self.global_stats["global_median_age"]
                )

            # No customer-level aggregations!
            df["hours_since_prev_tx"] = -1

        else:
            # ⚠️ OLD APPROACH: Use customer aggregations (has data leakage)

            if "Customer ID" in df.columns and enabled_features.get("Customer ID", True):
                df["customer_tx_count"] = df["Customer ID"].map(
                    self.customer_agg["customer_tx_count"]
                ).fillna(0)
                df["customer_avg_amount"] = df["Customer ID"].map(
                    self.customer_agg["customer_avg_amount"]
                ).fillna(0)
                df["customer_std_amount"] = df["Customer ID"].map(
                    self.customer_agg["customer_std_amount"]
                ).fillna(0)
                df["customer_n_unique_ip"] = df["Customer ID"].map(
                    self.customer_agg["customer_n_unique_ip"]
                ).fillna(0)
            else:
                df["customer_tx_count"] = 0
                df["customer_avg_amount"] = 0
                df["customer_std_amount"] = 0
                df["customer_n_unique_ip"] = 0

            df["hours_since_prev_tx"] = -1

        # Time features
        if "Transaction Date" in df.columns and df["Transaction Date"].notna().any():
            df['Is Weekend'] = np.where(df['Transaction Date'].dt.dayofweek >= 5, 1, 0)
            df['Hour'] = df['Transaction Date'].dt.hour
            df['Day_of_Week'] = df['Transaction Date'].dt.dayofweek
            df['Day_of_Month'] = df['Transaction Date'].dt.day

            df['Is Weekend'] = df['Is Weekend'].fillna(0).astype(int)
            df['Hour'] = df['Hour'].fillna(12).astype(int)
            df['Day_of_Week'] = df['Day_of_Week'].fillna(0).astype(int)
            df['Day_of_Month'] = df['Day_of_Month'].fillna(15).astype(int)
        else:
            df['Is Weekend'] = 0
            df['Hour'] = 12
            df['Day_of_Week'] = 0
            df['Day_of_Month'] = 15

        # Address match
        if all(col in df.columns for col in ["Shipping Address", "Billing Address"]):
            df['Is Address Same'] = np.where(
                df["Shipping Address"] == df["Billing Address"], 1, 0
            )
        else:
            df['Is Address Same'] = 0

        # Transaction Size
        if "Transaction Amount" in df.columns:
            df['Transaction_Size'] = pd.cut(
                df['Transaction Amount'],
                bins=self.bins,
                labels=['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large']
            )
        else:
            df['Transaction_Size'] = 'Medium'

        df['Transaction_Size'] = df['Transaction_Size'].fillna('Medium')

        # Risk flags
        if "Transaction Amount" in df.columns:
            df['High_Amount_Flag'] = (
                df['Transaction Amount'] >= self.high_amount_threshold
            ).astype(int)
        else:
            df['High_Amount_Flag'] = 0

        if "Quantity" in df.columns:
            df['High_Quantity_Flag'] = (
                df['Quantity'] >= self.high_quantity_threshold
            ).astype(int)
        else:
            df['High_Quantity_Flag'] = 0

        # Ratio features
        if self.use_global_stats:
            # ✅ Use global statistics
            if "Transaction Amount" in df.columns:
                df["Amount_to_GlobalAvg_Ratio"] = (
                    df["Transaction Amount"] / self.global_stats["global_avg_amount"]
                )
                df["Amount_Global_Deviation"] = (
                    (df["Transaction Amount"] - self.global_stats["global_avg_amount"])
                    / self.global_stats["global_std_amount"]
                )

                # Clip
                df["Amount_to_GlobalAvg_Ratio"] = df["Amount_to_GlobalAvg_Ratio"].clip(0, 10)
                df["Amount_Global_Deviation"] = df["Amount_Global_Deviation"].clip(-5, 5)
            else:
                df["Amount_to_GlobalAvg_Ratio"] = 1.0
                df["Amount_Global_Deviation"] = 0.0
        else:
            # ⚠️ Use customer aggregations (old approach)
            if "Transaction Amount" in df.columns:
                df['Amount_to_AvgAmount_Ratio'] = df['Transaction Amount'] / (
                    df['customer_avg_amount'] + 1e-5
                )
                df['Amount_Deviation'] = (
                    df['Transaction Amount'] - df['customer_avg_amount']
                ) / (df['customer_std_amount'] + 1e-5)
            else:
                df['Amount_to_AvgAmount_Ratio'] = 1.0
                df['Amount_Deviation'] = 0.0

        # Drop columns
        cols_to_drop = [
            "Shipping Address", "Billing Address", "Transaction ID",
            "Customer ID", "IP Address", "Customer Location",
            "Transaction Date", "Is Fraudulent"
        ]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # ============================================
        # ENCODING
        # ============================================
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [col for col in df.columns if col not in cat_cols]

        encoded_dfs = []

        for col in cat_cols:
            if col not in df.columns:
                continue

            if col in self.encoders:
                encoder = self.encoders[col]

                if isinstance(encoder, OneHotEncoder):
                    safe_col = df[col].astype(str).fillna("MISSING")

                    encoded = pd.DataFrame(
                        encoder.transform(safe_col.to_frame()),
                        columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                        index=df.index
                    )

                else:  # LabelEncoder
                    safe_col = encoded[col].astype(str).fillna("MISSING")

                    try:
                        encoded[col] = encoder.transform(safe_col)
                    except ValueError:
                        encoded[col] = 0

            else:
                encoded = pd.DataFrame({col: 0}, index=df.index)

            encoded_dfs.append(encoded)

        # Combine
        if encoded_dfs:
            cats_encoded = pd.concat(encoded_dfs, axis=1)
            X = pd.concat([df[num_cols].reset_index(drop=True),
                           cats_encoded.reset_index(drop=True)], axis=1)
        else:
            X = df[num_cols].reset_index(drop=True)

        # Align with training features
        expected_tabular_cols = [col for col in self.feature_names
                                 if not col.startswith("gnn_emb_")]
        X = X.reindex(columns=expected_tabular_cols, fill_value=0)

        # ============================================
        # SCALING
        # ============================================
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        # ============================================
        # GNN EMBEDDINGS: USE MEAN / CENTROID for unseen nodes (preferred)
        # ============================================
        n_rows = len(X_scaled)
        gnn_embeddings = self._get_replacement_gnn(n_rows=n_rows)
        # ensure correct shape
        if gnn_embeddings.shape != (n_rows, self.n_gnn_features):
            # last-resort fix (should not happen)
            gnn_embeddings = np.resize(gnn_embeddings, (n_rows, self.n_gnn_features)).astype(np.float32)

        print(f"Using GNN replacement embeddings (shape: {gnn_embeddings.shape})")
        if self.mean_gnn_embedding is not None:
            print("   Using MEAN GNN embedding instead of zeros for unseen nodes")
        else:
            print("   WARNING: mean embedding not available; using zeros (risky)")

        # Concatenate
        X_final = np.concatenate([X_scaled.values, gnn_embeddings], axis=1)
        X_final_df = pd.DataFrame(X_final, columns=self.feature_names)

        return X_final_df


def load_best_model(artifacts_dir="fraud_detection_artifacts"):
    """Load the best model"""

    with open(os.path.join(artifacts_dir, "model_performance.json"), "r") as f:
        perf = json.load(f)
        best_model_name = perf["best_model"]

    model_path = os.path.join(artifacts_dir, f"model_{best_model_name}_BEST.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"✓ Loaded best model: {best_model_name}")

    # Show metrics
    if "best_f1" in perf:
        print(f"  F1: {perf['best_f1']:.4f}")
        print(f"  Precision: {perf['best_precision']:.4f}")
        print(f"  Recall: {perf['best_recall']:.4f}")
    else:
        print(f"  Recall: {perf.get('best_recall', 'N/A')}")

    return model, best_model_name, perf
