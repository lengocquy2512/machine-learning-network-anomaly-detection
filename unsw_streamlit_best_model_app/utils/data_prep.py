from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


DROP_COLS_DEFAULT = [
    # your notebook / training
    "attack_cat",
    "id",
    "response_body_len",
    "ct_flw_http_mthd",
    "trans_depth",
    "dwin",
    "ct_ftp_cmd",
    "is_ftp_login",
]


KEEP_STATE = ["FIN", "INT", "CON", "REQ", "RST"]
KEEP_SERVICE = ["-", "dns", "http", "smtp", "ftp", "ftp-data", "ssh", "pop3"]
KEEP_PROTO = ["tcp", "udp", "arp", "ospf", "igmp_icmp_rtp"]


@dataclass
class Prepared:
    X: pd.DataFrame
    y: Optional[pd.Series]
    dropped_cols: list[str]


def _cap_outliers_like_notebook(df: pd.DataFrame, q: float = 0.95, ratio: float = 10.0, min_max: float = 10.0) -> pd.DataFrame:
    """Implements your clamping rule:
    if max(col) > ratio*median(col) AND max(col) > min_max:
        cap values to quantile(q)
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns

    for c in num_cols:
        s = out[c].astype(float)
        med = float(np.nanmedian(s.values))
        mx = float(np.nanmax(s.values))
        if (mx > (ratio * med)) and (mx > min_max):
            cap = float(np.nanquantile(s.values, q))
            out[c] = np.where(s < cap, s, cap)
    return out


def _group_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Match your feature-engineering step for proto/service/state."""
    out = df.copy()

    # state / service
    if "state" in out.columns:
        out.loc[~out["state"].isin(KEEP_STATE), "state"] = "others"
    if "service" in out.columns:
        out.loc[~out["service"].isin(KEEP_SERVICE), "service"] = "others"

    # proto (special merge for igmp/icmp/rtp)
    if "proto" in out.columns:
        # normalize strings
        out["proto"] = out["proto"].astype(str).str.lower()
        out.loc[out["proto"].isin(["igmp", "icmp", "rtp"]), "proto"] = "igmp_icmp_rtp"
        out.loc[~out["proto"].isin(KEEP_PROTO), "proto"] = "others"

    return out


def prepare_for_best_model(
    df_raw: pd.DataFrame,
    feature_ref: Sequence[str],
    drop_cols: Sequence[str] = DROP_COLS_DEFAULT,
    auto_preprocess: bool = True,
) -> Prepared:
    """Take a RAW UNSW CSV and return X aligned to training features."""
    df = df_raw.copy()

    # Separate label if present
    y = None
    if "label" in df.columns:
        y = df["label"].astype(int)
        df = df.drop(columns=["label"])

    # Drop columns (safe)
    to_drop = [c for c in drop_cols if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    if auto_preprocess:
        # feature engineering on categoricals
        df = _group_rare_categories(df)
        # clamp numeric outliers
        df = _cap_outliers_like_notebook(df)

        # one-hot
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols)

    # ensure float32 for model
    X = df.astype(np.float32, errors="ignore")

    # align to training columns (drop extra, add missing 0)
    X = X.reindex(columns=list(feature_ref), fill_value=0.0).astype(np.float32)

    return Prepared(X=X, y=y, dropped_cols=to_drop)
