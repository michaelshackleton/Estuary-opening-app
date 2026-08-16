"""
Turns the per-scene records produced by fetch.run_site_analysis() into a
clean time series: one row per date (preferring the finer-resolution
Sentinel-2 result when both Landsat and Sentinel-2 are available for the
same day), plus the mean-monthly proportion-closed statistic described in
the R script (equal-weighted across calendar months, not across images, so
that periods with more frequent imagery - e.g. post-2016 with Sentinel-2 -
don't dominate the estimate).
"""

from __future__ import annotations

import pandas as pd

SENSOR_RANK = {"sentinel2": 0, "landsat": 1}


def build_results_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if len(df) == 0:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "sensor"]).reset_index(drop=True)


def prefer_sentinel_on_shared_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Where both a Landsat and a Sentinel-2 result exist for the same
    calendar date, keep only the Sentinel-2 row (finer resolution).
    Error rows are dropped here too, since they carry no usable status."""
    if len(df) == 0:
        return df
    working = df[df["status"] != "error"].copy()
    working["sensor_rank"] = working["sensor"].map(SENSOR_RANK).fillna(99)
    working = working.sort_values(["date", "sensor_rank"])
    deduped = working.drop_duplicates(subset="date", keep="first")
    return deduped.drop(columns="sensor_rank").sort_values("date").reset_index(drop=True)


def mean_monthly_proportion_closed(df: pd.DataFrame, status_col: str = "status") -> float | None:
    """Equal-weighted mean, across calendar months, of that month's
    proportion of open/closed observations that were closed. Indeterminate
    and error rows are excluded (matches the R script's `binned` summary).

    `status_col` lets the caller compute this against the combined status
    (default), or against just `status_ndwi`/`status_fmask` (values "TRUE"/
    "FALSE"/"indeterminate" rather than "open"/"closed"/"indeterminate") if
    they want the statistic for a single method in isolation."""
    if len(df) == 0:
        return None
    open_vals = {"open", "TRUE"}
    closed_vals = {"closed", "FALSE"}
    valid = df[df[status_col].isin(open_vals | closed_vals)].copy()
    if len(valid) == 0:
        return None
    valid["month"] = valid["date"].dt.to_period("M")
    valid["is_closed"] = valid[status_col].isin(closed_vals)
    monthly = valid.groupby("month")["is_closed"].mean()
    return float(monthly.mean())


def summary_counts(df: pd.DataFrame, status_col: str = "status") -> dict:
    """Simple counts for a quick-glance summary panel in the app. See
    mean_monthly_proportion_closed() for what `status_col` is for."""
    if len(df) == 0:
        return dict(n_open=0, n_closed=0, n_indeterminate=0, n_error=0, n_total=0)
    counts = df[status_col].value_counts()
    return dict(
        n_open=int(counts.get("open", 0) + counts.get("TRUE", 0)),
        n_closed=int(counts.get("closed", 0) + counts.get("FALSE", 0)),
        n_indeterminate=int(counts.get("indeterminate", 0)),
        n_error=int(counts.get("error", 0)),
        n_total=len(df),
    )
