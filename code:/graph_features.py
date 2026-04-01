import pandas as pd
import numpy as np

def compute_graph_features(df):

    # Simulate children count using mentions + hashtags (proxy)
    df["num_children"] = df["mentions"] + df["hashtags"]

    # Simulate depth using text length variation
    df["cascade_depth"] = df["length"] // 50

    return df


def compute_temporal_features(df):

    df = df.sort_values("timestamp")

    # Real variation instead of simple diff
    df["time_diff"] = df["timestamp"].diff().fillna(0)

    # Add variability signal
    df["avg_time_gap"] = df["time_diff"].rolling(5).mean().fillna(0)

    # 🔥 NEW IMPORTANT FEATURE
    df["burstiness"] = df["time_diff"].rolling(5).std().fillna(0)

    return df