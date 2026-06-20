from pathlib import Path

import pandas as pd


RESULT_COLUMNS = ["filename", "prediction"]


def prediction_label(prediction: str) -> str:
    if prediction == "No confident match":
        return prediction
    return Path(prediction).stem


def batch_results_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def results_csv_bytes(results: pd.DataFrame) -> bytes:
    table = results.loc[:, RESULT_COLUMNS]
    return table.to_csv(index=False, lineterminator="\n").encode("utf-8")
