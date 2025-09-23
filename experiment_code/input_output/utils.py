import csv
import os
from typing import Any, Dict

import pandas as pd


def write_csv(
    query: Any,
    negative_tm_score: float,
    negative_plddt_score: float,
    recovery_score: float,
    csv_path: str,
    metadata: Dict[str, Any],
) -> None:
    raw_jobname, query_sequence, _ = query

    file_exists = os.path.isfile(csv_path)
    needs_header = not file_exists or os.path.getsize(csv_path) == 0

    parent_ids = metadata.get("parent_ids", []) or []
    parent_ids_str = "|".join(str(pid) for pid in parent_ids)

    with open(csv_path, mode="a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if needs_header:
            csv_writer.writerow(
                [
                    "id",
                    "parent_ids",
                    "generation",
                    "negative_tm_score",
                    "recovery",
                    "negative_plddt_score",
                    "raw_jobname",
                    "query_sequence",
                ]
            )
        csv_writer.writerow(
            [
                metadata.get("id"),
                parent_ids_str,
                metadata.get("generation"),
                negative_tm_score,
                recovery_score,
                negative_plddt_score,
                raw_jobname,
                query_sequence,
            ]
        )

def get_column_values(output_csv_path: int, query_list:list[any] , column_name: str) -> list[any]:
    df = pd.read_csv(output_csv_path)
    values = []

    for query in query_list:
        matched_rows = df[df['query_sequence'] == query]
        if matched_rows.empty:
            values.append(None)
        else:
            values.append(matched_rows.iloc[0][column_name])
    
    return values
