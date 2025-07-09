import csv
import pandas as pd

def write_csv(query: any, negative_tm_score: float, negative_plddt_score: float, recovery_score: float, csv_path: str)-> None:
    raw_jobname, query_sequence, a3m_lines = query

    with open(csv_path, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([negative_tm_score, recovery_score, negative_plddt_score, raw_jobname, query_sequence])

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
