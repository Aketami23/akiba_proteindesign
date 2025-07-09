import numpy as np
import json

def calculate_plddt(json_file_path: str) -> float:
    with open(json_file_path, 'r') as f:
        scores = json.load(f)
    
    if "plddt" in scores:
        plddt_score = float(np.mean(scores["plddt"]))

        return - plddt_score  # Return negative value for minimization
    return None