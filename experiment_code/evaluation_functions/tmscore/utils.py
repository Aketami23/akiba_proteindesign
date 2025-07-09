import subprocess
import re

def calculate_tmscore(purpose_pdb: str, new_pdb: str) -> float:
    from config_utils import load_config
    _config = load_config("./config.yaml")
    usalign_path = _config["usalign_path"]
    tm_command = [
        usalign_path,
        purpose_pdb,
        new_pdb,
        "-TMscore",
        "1"
    ]
    tm_result = subprocess.run(tm_command, shell=False, capture_output=True, text=True)
    tm_match = re.search(r"TM-score=\s*([0-9.]+)", tm_result.stdout)
    tm_score = float(tm_match.group(1))
    return - tm_score # Return negative value for minimization