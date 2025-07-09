import subprocess
import os
import re

# めっちゃハードコーディング
def calculate_sol(seq:str, result_dir:str) -> float:
    sequences = [{'id': 'seq1', 'seq': seq}]
    fasta_filename = os.path.join(result_dir, "protein-sol-sequence-prediction-software", "tmp.fasta")
    with open(fasta_filename, "w") as fasta_file:
        for entry in sequences:
            fasta_file.write(">" + entry['id'] + "\n")
            fasta_file.write(entry['seq'] + "\n")
    script_path = os.path.join(result_dir, "protein-sol-sequence-prediction-software", "multiple_prediction_wrapper_export.sh")
    subprocess.check_call([script_path, fasta_filename, result_dir])
    output_file = os.path.join(result_dir, "protein-sol-sequence-prediction-software", "seq_prediction.txt")
    with open(output_file, "r") as f:
            results = f.read()
    match = re.search(r">seq1,[^,]+,\s*([\d.]+)", results)
    os.remove(output_file)
    solubility_score = float(match.group(1))
    return - solubility_score # Return negative value for minimization