DUMMY_INPUT="./data/dummy.fasta"
mkdir output
OUTPUT_DIR="output"
RESULT_CSV="output/results.csv"
echo "id","parent_ids","generation","negative_tm_score","recovery","negative_plddt_score","raw_jobname","query_sequence", > "$RESULT_CSV"
YAML_CONFIG="config.yaml"
python3.10 main.py\
    "$DUMMY_INPUT" \
    "$OUTPUT_DIR" \
    "$RESULT_CSV" \
    "$YAML_CONFIG"