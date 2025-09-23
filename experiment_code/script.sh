DUMMY_INPUT="./data/dummy.fasta"
mkdir output
OUTPUT_DIR="output"
RESULT_CSV="output/results.csv"
YAML_CONFIG="config.yaml"
python3.10 main.py\
    "$DUMMY_INPUT" \
    "$OUTPUT_DIR" \
    "$RESULT_CSV" \
    "$YAML_CONFIG"