DUMMY_INPUT="/gs/bs/tga-cddlab/akiba/simulated-annealing_seq_top7/input/20250219/initial-1qys-20250219.fasta"
OUTPUT_DIR="output"
RESULT_CSV="output/results.csv"
YAML_CONFIG="config.yaml"
python3.10 main.py\
    "$DUMMY_INPUT" \
    "$OUTPUT_DIR" \
    "$RESULT_CSV" \
    "$YAML_CONFIG"