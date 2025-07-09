#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=10:00:00

DATE=$(date +"%Y%m%d")

echo "$DATE"

BASE_OUTPUT="/gs/bs/tga-cddlab/akiba/mpnn/output/${DATE}"

if [ ! -d "$BASE_OUTPUT" ]; then
    mkdir "$BASE_OUTPUT"
fi

max_part=$(find "$BASE_OUTPUT" -maxdepth 1 -type d -name "part*" \
  | sed -E 's|.*/part([0-9]+)$|\1|' \
  | sort -n \
  | tail -n 1)

if [ -z "$max_part" ]; then
  max_part=1
else
  max_part=$((max_part + 1))
fi

PART="part${max_part}"
OUTPUT_DIR="${BASE_OUTPUT}/${PART}"
echo "$PART"
RESULT_CSV="${BASE_OUTPUT}/${DATE}-${PART}-results.csv"

mkdir -p "$OUTPUT_DIR"
touch "$RESULT_CSV"

echo "negative_tm_score,recovery,negative_plddt,raw_jobname,query_sequence" > "$RESULT_CSV"

# pySOL_DIR="/gs/bs/tga-cddlab/akiba/nsga/protein-sol-sequence-prediction-software"

# cp -r "$pySOL_DIR" "$OUTPUT_DIR"

DUMMY_INPUT="/gs/bs/tga-cddlab/akiba/simulated-annealing_seq_top7/input/20250219/initial-1qys-20250219.fasta"

# conda activate /gs/fs/tga-cddlab/akiba/apps/localcolabfold/colabfold-conda

RUN_SCRIPT="/gs/bs/tga-cddlab/akiba/protein_design/experiment_code/main.py"

YAML_CONFIG="/gs/bs/tga-cddlab/akiba/protein_design/experiment_code/config.yaml"

python "$RUN_SCRIPT"\
    "$DUMMY_INPUT" \
    "$OUTPUT_DIR" \
    "$RESULT_CSV" \
    "$YAML_CONFIG" \

conda deactivate