#!/bin/bash
set -euxo pipefail

# =========[ 1. Initialize and Create Virtual Environment with uv ]=========
uv init
uv venv
source .venv/bin/activate

# =========[ 2. Install Base Dependencies ]=========
source install_dependencies.sh

# =========[ 3. Add ColabFold via Git URL ]=========
COLABFOLD_COMMIT="406d4c6cf25a0755f61b3adac7c5d47d3025f42c"
COLABFOLD_REPO="git+https://github.com/sokrypton/ColabFold@${COLABFOLD_COMMIT}#egg=colabfold"

uv add "${COLABFOLD_REPO}"

# Sometimes necessary to also install with pip directly
uv pip install "${COLABFOLD_REPO}"

# =========[ 4. Install Additional Dependencies from requirements.txt ]=========
uv add -r requirements.txt
