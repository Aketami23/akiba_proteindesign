#!/bin/bash
set -eux

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

uv python pin 3.10.16

uv add Cython setuptools numpy

wget https://github.com/openmm/openmm/archive/refs/tags/8.1.1.tar.gz
tar zxvf 8.1.1.tar.gz
cd openmm-8.1.1
mkdir build
cd build

# ===== 1. Doxygen =====
DOXYGEN_VER=1.9.8
if [ ! -d "doxygen-${DOXYGEN_VER}" ]; then
  wget -q https://www.doxygen.nl/files/doxygen-${DOXYGEN_VER}.linux.bin.tar.gz
  tar -xzf doxygen-${DOXYGEN_VER}.linux.bin.tar.gz
fi
export PATH="$PWD/doxygen-${DOXYGEN_VER}/bin:$PATH"

# ===== 2. SWIG =====
SWIG_VER=4.2.1
SWIG_PREFIX="${HOME}/.local/swig/${SWIG_VER}"
if [ ! -x "${SWIG_PREFIX}/bin/swig" ]; then
  wget -q https://github.com/swig/swig/archive/refs/tags/v${SWIG_VER}.tar.gz
  tar -xzf v${SWIG_VER}.tar.gz
  pushd swig-${SWIG_VER}
  ./autogen.sh
  ./configure --prefix="${SWIG_PREFIX}"
  make -j"$(nproc)" && make install
  popd
fi
export PATH="${SWIG_PREFIX}/bin:$PATH"

# ===== 3. OpenMM =====
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/OpenMMCWrapper.h"
#     "${CMAKE_CURRENT_BINARY_DIR}/OpenMMFortranModule.f90"
#     DESTINATION include
# )

cmake .. -DCMAKE_INSTALL_PREFIX="${HOME}/apps/openmm/8.1.1" -DPYTHON_EXECUTABLE="/path/to/your/.venv/bin/python3.10.16"
make -j8 instal

uv add numpy==1.26.4

make PythonInstall

wget https://github.com/openmm/pdbfixer/archive/refs/tags/1.9.tar.gz
tar zxvf 1.9.tar.gz
cd pdbfixer-1.9
python3.10 -m pip install .