#!/bin/bash
set -eux

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

pip install --upgrade pip
pip install Cython setuptools numpy

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
OPENMM_VER=8.1.1
if [ ! -d "openmm-${OPENMM_VER}" ]; then
  wget -q https://github.com/openmm/openmm/archive/refs/tags/${OPENMM_VER}.tar.gz -O openmm-${OPENMM_VER}.tar.gz
  tar -xzf openmm-${OPENMM_VER}.tar.gz
fi

SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')

pushd openmm-${OPENMM_VER}
rm -rf build
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX="${HOME}/.local/openmm/${OPENMM_VER}" \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DDOXYGEN_EXECUTABLE="$(which doxygen)" \
  -DSWIG_EXECUTABLE="$(which swig)" \
  -DPYTHON_INSTALL_PREFIX="${SITE_PACKAGES}" \
  -DOPENMM_GENERATE_API_DOCS=OFF

make -j"$(nproc)" install
make PythonInstall

popd

python -m openmm.testInstallation
