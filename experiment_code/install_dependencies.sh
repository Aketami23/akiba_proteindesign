#!/bin/bash
set -eux 

source .venv/bin/activate

pip install --upgrade pip
pip install Cython setuptools

# 1. Doxygen
DOXYGEN_VER=1.9.8
if [ ! -d "doxygen-${DOXYGEN_VER}" ]; then
  wget -q https://www.doxygen.nl/files/doxygen-${DOXYGEN_VER}.linux.bin.tar.gz
  tar -xzf doxygen-${DOXYGEN_VER}.linux.bin.tar.gz
fi
export PATH=$PWD/doxygen-${DOXYGEN_VER}/bin:$PATH

# 2. SWIG
SWIG_VER=4.2.1
if [ ! -x "$HOME/apps/swig/${SWIG_VER}/bin/swig" ]; then
  wget -q https://github.com/swig/swig/archive/refs/tags/v${SWIG_VER}.tar.gz
  tar -xzf v${SWIG_VER}.tar.gz
  cd swig-${SWIG_VER}
  ./autogen.sh
  ./configure --prefix=$HOME/apps/swig/${SWIG_VER}
  make -j$(nproc) && make install
  cd ..
fi
export PATH=$HOME/apps/swig/${SWIG_VER}/bin:$PATH

# 3. OpenMM
OPENMM_VER=8.1.1
if [ ! -d "openmm-${OPENMM_VER}" ]; then
  wget -q https://github.com/openmm/openmm/archive/refs/tags/${OPENMM_VER}.tar.gz
  tar -xzf ${OPENMM_VER}.tar.gz
fi

cd openmm-${OPENMM_VER}
rm -rf build
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/apps/openmm/${OPENMM_VER} \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DDOXYGEN_EXECUTABLE=$(which doxygen) \
  -DSWIG_EXECUTABLE=$(which swig)

make -j$(nproc) install
make PythonInstall

cd ../..
