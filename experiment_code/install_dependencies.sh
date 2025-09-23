#!/bin/bash
set -e

# 1. Doxygen
wget -q https://www.doxygen.nl/files/doxygen-1.9.8.linux.bin.tar.gz
tar -xzf doxygen-1.9.8.linux.bin.tar.gz
export PATH=$PWD/doxygen-1.9.8/bin:$PATH

# 2. SWIG
wget -q https://github.com/swig/swig/archive/refs/tags/v4.2.1.tar.gz
tar -xzf v4.2.1.tar.gz
cd swig-4.2.1
./autogen.sh
./configure --prefix=$HOME/apps/swig/4.2.1
make -j8 && make install
export PATH=$HOME/apps/swig/4.2.1/bin:$PATH
cd ..

# 3. OpenMM
wget -q https://github.com/openmm/openmm/archive/refs/tags/8.1.1.tar.gz
tar -xzf 8.1.1.tar.gz
cd openmm-8.1.1
rm -rf build
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/apps/openmm/8.1.1 \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DDOXYGEN_EXECUTABLE=$(which doxygen) \
  -DSWIG_EXECUTABLE=$(which swig)

make -j8 install
make PythonInstall

cd ../..