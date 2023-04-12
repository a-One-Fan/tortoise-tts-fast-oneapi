#!/bin/bash
set -x

# Original: https://github.com/intel/intel-extension-for-pytorch/blob/xpu-master/scripts/compile_bundle.sh

VER_LLVM="llvmorg-13.0.0"
VER_PYTORCH="v1.13.1"
VER_TORCHVISION="v0.14.1"
VER_TORCHAUDIO="v0.13.1"
VER_IPEX="xpu-master"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <DPCPPROOT> <MKLROOT> [AOT] [WHICHCOMPILE]"
    echo "DPCPPROOT and MKLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler and oneMKL respectively."
    echo "They are likely the following 2 paths: /opt/intel/oneapi/compiler/latest /opt/intel/oneapi/mkl/latest"
    echo "AOT is optional, should be the text string for environment variable USE_AOT_DEVLIST. ats-m150 is the string for an Arc A770."
	echo "WHICHCOMPILE is optional, specifying only one component to compile, to redo bad compiles; please only use it after trying to compile."
	echo "The components are: 0 = LLVM, 1 = PyTorch, 2 = TorchVision, 3 = TorchAudio, 4 = Intel Extension for Pytorch"
    exit 1
fi
DPCPP_ROOT=$1
ONEMKL_ROOT=$2
AOT=""
if [[ $# -ge 3 ]]; then
    AOT=$3
fi

DO_LLVM=1
DO_TORCH=1
DO_VISION=1
DO_AUDIO=1
DO_IPEX=1

SUCCESS_LLVM=1
SUCCESS_TORCH=1
SUCCESS_VISION=1
SUCCESS_AUDIO=1
SUCCESS_IPEX=1

if [[ $# -ge 4 ]]; then
    DO_LLVM=0
    DO_TORCH=0
    DO_VISION=0
    DO_AUDIO=0
    DO_IPEX=0
    if [[ $4 -eq 0 ]]; then
        DO_LLVM=1
        echo "Compiling only LLVM."
    elif [[ $4 -eq 1 ]]; then
        DO_TORCH=1
        echo "Compiling only PyTorch."
    elif [[ $4 -eq 2 ]]; then
        DO_VISION=1
        echo "Compiling only TorchVision."
    elif [[ $4 -eq 3 ]]; then
        DO_AUDIO=1
        echo "Compiling only TorchAudio."
    elif [[ $4 -eq 4 ]]; then
        DO_IPEX=1
        echo "Compiling only Intel Extension for Pytorch."
    else
        echo "Invalid component chosen to compile!"
        exit 1
    fi
    sleep 3
fi

	
# Check existance of DPCPP and ONEMKL environments
DPCPP_ENV=${DPCPP_ROOT}/env/vars.sh
if [ ! -f ${DPCPP_ENV} ]; then
    echo "DPC++ compiler environment ${DPCPP_ENV} doesn't seem to exist."
    exit 2
fi
ONEMKL_ENV=${ONEMKL_ROOT}/env/vars.sh
if [ ! -f ${ONEMKL_ENV} ]; then
    echo "oneMKL environment ${ONEMKL_ENV} doesn't seem to exist."
    exit 3
fi

# Check existance of required Linux commands
which python > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"python\" not found."
    exit 4
fi
which git > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"git\" not found."
    exit 5
fi
which patch > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"patch\" not found."
    exit 6
fi
which pkg-config > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"pkg-config\" not found."
    exit 7
fi
env | grep CONDA_PREFIX > /dev/null 2>&1
CONDA=$?

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}

# Checkout individual components
if [ ! -d llvm-project ]; then
    git clone https://github.com/llvm/llvm-project.git
fi
if [ ! -d pytorch ]; then
    git clone https://github.com/pytorch/pytorch.git
fi
if [ ! -d vision ]; then
    git clone https://github.com/pytorch/vision.git
fi
if [ ! -d audio ]; then
    git clone https://github.com/pytorch/audio.git
fi
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git
fi

# Checkout required branch/commit and update submodules
cd llvm-project
git checkout ${VER_LLVM}
git submodule sync
git submodule update --init --recursive
cd ../pytorch
git checkout ${VER_PYTORCH}
git submodule sync
git submodule update --init --recursive
cd ../vision
git checkout ${VER_TORCHVISION}
git submodule sync
git submodule update --init --recursive
cd ../audio
git checkout ${VER_TORCHAUDIO}
git submodule sync
git submodule update --init --recursive
cd ../intel-extension-for-pytorch
git checkout ${VER_IPEX}
git submodule sync
git submodule update --init --recursive

# Install the basic dependency cmake
python -m pip install cmake

# Compile individual component
#  LLVM
cd ../llvm-project
git config --global --add safe.directory `pwd`
if [[ DO_LLVM -eq 1 ]]; then
    if [ -d build ]; then
        rm -rf build
    fi
    mkdir build
    cd build
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF ../llvm/
    if [[ $? -ne 0 ]]; then
        SUCCESS_LLVM=0
    fi
    cmake --build . -j $(nproc)
    if [[ $? -ne 0 ]]; then
        SUCCESS_LLVM=0
    fi
else
    cd build
fi
LLVM_ROOT=`pwd`/../release
if [[ DO_LLVM -eq 1 ]]; then
    if [ -d ${LLVM_ROOT} ]; then
        rm -rf ${LLVM_ROOT}
    fi
    cmake -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT}/../release/ -P cmake_install.cmake
    if [[ $? -ne 0 ]]; then
        SUCCESS_LLVM=0
    fi
    #xargs rm -rf < install_manifest.txt
    ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
fi
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
cd ..
git config --global --unset safe.directory

#  PyTorch
cd ../pytorch
git config --global --add safe.directory `pwd`
if [[ DO_TORCH -eq 1 ]]; then
    git stash
    git clean -f
    git apply ../intel-extension-for-pytorch/torch_patches/*.patch
    python -m pip install astunparse numpy ninja pyyaml mkl-static mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
fi
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
if [[ ${CONDA} -eq 0 ]]; then
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
else
    export CMAKE_PREFIX_PATH=${VIRTUAL_ENV:-"$(dirname $(which python))/../"}
fi
export USE_STATIC_MKL=1
export _GLIBCXX_USE_CXX11_ABI=1
export USE_NUMA=0
export USE_CUDA=0

if [[ DO_TORCH -eq 1 ]]; then
    python setup.py clean
    python setup.py bdist_wheel 2>&1 | tee build.log
fi
unset USE_CUDA
unset USE_NUMA
unset _GLIBCXX_USE_CXX11_ABI
unset USE_STATIC_MKL
unset CMAKE_PREFIX_PATH
unset LLVM_DIR
unset USE_LLVM
if [[ DO_TORCH -eq 1 ]]; then
    python -m pip uninstall -y mkl-static mkl-include
    python -m pip install --force-reinstall dist/*.whl
    if [[ $? -ne 0 ]]; then
        SUCCESS_TORCH=0
    fi
fi
git config --global --unset safe.directory
#  TorchVision
cd ../vision
git config --global --add safe.directory `pwd`
if [[ DO_VISION -eq 1 ]]; then
    conda install -y libpng jpeg
    python setup.py clean
    python setup.py bdist_wheel 2>&1 | tee build.log
    python -m pip install --force-reinstall --no-deps dist/*.whl
    if [[ $? -ne 0 ]]; then
        SUCCESS_VISION=0
    fi
    python -m pip install Pillow
fi
git config --global --unset safe.directory
#  TorchAudio
cd ../audio
git config --global --add safe.directory `pwd`
source ${DPCPP_ENV}
source ${ONEMKL_ENV}
if [[ DO_AUDIO -eq 1 ]]; then
    conda install -y bzip2
    python -m pip install -r requirements.txt
    python setup.py clean
    python setup.py bdist_wheel 2>&1 | tee build.log
    python -m pip install --force-reinstall --no-deps dist/*.whl
    if [[ $? -ne 0 ]]; then
        SUCCESS_AUDIO=0
    fi
fi
git config --global --unset safe.directory
#  Intel® Extension for PyTorch*
cd ../intel-extension-for-pytorch
git config --global --add safe.directory `pwd`
if [[ DO_IPEX -eq 1 ]]; then
    python -m pip install -r requirements.txt
fi
if [[ ! ${AOT} == "" ]]; then
    export USE_AOT_DEVLIST=${AOT}
fi
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
if [[ DO_IPEX -eq 1 ]]; then
    python setup.py clean
    python setup.py bdist_wheel 2>&1 | tee build.log
fi
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
if [[ ! ${AOT} == "" ]]; then
    unset USE_AOT_DEVLIST
fi
if [[ DO_IPEX -eq 1 ]]; then
    python -m pip install --force-reinstall dist/*.whl
    if [[ $? -ne 0 ]]; then
        SUCCESS_IPEX=0
    fi
fi
git config --global --unset safe.directory

if [[ SUCCESS_LLVM -eq 0 ]]; then
    echo "LLVM did not compile successfully!"
fi
if [[ SUCCESS_TORCH -eq 0 ]]; then
    echo "PyTorch did not compile successfully!"
fi
if [[ SUCCESS_VISION -eq 0 ]]; then
    echo "TorchVision did not compile successfully!"
fi
if [[ SUCCESS_AUDIO -eq 0 ]]; then
    echo "TorchAudio did not compile successfully!"
fi
if [[ SUCCESS_IPEX -eq 0 ]]; then
    echo "Intel Extension for Pytorch did not compile successfully!"
fi

# Sanity Test
cd ..
python -c "import torch; import torchvision; import torchaudio; import intel_extension_for_pytorch as ipex; print(f'torch_cxx11_abi:     {torch.compiled_with_cxx11_abi()}'); print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'torchaudio_version:  {torchaudio.__version__}'); print(f'ipex_version:        {ipex.__version__}');"
