# Reproducible script to build and prepare environment
# However, the installation is based on previously installed miniconda.
# Assuming that, under my aliases
# conda_base ... then bash install.sh

# Exit immediately if a command exits with a non-zero status
set -e

REPO=$(pwd)

# modify the installation path and env name if you want
INSTALLDIR=${WRKSPC}

export UV_CACHE_DIR=$INSTALLDIR/.cache/uv

ENV_NAME="frontier_uv_adsfp"

cd ${INSTALLDIR}

echo "Conda Version:" 
conda env list | grep '*'

# Create conda environment, and print whether it is loaded correctly
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.12 --yes -c defaults
source activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Conda packages:
conda install -c conda-forge conda-pack libstdcxx-ng --yes

# Load modules
rocm_version=6.4.2
libfabric_path=/opt/cray/libfabric/1.22.0

module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
# module swap PrgEnv-cray PrgEnv-gnu
module swap PrgEnv-gnu

module list

######### COMPILE UV PACKAGES ########################

cd "${REPO}"

# since I'm normally used to caching and installing envs under $WRKSPC, a specific nfs drive,
# uv gets mad bc it tries to install the venv inside the repo and these are cross filesystems. eg.:
# warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
# If the cache and target directories are on different filesystems, hardlinking may not be supported.
# If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.

# Sean's reloacatable version is sort of like this:
# uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy ${INSTALLDIR}/${ENV_NAME}/.venv
# cd ${INSTALLDIR}/$ENV_NAME
# source .venv/bin/activate

# But for now we can just try and get everything, code and env, on the same filesystem.

pip install uv

# If you run this line, or a sync command standalone, remember to set the cache dir like above
# coudl specify the venv directory, but not doing so currently.
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy --no-cache-dir --index-strategy unsafe-best-match
uv sync --link-mode=copy --no-cache-dir --index-strategy unsafe-best-match

source .venv/bin/activate
uv pip list

# note that since uv is installed in conda, need to activate that first

######### COMPILE AWS OFI RCCL PLUGIN ########################

cd ${INSTALLDIR}

# Download the plugin repo
git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl aws-ofi-rccl_$ENV_NAME
cd aws-ofi-rccl_$ENV_NAME

# Build the plugin
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/hip/lib:$LD_LIBRARY_PATH
PLUG_PREFIX=$PWD

CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
--with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version/lib --enable-trace \
--prefix=$PLUG_PREFIX --with-hip=/opt/rocm-$rocm_version/lib --with-mpi=$MPICH_DIR

make
make install

# Reminder to export the plugin to your path
echo $PLUG_PREFIX
echo "Add the following line in the environment to use the AWS OFI RCCL plugin"
echo "export LD_LIBRARY_PATH="$PLUG_PREFIX"/lib:$""LD_LIBRARY_PATH"

cd ${INSTALLDIR}


cd ${REPO}

# Final messages
echo Installation finished. To use the environment, run: 
echo "conda_activate ${INSTALLDIR}/$ENV_NAME" 
echo and then from inside $REPO, run:
echo '"source .venv/bin/activate && python -u train.py ... or whatever the target python script is."'
echo "Remember to set UV_CACHE_DIR=${UV_CACHE_DIR} before running any uv commands."
