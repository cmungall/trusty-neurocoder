# Running EcoSIM on NERSC Perlmutter

## Why NERSC

Building EcoSIM on macOS requires compiling HDF5 and NetCDF from source
via autotools, which hits multiple toolchain issues (cmake 4.x compat,
perl shebang in autoconf, missing `-lSystem` for gcc). On Perlmutter,
HDF5 and NetCDF are pre-installed as modules.

## Build Steps (Perlmutter)

```bash
# Login
ssh perlmutter.nersc.gov

# Clone with submodules
cd $SCRATCH
git clone --recursive https://github.com/jinyun1tang/EcoSIM.git
cd EcoSIM

# Load modules
module load PrgEnv-gnu
module load cray-hdf5
module load cray-netcdf

# Build using Cray compiler wrappers
# May need to add cmake_minimum_required bump and POLICY_VERSION_MINIMUM
./build_EcoSIM.sh CC=cc CXX=CC FC=ftn

# Or with explicit GNU compilers
module load gcc
./build_EcoSIM.sh CC=gcc CXX=g++ FC=gfortran
```

## Generate Training Data

Once built, run the bare_soil example to generate carbon pool trajectories:

```bash
cd examples/run_dir/bare_soil
../../../local/bin/ecosim.f90.x BareSoil.namelist

# Output will be in a NetCDF file
ncdump -h BareSoil.output.nc | head -30
```

## Extract Trajectories for Surrogate Training

```python
import netCDF4 as nc
import numpy as np

ds = nc.Dataset("BareSoil.output.nc")

# Look for soil carbon variables
for var in ds.variables:
    if "carbon" in var.lower() or "SOC" in var or "DOM" in var:
        print(var, ds[var].shape, ds[var].units)

# Extract time series for training
# Variables of interest (names TBD from actual output):
# - soil organic carbon pools
# - dissolved organic carbon
# - soil temperature
# - soil moisture / water potential
# - respiration flux (for mass balance verification)
```

## Alternative: Docker

Based on the CI workflow (ecosim-ci.yml), a working Dockerfile:

```dockerfile
FROM rockylinux/rockylinux:8.10

RUN dnf -qq -y update && \
    dnf -qq -y install gcc-gfortran gcc-c++ cmake which git \
    curl curl-devel autoconf automake libtool libxml2-devel

WORKDIR /app
COPY . .
RUN git submodule update --init --recursive
RUN bash build_EcoSIM.sh CC=$(which gcc) CXX=$(which g++) FC=$(which gfortran)
```

## What We Need from EcoSIM Output

For the Cajal surrogate training, we need time series of:

1. **State variables**: 4 solid OM pools (protein, carbohydrate, cellulose, lignin) + DOM
2. **Environmental drivers**: soil temperature (K), soil water potential (kPa)
3. **Fluxes**: decomposition rates, respiration (for mass balance verification)

Multiple trajectories at different environmental conditions (different
grid cells, different seasons, or different forcing scenarios) are needed
to train the separable f_temp × f_water response functions.

The bare_soil example runs 5 years at hourly timestep across 3 cells.
This should produce ~13,000 hourly snapshots per cell — more than enough
training data.
