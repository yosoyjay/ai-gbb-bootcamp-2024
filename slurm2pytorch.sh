#!/bin/bash

# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


###########################################################################
# Pytorch multi-node jobs require a bunch of non-standard envvars to be set.
# This script derives and sets reasonable value for those variables from SLURM.
###########################################################################

set -euo pipefail

# only if we're in a pytorch container:
if [[ "${PYTORCH_VERSION-}" ]]; then

    # MLPERF_SLURM_FIRSTNODE should be set from host scripts.  Setting it
    # requires `scontrol` which is typically unavailable from inside
    # containers.  a typical way to set it from host would be something like:
    # $(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)
    if [[ "${MLPERF_SLURM_FIRSTNODE-}" ]] && [[ ! "${MASTER_ADDR-}" ]]; then
	export MASTER_ADDR="${MLPERF_SLURM_FIRSTNODE}"
    fi

    if [[ "${SLURM_JOB_ID-}" ]] && [[ ! "${MASTER_PORT-}" ]]; then
	export MASTER_PORT="$((${SLURM_JOB_ID} % 16384 + 49152))"
    fi
    if [[ "${SLURM_NTASKS-}" ]] && [[ ! "${WORLD_SIZE-}" ]]; then
	export WORLD_SIZE="${SLURM_NTASKS}"
    fi
    if [[ "${SLURM_PROCID-}" ]] && [[ ! "${RANK-}" ]]; then
	export RANK="${SLURM_PROCID}"
    fi
    if [[ "${SLURM_LOCALID-}" ]] && [[ ! "${LOCAL_RANK-}" ]]; then
	export LOCAL_RANK="${SLURM_LOCALID}"
    fi

    # Match the behavior of torch.distributed.run:
    # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/distributed/run.py#L521-L532
    if [[ "${SLURM_NTASKS_PER_NODE:-1}" -gt 1 ]] && [[ ! "${OMP_NUM_THREADS-}" ]]; then
	export OMP_NUM_THREADS=1
    fi

fi # end if [[ "${PYTORCH_VERSION-}" ]]

#############################################################################
# Exec the child script with the new variables
#############################################################################

exec "${@}"