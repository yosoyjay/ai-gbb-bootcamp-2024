# Benchmark OPT

This document describes the steps to prepare an environment and to benchmark the training of an OPT model on a Slurm cluster deployed on Azure.

These directions assume that a a Slurm cluster with the appropriate VM (e.g. Standard_ND96amsr_A100_v4) has already been provisioned (See [README.md](../README.md) for details).

# Preparing the environment

A couple of steps require a CUDA environment so make sure you're on a compute node:

```bash
[login] srun --pty bash -i
[compute]
```

## Step 1. Create Python environment

This benchmark is run on bare metal in a conda virtual environment following the instructions in the [Metaseq README](https://github.com/facebookresearch/metaseq/blob/main/docs/setup.md).

Install Python enviroment for this purpose using miniconda:

```bash
[compute] curl -fsO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
[compute] Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
[compute] $HOME/miniconda/bin/conda init
```

Create and activate conda environment for environmental isolation, but everything will be installed via `pip`:

```bash
[compute] source $HOME/.bashrc
[compute] conda create -y --name metaseq python=3.9
[compute] conda activate metaseq
```

```bash
[compute] pip install --pre torch --index-url https://download.pytorch.org/whl/cu121
```

## Step 2. Install NVIDIA Apex to enable training optimizations

Install the Apex extension to PyTorch to enable mixed precision and distributed training optimizations.

In some cases, as in this case, the CUDA version on the VM is one minor version off from one PyTroch is compiled with, so must disable a check in the Apex setup script.


```bash
[compute] git clone https://github.com/NVIDIA/apex
[compute pushd apex
[compute] sed -i "s/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)//g" setup.py
[compute] pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
[comptue] popd
```

## Step 3. Install Megatron

Install Megatron fork as specified in the aforementioned README.

```bash
[compute] git clone -b fairqeq_v3 https://github.com/ngoyal2707/Megatron-LM.git
[compute] pushd Megatron-LM
[compute] pip install -e .
[compute] popd
```

## Step 4. Install Metaseq

```bash
[compute] git clone https://github.com/facebookresearch/metaseq.git
[compute] pushd metaseq
[compute] python setup.py build_ext --inplace
[compute] pip install -e .
[compure] popd
```

## Step 5. Install Fairscale

Note, this install via pip is not editable (i.e. no `-e`) as the `metaseq/train.py` checks the `fairscale` version which will not be defined if installed in editable mode.

```bash
[compute] git clone https://github.com/facebookresearch/fairscale.git
[compute] pushd fairscale
[compute] git checkout fixing_memory_issues_with_keeping_overlap
[compute] pip install .
[compute] popd
```


## Step 6. Run benchmark with synthetic data

Ensure that environmental variables are properly set for optimal performance:

```bash
[compute] source nccl-env-var.sh
```

This '--local' flag will run the job without Slurm. We specify a 125M parameter model to fit in memory.  The test will take about 2.5 minutes on a single NDv4.
There will be a lot of output, you can ignore the NCCL warnings.  The important part is the WPS (words per second) which should be at least 200K.

```bash
[compute] time opt-baselines --model-size 125m --benchmark -t 1 -g 8 -n 1 -p test-125m-local --local --azure
```

Now we'll run the same job on the Slurm cluster.  We'll use the same model size and number of GPUs as before.  The job will take about 2.5 minutes on a single NDv4.

Back to the login node:

```bash
[compute] exit
[login] sbatch
```

Submit the job to the scheduler, notes the lack of `--local` flag below.  The will generate a Slurm script to run the same job on the cluster and submit it to the scheduler.
The output for the job will be listed in the output of the output from the command.  Look for the `--output` field.

```bash
[login] opt-baselines --model-size 125m --benchmark -t 1 -g 8 -n 1 -p test-125m-slurm-01 --azure
```

Now change the command to run with two nodes, note the name change for the job `-p test-125m-slurm-02`.  If you don't change that the job will not run because it will see that a job with that name has already run to completion:

```bash
[login] opt-baselines --model-size 125m --benchmark -t 1 -g 16 -n 1 -p test-125m-slurm-02 --azure
```

🎉🎉🎉 You've validated the cluster works as expected.