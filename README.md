# AI GBB Bootcamp: Slurm HPC cluster hands-on

This tutorial will walk through the deployment of a Slurm HPC cluster on Azure using CycleCloud.

The tutorial will cover the following topics:
- Deployment of a CycleCloud Slurm cluster using [ccslurm4ai](https://github.com/yosoyjay/ccslurm4ai.git)
- Verification that the Slurm cluster is operational and performing as expected
    - Manually running Node Health Checks from azurehpc VM image on all the nodes
    - NCCL all reduce to verify GPU and Infiniband performance
    - Metaseq / OPT as a test job


## Prerequisites
- Azure subscription
- Azure CLI configured with a subscription

## Step 1. Deploy CycleCloud managed Slurm cluster using ccslurm4ai

Generally follow the instructions in the [ccslurm4ai](https://github.com/yosoyjay/ccslurm4ai.git) repository.

TL;DR:

```bash
git clone -b ai-gbb-workshop https://github.com/yosoyjay/ccslurm4ai.git
cd ccslurm4ai
<edit-hard-coded-vars-in-install.sh>
bash install.sh
```

## Step 2. Connect to the Slurm scheduler node

Deployment of the Slurm cluster will create a set of login nodes, a scheduler, and a set of compute nodes.  The login nodes are the entry point to the cluster that we will connect with using the Bastion scripts created as part of the install process.  But, first we will connect to the scheduler and provision the compute nodes.

Provisioning of GPU VMs can take up to 10 minutes.  Now is a good time to read the basics on using Slurm, like this [Introducing Slurm](https://researchcomputing.princeton.edu/support/knowledge-base/slurm) from Princeton Research Computing.

```bash
[local]$ bash bastion_ssh_scheduler.sh
[scheduler]$ sudo /opt/azurehpc/slurm/resume_program.sh slurmcluster-hpc-[1-2]
[scheduler]$ sinfo # show partition and node status
[scheduler]$ squeue # show running jobs
```

### Check the status of the nodes

We should see two nodes in the "slurmcluster-hpc" partition with the "idle" state. If any of the nodes have a "drained" state, that means that the node has failed the health checks and is not available for scheduling jobs.  The source of the failed health check will have to be identified and resolved before the node can be used.

### Investigate and resolve the failed health checks

NHC runs on the node with a configuration defined at `/etc/nhc/nhc.conf` and produces a log file `/var/log/nhc.log `.  You can ssh into the node and check the log file to investigate the cause of the failed health check.

Once you believe the cause is resolved, you can re-run the health check via `sudo nhc`.  If the health check passes, the node will be available for scheduling jobs.

You can then tell Slurm to "undrain" the node with `sudo scontrol update nodename=<node-name> state=resume` and it will be available for scheduling jobs.

### Step 3. Clone this repo!

Clone this repository.  Because the home directories of the users are on an NFS mount, home directories will be available across the cluster.

```bash
[login] git clone https://github.com/yosoyjay/ai-gbb-bootcamp-2024
```

The directory of ai-gbb-bootcamp-2024 will be referenced as $AI_GBB_REPO throughout the rest of this document.

### Step 4. You can also run the health checks built into the azurehpc VM image.

Here we'll submit a job that runs the node health checks built into the azurehpc VM image at /opt/azurehpc/healthcheck.sh.  This script will run the health checks on all the nodes in the cluster and report the results.

The command can be invoke on any compute node.  For example, we can run it on a compute node gaining access to the node through an interactive job:

```bash
[login] srun --pty bash -i
[compute] /opt/azurehpc/test/run-tests.sh --mofed-lts true
 ...
[compute] exit
```

We can also run the health checks on all the nodes in the cluster by submitting a job to the scheduler.  Note, on larger clusters you would have to adjust the number of nodes and tasks in the job script.  Here we are assuming there are two node available.

```bash
[login] sbatch $AI_GBB_REPO/slurm/run_vm_healthcheck.sh
```
### Step 5. Run NCCL all reduce to verify GPU and Infiniband performance

Next we test a critical performance feature of the cluster, the ability to perform fast all-reduce operations across the GPUs using NCCL.  This is a critical feature for deep learning training.

```bash
[login] sbatch $AI_GBB_REPO/slurm/run_nccl_all_reduce.sh
```

You should see bandwidth of 180 GB/s on NDv4 and 490 GB/s on NDv5 on clusters enabled with [SHARP](https://docs.nvidia.com/networking/display/sharpv261/using+nvidia+sharp+with+nvidia+nccl).

### Step 6. Run Metaseq / OPT to verify cluster works for benchmark training

Finally, we will run a test job using Metaseq / OPT.  We first need to demonstrate why it is so much better to use containers, so we're going to install the dependencies for this on the compute nodes.

```bash
[login] srun --pty bash -i
[compute] bash $AI_GBB_REPO/metaseq/install_opt.sh
[compute] exit
[login] sbatch $AI_GBB_REPO/metaseq/run_opt.slurm
```

The job will take a few minutes to start, but you can monitor its progress with `squeue`.  Sometimes I'll run `watch -n 10 squeue` to see the job status update every 10 seconds.

When the job is completed, you can the logs and you should see entries of WPS (words per second) of ~200K.  This is a good indication that the cluster is working well for deep learning training.

🏆🏆🏆

### Step 7. Train a model you know!

Now that you have a working cluster, you can train a model you know.  You can use the same approach as in Step 6 to install the dependencies and submit a job to train your model.  Or, you can use the `$AI_GBB_REPO/nccl/all-reduce-container.slurm` as a template to create a new job script for a container.

If your job usually launches via `torchrun`, you can use the `$AI_GBB_REPO/slurm2torchrun.sh` that is a wrapper to convert Slurm environment variables and job configuration to `torchrun` environment variables.