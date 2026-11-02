# Lab 1: Judoor and HPC Access

**Status:** Required  
**Duration:** 2 hours scheduled, plus up to 1 hour of optional work  
**Where you work:** your own laptop, and the JURECA login nodes over SSH  
**Notebook:** [lab1_judoor_hpc_access.ipynb](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb)

The notebook is the full walkthrough. This page is the map: what the lab is for, what you need
before you start, what you hand in, and where the common traps are. Read this page once before you
open the notebook, and come back to the checklist at the end when you think you are done.

## What this lab is for

Every later lab in this course runs on a supercomputer, not on your laptop. Lab 1 does not teach
machine learning. It removes every obstacle between you and the machine that will run your models,
so that from Lab 2 onwards the only thing standing between you and a result is the experiment
itself.

By the end you will be able to:

1. Create and secure a Judoor account and join the course project.
2. Generate an SSH key pair that meets JSC's security rules, and register it in Judoor.
3. Log in to JURECA and orient yourself: who you are, which machine you are on, what you can use there.
4. Tell `HOME`, `PROJECT` and `SCRATCH` apart, and know which one your files belong in.
5. Build a course workspace you will reuse in every lab.
6. Submit a batch job, watch it, read its output, and check what it cost.

## Before you start

You need these before the session begins. Two of them take longer than you expect.

| What | Why | Where to get it |
| --- | --- | --- |
| A personal email address | Judoor registration and all account mail go there, not to a university alias you may lose access to | Your own inbox |
| A laptop you can install software on | You will configure SSH and, optionally, VS Code Remote SSH | Your own machine |
| An SSH client | Everything in this lab runs over SSH | Built in on Windows 10 1809 and later, macOS, and most Linux. Section 3 tells you how to check and how to install it |
| About 30 minutes of waiting time | Judoor account approval is not instant | Start before the session, not during it |

No prior HPC experience is assumed. If you have never used a terminal, work through Sections 3 and 4
with the instructor rather than pushing ahead alone.

## How the two hours are spent

| Part | Notebook section | Rough time | Type |
| --- | --- | --- | --- |
| Orientation and account setup | Sections 1 to 2 | 20 min | Required |
| SSH keys and first login | Sections 3 to 4 | 40 min | Required |
| Storage areas and workspace | Sections 5 to 6 | 25 min | Required |
| Your first batch job | Section 7 | 30 min | Required |
| Wrap up and troubleshooting | Sections 8 to 9 | 5 min | Required |
| File transfer, VS Code, interactive jobs, Python environments | Optional A to D | 60 min | Optional |

The account approval in Section 2 is the part you cannot compress. Do it first, and generate your
SSH key locally while you wait.

## What you hand in

Three things, all of which you should already have as a side effect of finishing the checklist.

- The output of `whoami`, `hostname` and `pwd` from a JURECA login node.
- The job ID of a completed job, plus the first lines of its `.out` file.
- The output of `sacct` for that job, showing elapsed time and the resources it was charged.

Paste them into whatever the instructor asks for. Do not submit your private key or your passphrase,
and do not screenshot either one. Under no circumstances copy a private key onto the cluster; JSC
policy prohibits it, and it is a security incident rather than a shortcut.

## The traps that cost students the most time

These four come up every year. The notebook covers each of them in place; they are listed here so you
know which section to jump to.

!!! warning "The project is not active, so the paths are empty"
    `$PROJECT` and `$SCRATCH` are empty strings until you run `jutil env activate -p training2600`.
    A command like `cd $PROJECT` then silently leaves you in your home directory instead of failing,
    and `mkdir $PROJECT/foo` creates something you did not intend. Section 5 covers this, and
    suggests putting the activation in `~/.bashrc` so you never hit it again.

!!! warning "Forgetting `--partition` puts you on the GPU queue"
    The default partition on JURECA is `dc-gpu`, not a CPU partition. A small CPU test job submitted
    without `--partition` sits behind GPU jobs and looks broken. Always name the partition
    explicitly. Section 7 uses `dc-cpu-devel` for your first job because it turns around quickly and
    has internet access.

!!! warning "A passphrase is mandatory, not optional"
    JSC refuses keys without one. If you press Enter when `ssh-keygen` asks, the key will be rejected
    at login with an unhelpful error. Section 3 explains why, and sets up the ssh-agent so you type
    the passphrase once per boot rather than once per connection.

!!! warning "Compute nodes have no internet"
    `pip install` inside a job script fails unless you asked for a partition that has network access.
    This surprises almost everyone, and the error message points at pip rather than at the network.
    Section 7 and Optional D both flag it.

## Getting help

- Section 9 of the notebook is a troubleshooting table built from the errors you will actually see.
  Search it before asking.
- JSC support is at `juser-f@fz-juelich.de`, and the JURECA documentation is at
  [apps.fz-juelich.de/jsc/hps/jureca](https://apps.fz-juelich.de/jsc/hps/jureca/). When you write to
  them, include the exact command, the exact message, and the node you typed it on.
- Ask in the course channel with the command and the full error text. The error is usually enough to
  diagnose the problem in a single reply.

## Optional extensions

Everything below is optional. Return to it once every required checkpoint passes.

- **Optional A: move files between your laptop and JURECA.** `scp` for one-off copies, `rsync` for
  anything you will copy more than once, and why neither behaves the same way on Windows.
- **Optional B: edit JURECA files from VS Code.** Remote SSH, so the notebook and the data live on
  the cluster while the editor lives on your laptop.
- **Optional C: interactive sessions and a GPU check.** `salloc` for exploratory work, and how to
  confirm a node really has the GPUs you asked for.
- **Optional D: software modules and your first Python environment.** Why `module avail Python` looks
  nearly empty until you load a stage, and how to build a virtual environment you control.

## Next

Lab 2 replaces this manual SSH workflow with Jupyter-JSC, and sets up the Git and Python environment
that Labs 3 to 6 depend on. The workspace layout you build in Section 6 is the one Lab 2 expects to
find, so do not invent your own.
