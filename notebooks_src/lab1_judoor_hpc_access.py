# %% [markdown]
# # Lab 1 — Judoor and JURECA access
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers**
#
# ## What this lab produces
#
# Four machine-checkable artifacts, all written by cells you run:
#
# 1. A course workspace layout under your project space, asserted to exist.
# 2. One SLURM job you submitted from this notebook, with its exit status read back
#    from the accounting database (`sacct`) — not from your memory of running it.
# 3. A parsed `sinfo`/`squeue` view that proves you can see `dc-cpu` and `dc-gpu`
#    and read why a job is sitting in `PENDING`.
# 4. A record in `results.json` written by `eo_course.results.record_run`, carrying
#    your predictions and the outcomes you recorded.
#
# ## What it assumes
#
# Nothing except a Judoor account and a working SSH key. If you have not joined
# `training2653`, do that first: <https://judoor.fz-juelich.de/projects/join/training2653>.
# Approval is asynchronous and is not something the 2 hours in this lab can absorb,
# so start it before the session, not during it.
#
# ## You must run this yourself
#
# The 2025/26 edition of this notebook had **eleven markdown cells and zero code
# cells**. Every command lived inside a markdown fence, so nothing executed, nothing
# was asserted, and the deliverable was a list of checkboxes you ticked yourself.
# A student who never logged in could submit a completed Lab 1. That is not
# possible here: the cells below `assert`, and an `AssertionError` in your copy is
# the lab telling you the thing you think you have is not what you have.
#
# **Run this notebook on JURECA**, inside a Jupyter-JSC session or over a Jupyter
# server you started yourself. Cell 6 fails loudly if you are on a laptop.

# %% [markdown]
# ## 1. Setup
#
# One import block, one seed, one place where paths come from. `eo_course.paths`
# resolves every directory from environment variables, so this notebook contains no
# hard-coded absolute paths and cannot accidentally point at another student's data.

# %%
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from eo_course import gates, paths, results

SEED = 20260115
# One RNG for the whole notebook, per the course rule. Lab 1 has no stochastic step,
# so it is created here and unused; later labs pass it down instead of calling
# np.random.* directly, which is what made the 2025/26 outputs irreproducible.
rng = np.random.default_rng(SEED)

print(paths.describe())

# %%
def run(cmd, cwd=None):
    """Run a command, echo it, capture stdout/stderr, return the CompletedProcess.

    Every shell interaction in this lab goes through here. `subprocess.run` takes an
    explicit `cwd`, which is the property IPython's `!` operator does not have: each
    `!` line is its own `/bin/sh -c` process, so a `!cd` on one line cannot affect
    the next. Lab 2 demonstrates that failure in detail.
    """
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"$ {printable}" + (f"    [cwd={cwd}]" if cwd else ""))
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print("[stderr] " + proc.stderr.rstrip())
    if proc.returncode != 0:
        print(f"[returncode] {proc.returncode}")
    return proc

# %% [markdown]
# ## 2. Are you actually on JURECA?
#
# This is not pedantry. The 2025/26 Lab 2 contained a cell titled "HPC Environment
# Check" whose stored output was `Hostname: jrlogin04.jureca`, while the architecture
# diagram two cells earlier claimed the notebook ran on a compute node. One of the
# two was wrong, nobody noticed, and the labs that followed did multi-gigabyte raster
# I/O on the machine the check was supposed to be verifying.

# %%
HOST = socket.gethostname()

print(f"hostname      : {HOST}")
print(f"user          : {paths.username()}")
print(f"on JURECA     : {paths.on_jureca()}")
print(f"login node    : {paths.on_login_node()}")
print(f"SLURM_JOB_ID  : {os.environ.get('SLURM_JOB_ID', 'unset (notebook not inside a SLURM job)')}")
print(f"python        : {sys.executable}")

assert paths.on_jureca(), (
    "This notebook must run on JURECA. eo_course.paths could not find cluster storage.\n"
    f"  resolved scratch root: {paths.scratch_root()}\n"
    f"  resolved project root: {paths.project_root()}\n"
    "Start a Jupyter-JSC session (https://jupyter-jsc.fz-juelich.de, System=JURECA, "
    f"Project={paths.PROJECT_ACCOUNT}) and open this notebook there."
)

# %% [markdown]
# ## 3. Judoor, SSH keys, and the `from=` restriction
#
# Do these steps **on your own laptop**, not in this notebook. They are the only
# parts of Lab 1 that cannot be executed from a Jupyter cell.
#
# ### 3.1 Register and join
#
# 1. Register at <https://judoor.fz-juelich.de/register>. Use your university email.
# 2. Verify your email address.
# 3. Join the course project: <https://judoor.fz-juelich.de/projects/join/training2653>.
# 4. Accept the usage policy. Training-project approval is normally fast, but it is
#    not instantaneous, and it is the single most common reason a student cannot
#    start Lab 2.
#
# ### 3.2 Generate a key pair
#
# ```bash
# ssh-keygen -t ed25519 -C "you@your-institution" -f ~/.ssh/jureca_key
# cat ~/.ssh/jureca_key.pub
# ```
#
# Mechanism, not conclusion: `ssh-keygen` writes a private key (never leaves your
# laptop) and a public key (you paste that into Judoor). Authentication works
# because the server holds your public key and asks your client to prove it holds
# the matching private one by signing a challenge. There is no password on the wire,
# which is why JSC disables password authentication for these accounts.
#
# ### 3.3 Upload the key **with** the `from=` option — and know what it does
#
# In Judoor, under "JURECA Manage SSH-Keys", the key line has the form
#
# ```
# from="130.208.0.0/16" ssh-ed25519 AAAAC3Nza... you@your-institution
# ```
#
# `from=` is an **authorized_keys source-address restriction**: the server accepts
# this key only when the TCP connection arrives from an address inside that CIDR
# range. `130.208.0.0/16` is the University of Iceland network range. Judoor
# requires a restriction of this kind; the 2025/26 lab mandated it and never said
# what it was, which produced the most common support request of the course.
#
# **The failure mode, precisely.** Your source IP is whatever your traffic egresses
# as, not what your laptop thinks its address is. On campus you are inside
# `130.208.0.0/16`. On eduroam at a different institution, on a mobile hotspot, or
# behind a VPN whose egress pool is not that range, you are outside it — and the key
# stops working *silently*. You get `Permission denied (publickey)` with no hint that
# the reason is your IP, and the natural conclusion is "my key is broken", so
# students regenerate keys forever.
#
# Diagnose it before you email anyone:
#
# ```bash
# curl -s https://ifconfig.me; echo      # your real egress address
# ssh -vvv jureca 2>&1 | grep -iE 'Offering|Authenticated|source|denied'
# ```
#
# If the egress address is outside your `from=` range, that is the whole problem.
# Either connect from inside the range, or update the range in Judoor to the one
# Judoor suggests for your current location.
#
# ### 3.4 SSH client config — **required**, not optional
#
# The 2025/26 lab listed "Configured SSH config file" as an *optional* deliverable
# while its very next step was `ssh jureca`. That alias only resolves because of
# that file. It is mandatory.
#
# Create `~/.ssh/config` on your laptop:
#
# ```
# Host jureca
#     HostName jureca.fz-juelich.de
#     User YOUR_JUDOOR_USERNAME
#     IdentityFile ~/.ssh/jureca_key
#     ServerAliveInterval 60
#     ServerAliveCountMax 30
# ```
#
# `ServerAliveInterval` keeps idle connections alive through the firewalls that drop
# them after ~15 minutes; without it, long `sbatch` waits and `tail -f` on job logs
# die mid-command and you lose your shell state.
#
# Verify from your laptop terminal:
#
# ```bash
# ssh jureca 'hostname; echo $USER; sinfo --version'
# ```

# %% [markdown]
# ## 4. The filesystem, with the numbers that matter
#
# | Area | Root | Backed up | Retention / quota | Use it for |
# |---|---|---|---|---|
# | Home | `/p/home/jusers/$USER/jureca` | yes | ~50 GB | dotfiles, SSH config, small configs |
# | Project | `/p/project1/training2653` | yes | shared project quota | code, environments, results, anything not regenerable |
# | Scratch | `/p/scratch/training2653` | **no** | deleted after **90 days without access** | large, regenerable intermediate data |
#
# Three facts students get wrong, each of which cost someone in 2025/26:
#
# 1. **Project space is `/p/project1/training2653` — note the `1`.** The 2025/26
#    Lab 1 taught a project root *without* the `1`, and printed it as if it were
#    observed output. Every student who typed it literally ended up in a directory
#    that does not exist, and then wrote their Lab 4 outputs somewhere that no later
#    lab looked at.
# 2. **Scratch is deleted after 90 days *without access*, not 90 days after
#    creation.** The old lab's wording ("files deleted after 90 days") implies a
#    countdown from `touch`. It is a countdown from your last read or write. Keep
#    working in it and your data survives a term break; touch nothing over the
#    summer and it is gone. Scratch is **not backed up** — deletion is unrecoverable.
# 3. **`$SCRATCH` points at the *project* scratch root, not your own directory.**
#    On JURECA it expands to `/p/scratch/training2653`. You must append `$USER`
#    yourself. The 2025/26 Lab 2 printed `$SCRATCH` bare, and three different
#    spellings of the data path then appeared across labs 3–6, so lab 5 and lab 6
#    failed their own existence checks for students who had correctly followed
#    lab 4.2.

# %%
print("environment variables as JSC set them:")
for var in ("HOME", f"PROJECT_{paths.PROJECT_ACCOUNT.replace('-', '_')}", "PROJECT",
            f"SCRATCH_{paths.PROJECT_ACCOUNT.replace('-', '_')}", "SCRATCH"):
    print(f"  {var:24s}= {os.environ.get(var, '<unset>')}")

scratch_root = paths.scratch_root()
project_root = paths.project_root()
print(f"\npaths.scratch_root()   = {scratch_root}")
print(f"paths.user_scratch()   = {paths.user_scratch()}")
print(f"paths.project_root()   = {project_root}")
print(f"paths.user_project()   = {paths.user_project()}")

assert project_root.name == paths.PROJECT_ACCOUNT, project_root
assert "project1" in str(project_root), (
    f"project root {project_root} is not under /p/project1 — check "
    f"PROJECT_{paths.PROJECT_ACCOUNT.replace('-', '_')}"
)
assert paths.user_scratch() != scratch_root, (
    "your personal scratch dir must be a subdirectory of the scratch root, not the root itself"
)
print("\nOK: $SCRATCH is the shared root; your work lives one level down, under your username.")

# %% [markdown]
# ### 4.1 Storage quota
#
# The lab keeps telling you not to work in `$HOME`. Here is the number behind that
# instruction, plus the authoritative tool. JSC project space is a Lustre filesystem,
# so the quota command is `lfs quota`, and project quotas are tracked **per group**,
# not per user — `-g $(id -gn)` is what you want for project space.
#
# If `lfs` is unavailable, the cell falls back to `du`, which measures what you are
# using but cannot tell you the limit. That difference is recorded in the deliverable
# rather than hidden.

# %%
quota_lines = []

# Project space on JURECA is Lustre with a GROUP quota, so -g $(id -gn) is the
# authoritative query. Home is a per-user quota on a different filesystem.
grp = run(["sh", "-c", "id -gn"]).stdout.strip()
home = Path(os.environ.get("HOME", str(Path.home())))
for label, target, flag in (("home", home, "-u"), ("project", project_root, "-g")):
    who = paths.username() if flag == "-u" else grp
    p = run(["lfs", "quota", "-h", flag, who, str(target)])
    ok = p.returncode == 0 and p.stdout.strip()
    quota_lines.append((label, "lfs" if ok else "unavailable", p.stdout.strip() or p.stderr.strip()))

for label, kind, text in quota_lines:
    print(f"\n--- {label} quota via {kind} ---\n{text}")

own = {}
for label, d in (("user_project", paths.user_project()), ("user_scratch", paths.user_scratch())):
    d.mkdir(parents=True, exist_ok=True)
    du = run(["du", "-sh", str(d)])
    own[label] = du.stdout.split("\t")[0].strip() if du.returncode == 0 else "n/a"
print("\nyour own usage:", own)

QUOTA_VERIFIED = any(kind == "lfs" for _, kind, _ in quota_lines)
print(f"\nlfs quota available: {QUOTA_VERIFIED} "
      f"({'you have the real limits' if QUOTA_VERIFIED else 'you only have du; record the limit from Judoor'})")

# %% [markdown]
# ## 5. Exercise 1 — predict before you run: what does `sinfo` show?
#
# Write your answers down **in the next markdown cell** before running anything.
#
# 1. How many partitions will `sinfo` list for this account? Give an integer.
# 2. Which of them will report state `up`?
# 3. On `dc-cpu`, roughly what fraction of CPUs will be `allocated` right now —
#    under 25 %, 25–75 %, or over 75 %?
#
# The 2025/26 lab named the partitions with the machine name and an underscore in
# upper case (`JURECA-DC_` plus `CPU`/`GPU`). Those are not partition names. Its own
# commands used the correct lower-case names two cells later, so the notebook
# contradicted itself and students copied the wrong one into `#SBATCH --partition=`.
# The real names are **`dc-cpu`** (128 physical cores and 512 GB per node),
# **`dc-cpu-bigmem`** (1 TB per node) and **`dc-gpu`** (4× A100-40GB and 128 cores
# per node). Confirm them live, every time, because they are the only source of
# truth:

# %%
PREDICTION = {
    "n_partitions": None,   # TODO(you): integer
    "partitions_up": [],    # TODO(you): list of names you expect in state up
    "cpu_load_band": None,  # TODO(you): "<25%" | "25-75%" | ">75%"
}
print("Record your prediction in this cell, then run the next one.")
print(PREDICTION)

# %%
SINFO_FMT = "%P|%S|%D|%a|%A|%C|%G|%l|%T"
sinfo = run(["sinfo", "--noheader", f"--format={SINFO_FMT}"])
assert sinfo.returncode == 0, "sinfo failed — are you on a JURECA login or compute node?"

partitions = {}
for line in sinfo.stdout.strip().splitlines():
    f = line.split("|")
    if len(f) < 9:
        continue
    # `%P` carries a trailing `*` on your default partition; strip it so names match
    # what you write in `#SBATCH --partition=`.
    name = f[0].strip().rstrip("*")
    row = partitions.setdefault(name, {"states": set(), "nodes": 0, "cpu": [0, 0, 0, 0], "gres": set()})
    row["states"].add(f[1].strip().rstrip("*"))
    row["nodes"] += int(f[2])
    # `%C` is allocated/idle/other/down
    row["cpu"] = [a + b for a, b in zip(row["cpu"], [int(x) for x in f[5].split("/")])]
    row["gres"].add(f[6])

print(f"\n{'partition':16s} {'states':12s} {'nodes':>6s}  {'cpus  alloc/idle/other/down':26s} gres")
for name, r in sorted(partitions.items()):
    alloc, idle, other, down = r["cpu"]
    print(f"{name:16s} {'/'.join(sorted(r['states'])):12s} {r['nodes']:6d}  "
          f"{alloc:>7d}/{idle:<6d}/{other:<6d}/{down:<6d} {','.join(sorted(r['gres']))}")

seen = set(partitions)
assert {"dc-cpu", "dc-gpu"} <= seen, (
    f"expected dc-cpu and dc-gpu in {sorted(seen)} — check the account/partition list with `sinfo`"
)
print("\nOK: dc-cpu and dc-gpu are visible to this account.")

# %% [markdown]
# ### What to look for in that table
#
# * **State.** `up` means schedulable. `drain` / `drain*` means the node is
#   finishing existing jobs and taking no new ones — a job queued to a drained-only
#   partition waits indefinitely with reason `Resources`. `down` nodes are excluded
#   from the allocation entirely.
# * **CPUs `allocated/idle/other/down`.** Idle CPUs are what your job can start
#   immediately. If idle is 0 your job is queued no matter how small it is.
# * **Gres.** The GPU inventory per partition. If `dc-cpu` shows `(null)` it has no
#   GPUs, which is the answer to Exercise 3 below.
# * **The trailing `*`** marks your default partition — the one `sbatch` uses when
#   your script has no `#SBATCH --partition=` line. Never rely on it.
#
# Now the queue itself, and the column students never read:

# %%
SQUEUE_FMT = "%i|%u|%T|%p|%D|%C|%M|%r|%l|%j"
squeue = run(["squeue", "--noheader", f"--format={SQUEUE_FMT}"])
assert squeue.returncode == 0, "squeue failed"

jobs = []
for line in squeue.stdout.strip().splitlines():
    f = line.split("|")
    if len(f) >= 10:
        jobs.append(dict(zip(
            ("jobid", "user", "state", "partition", "nodes", "cpus", "time", "reason", "timelimit", "name"), f)))

print(f"\n{len(jobs)} job(s) in the queue across the whole system.")
mine = [j for j in jobs if j["user"] == paths.username()]
print(f"{len(mine)} of them are yours.")

from collections import Counter
print("\nwhy jobs are waiting (Reason column, whole system):")
for reason, count in Counter(j["reason"] for j in jobs).most_common(8):
    print(f"  {count:5d}  {reason}")

# %% [markdown]
# ### Why a job sits in `PENDING`
#
# Read the `Reason` column; it is the scheduler telling you the answer.
#
# | Reason | Meaning | What you do |
# |---|---|---|
# | `Priority` | Your job is valid and queued behind equally-or-higher priority work | Wait. Nothing is wrong. |
# | `Resources` | No node currently satisfies your request | Check idle CPUs above; shrink the request or wait |
# | `Dependency` | A `--dependency=afterok:ID` job has not finished | Check that job |
# | `QOSMaxWallPerJob` | Your `--time` exceeds what the account's QOS allows | Reduce `--time`; see below |
# | `QOSMaxSubmitJobPerUserLimit` | You already have your maximum number of jobs running | Finish or `scancel` some |
# | `NotAvailable` | The named resource (e.g. a GPU type) does not exist in that partition | Fix `--gres` or the partition |
#
# For a specific job: `scontrol show job <ID>` prints the full reason and the
# estimated start. For your fair-share standing: `sshare -a --user=$USER`.
#
# Your account's quality-of-service limits, which decide the maximum `--time` you may
# request. If `--time` exceeds the QOS wall limit, `sbatch` refuses and the error
# names `QOSMaxWallPerJob`; the 2025/26 lab gave no guidance for that message.

# %%
qos = run(["sh", "-c",
           f"sacctmgr -nP show user $USER withassoc account={paths.PROJECT_ACCOUNT} format=Account,QOS"])
if qos.returncode != 0 or not qos.stdout.strip():
    print("Could not read QOS associations (normal on some login nodes). "
          "If sbatch refuses a --time value, ask the instructor for the QOS name.")
else:
    print(f"account,QOS lines for {paths.PROJECT_ACCOUNT}:")
    print(qos.stdout.strip())

# %% [markdown]
# ## 6. Your workspace layout
#
# One layout, created once, used by every later lab. It lives in **project** space
# because environments and results are not regenerable; the large data directory
# created by labs 3–4 lives in **scratch** because it is.
#
# The 2025/26 lab created these directories with
#
# ```bash
# mkdir -p {repos,envs,scripts,logs,results}
# ```
#
# Brace expansion is a **bash** feature. Under `/bin/sh` (which on most Linux
# systems is `dash`), or in any context that is not bash, the shell creates a single
# directory literally named `{repos,envs,scripts,logs,results}` and reports success.
# Nothing fails; the layout is just wrong, and the first thing that breaks is a later
# lab that expects `envs/` to exist. We use explicit names, and we prove the bug.

# %%
WS = paths.user_project()
DIRS = ["repos", "envs", "scripts", "logs", "results"]
paths.ensure(*(WS / d for d in DIRS))
paths.ensure(paths.data_root())          # labs 3-4 write here; lab 1 never created it in 2025/26

for d in DIRS:
    p = WS / d
    assert p.is_dir(), f"missing {p}"
    print(f"OK  {p}")
assert paths.data_root().is_dir(), paths.data_root()
print(f"OK  {paths.data_root()}   (scratch; labs 3-4 write here)")

weird = [p.name for p in WS.iterdir() if p.name.startswith("{")]
assert not weird, f"brace-expansion leftovers present: {weird}"

# %% [markdown]
# ### Exercise — predict what `mkdir -p {repos,envs,...}` does
#
# Brace expansion is a **shell feature**, not a `mkdir` feature, and not every shell
# has it. Predict, then run:
#
# 1. How many directories does the braced command create under `/bin/sh` on this
#    machine? Give an integer.
# 2. Is `/bin/sh` on this machine bash or not? Answer yes or no.
#
# Why the question is worth asking: on Debian and Ubuntu `/bin/sh` is `dash`, which has
# no brace expansion, so `mkdir -p {repos,envs,scripts,logs,results}` creates **one**
# directory literally named `{repos,envs,scripts,logs,results}` and reports success.
# Nothing fails; the layout is just wrong, and the first thing that breaks is a later
# lab that expects `envs/` to exist. On RHEL-family systems `/bin/sh` is bash, which
# does expand it — so the same command is correct on one cluster and quietly wrong on
# another. Explicit names are correct everywhere, which is why the workspace cell above
# used them.

# %%
PREDICTION["braced_dirs_under_sh"] = None  # TODO(you): integer
PREDICTION["sh_is_bash"] = None            # TODO(you): True | False
print("Fill in PREDICTION, run this cell, then the next one.")

# %%
demo = paths.user_scratch("lab1_demo")
shutil.rmtree(demo, ignore_errors=True)
demo.mkdir(parents=True)

sh_is_bash = run(["sh", "-c", "[ -n \"$BASH_VERSION\" ] && echo yes || echo no"]).stdout.strip()
broken = run(["sh", "-c", f"cd {demo} && mkdir -p {{repos,envs,scripts,logs,results}} && ls -1"])
braced = broken.stdout.split()

print(f"\n/bin/sh is bash here: {sh_is_bash}   (you predicted {PREDICTION['sh_is_bash']})")
print(f"braced command created {len(braced)} entries: {braced}")
if len(braced) == 1 and braced[0].startswith("{"):
    print("  -> this shell did NOT expand the braces: one garbage directory, exit status 0.")
else:
    print("  -> this shell DID expand the braces. On a dash-based /bin/sh the same "
          "command creates one directory named '{repos,envs,scripts,logs,results}'.")

shutil.rmtree(demo, ignore_errors=True)
demo.mkdir(parents=True)
good = run(["sh", "-c", f"cd {demo} && mkdir -p repos envs scripts logs results && ls -1"])
explicit = good.stdout.split()
print(f"\nexplicit names created {len(explicit)} entries under /bin/sh: {explicit}")
assert len(explicit) == 5, f"expected 5 directories, got {explicit}"
print("OK: the explicit form is shell-independent. That is the only form to use.")
shutil.rmtree(demo, ignore_errors=True)

# %% [markdown]
# ## 7. Copying files: the `$VAR` that expands on the wrong machine
#
# The 2025/26 lab taught:
#
# ```bash
# scp myfile.txt jureca:$PROJECT_training2653/scripts/
# rsync -avzP mydir/ jureca:$PROJECT_training2653/mydir/
# ```
#
# Both are broken, and the error message does not hint at the cause. `scp` and
# `rsync` run on your **laptop**, so your **local** shell expands `$PROJECT_training2653`
# before anything reaches the network. On your laptop that variable is unset, so the
# remote path collapses to `/scripts/` and you get `Permission denied` writing to the
# root of the remote filesystem. The fix is to stop the local shell from expanding
# it — single quotes, or a backslash — so the **remote** shell expands it.

# %%
demo = paths.user_scratch("lab1_demo")
demo.mkdir(parents=True, exist_ok=True)

acct = paths.PROJECT_ACCOUNT
acct_env = acct.replace("-", "_")
broken = run(["sh", "-c", f"echo jureca:$PROJECT_{acct_env}/scripts/"])
fixed = run(["sh", "-c", rf"echo 'jureca:/p/project1/{acct}/$USER/scripts/'"])

b = broken.stdout.strip()
f = fixed.stdout.strip().replace("$USER", paths.username())
print(f"\nbroken form expands locally to : {b}")
print(f"quoted form resolves remotely to: {f}")
assert b.endswith("/scripts/") and acct not in b, "demo did not reproduce the bug"
assert acct in f and paths.username() in f

print(f"""
Use, from your laptop:
  scp myfile.txt 'jureca:/p/project1/{acct}/$USER/scripts/'
  rsync -avzP mydir/ 'jureca:/p/project1/{acct}/$USER/mydir/'
or an absolute path with no variables at all. Single quotes are the reliable habit:
they also protect spaces, which OneDrive-backed paths on Windows have in abundance.""")

# %% [markdown]
# ## 8. Exercise 2 — predict the allocation, then submit a real job
#
# This was the 2025/26 example job. Read the directives and answer before running.
#
# ```bash
# #SBATCH --partition=dc-cpu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=4
# #SBATCH --cpus-per-task=20
# #SBATCH --threads-per-core=2
# #SBATCH --time=00:10:00
#
# sleep 30
# ```
#
# 1. How many **logical** CPUs does SLURM allocate for this job? Give an integer.
# 2. How many **physical** cores is that, given `--threads-per-core=2`?
# 3. The job body is `sleep 30`. How many of those cores does it use?
#
# Write the three numbers in the cell below. Then run the next cell to see what a
# sane request looks like and what SLURM actually reports back.
#
# Why this matters more than the arithmetic: `training2653` is a **shared**
# allocation with a fixed core-hour budget. A job that reserves 80 logical CPUs for
# 10 minutes burns 13.3 core-hours to run `sleep`. Twelve students doing that once
# consume a day's worth of the course's budget. Over-requesting is not free, and
# `sacct` shows exactly who did it.

# %%
PREDICTION["alloc_logical_cpus"] = None   # TODO(you): integer
PREDICTION["alloc_physical_cores"] = None  # TODO(you): integer
PREDICTION["cores_used_by_sleep"] = None   # TODO(you): integer
print("Fill in PREDICTION above, run this cell, then submit the job.")

# %%
SBATCH_SMOKE = """#!/bin/bash
# Lab 1 smoke test. You must be able to explain every #SBATCH line below.
#SBATCH --job-name=lab1_smoke
#SBATCH --account=training2653
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:05:00
#SBATCH --output=logs/lab1_smoke_%j.out
#SBATCH --error=logs/lab1_smoke_%j.err
set -euo pipefail

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "partition=$SLURM_JOB_PARTITION"
echo "cpus_per_task=$SLURM_CPUS_PER_TASK"
echo "user=$USER"
sleep 5
echo "smoke_ok=1"
"""

smoke = WS / "scripts" / "lab1_smoke.sbatch"
smoke.write_text(SBATCH_SMOKE)
print(smoke.read_text())
print(f"written to {smoke}")

# %% [markdown]
# ### Reading the directives
#
# | Directive | Value here | Why this value |
# |---|---|---|
# | `--account` | `training2653` | Charges the job to the course allocation. Without it SLURM uses your default account, which may have no funds. |
# | `--partition` | `dc-cpu` | Explicit. Never rely on the `*` default partition. |
# | `--nodes=1 --ntasks-per-node=1` | one process | The job body is `hostname` and `sleep`. One process. |
# | `--cpus-per-task=1` | 1 CPU | The 2025/26 example asked for 80 logical CPUs for the same body. |
# | `--mem-per-cpu=1000` | 1 GB | Without a memory request you get the partition default per CPU, and on a busy partition that is what makes jobs fail to start. |
# | `--time=00:05:00` | 5 minutes | A ceiling on your mistake. Over-requesting wall time also delays your own next job. |
# | `--output=logs/lab1_smoke_%j.out` | `%j` = job id | **Relative to the directory you submitted from**, not to the script's location. That is why we submit from the workspace root. |
#
# The 2025/26 lab said "create `scripts/test_job.sh`" and then ran `sbatch test_job.sh`,
# which fails with `Unable to open test_job.sh` from the workspace root. We use
# `--parsable` so the cell can capture the job id without parsing prose.

# %%
def job_record(job_id: str) -> dict:
    """Read one job's accounting record. Returns {} if it is not in the DB yet.

    `--parsable2` gives a `|`-delimited line with no trailing pipe, so the field
    count is stable. We match on the job id with any `.step` suffix stripped, and
    take the row whose first field is exactly our id (not `8123.batch`).
    """
    fmt = "JobID,JobName,Partition,AllocCPUS,AllocTRES,State,ExitCode,Submit,Start,End,Elapsed,MaxRSS,Reason"
    p = subprocess.run(
        ["sacct", "-X", "-j", job_id, "--noheader", "--parsable2", f"--format={fmt}", "--state=all"],
        capture_output=True, text=True,
    )
    cols = fmt.split(",")
    for line in p.stdout.strip().splitlines():
        f = line.split("|")
        if len(f) >= len(cols) and f[0].split(".")[0] == job_id:
            return dict(zip(cols, f[: len(cols)]))
    return {}


def wait_for_job(job_id: str, timeout_s: int = 300, poll_s: int = 5) -> dict:
    """Poll until the job leaves the queue, then return its final accounting record."""
    seen = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rec = job_record(job_id)
        if not rec and not seen:
            print(f"  no sacct row yet for {job_id} — the accounting database lags "
                  "submission by a few seconds; keep waiting")
        state = rec.get("State", "NOTSTARTED").strip()
        if state not in seen:
            seen.append(state)
            print(f"  [{time.strftime('%H:%M:%S')}] {job_id} -> {state}"
                  + (f"  reason={rec.get('Reason')}" if state == "PENDING" else ""))
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
            return rec
        time.sleep(poll_s)
    raise TimeoutError(
        f"job {job_id} did not finish within {timeout_s}s; states seen: {seen}. "
        f"Check `scontrol show job {job_id}` for the pending reason."
    )


sub = run(["sbatch", "--parsable", "scripts/lab1_smoke.sbatch"], cwd=WS)
assert sub.returncode == 0, "sbatch refused the job; read the stderr above before asking for help"
JOB_ID = sub.stdout.strip().splitlines()[-1].split("_")[0]
print(f"\nsubmitted job {JOB_ID}; watching it through the scheduler.")
final = wait_for_job(JOB_ID)

# %%
print(f"{'field':12s} value")
for k, v in final.items():
    print(f"{k:12s} {v}")

print("\njob stdout:")
outs = sorted((WS / "logs").glob(f"lab1_smoke_{JOB_ID}.out"))
assert outs, f"no stdout file matching logs/lab1_smoke_{JOB_ID}.out in {WS / 'logs'}"
print(outs[0].read_text().rstrip())

assert final.get("State", "").strip() == "COMPLETED", final
assert final.get("ExitCode", "").strip() == "0:0", (
    f"ExitCode {final.get('ExitCode')!r} is not 0:0 — the job ran and failed; "
    "read logs/ in the workspace"
)
print(f"\nOK: job {JOB_ID} completed with exit status 0:0 on partition {final.get('Partition')}.")

# %% [markdown]
# ## 9. Exercise 3 — predict the GPU request
#
# You are on `dc-cpu`, which `sinfo` above reported with no GPU gres. You now submit
# the *same* script with `--gres=gpu:1` added on the command line.
#
# Predict, in the cell below, and write one sentence for each:
#
# 1. Which partition will the job run on?
# 2. Will `sbatch` accept the submission, or refuse it? If it refuses, what does the
#    message say?
# 3. If it accepted, what would `sacct` report in `AllocTRES` for `gres/gpu`?
#
# Command-line `sbatch` options **override** `#SBATCH` lines in the script. That
# precedence is worth knowing: it is how you test a variant without editing the file,
# and it is also how students end up running a job on a partition they never wrote
# down.

# %%
PREDICTION["gpu_partition"] = None      # TODO(you): partition name
PREDICTION["gpu_submission"] = None     # TODO(you): "accepted" | "refused"
PREDICTION["gpu_tres"] = None           # TODO(you): what AllocTRES will contain for gres/gpu
print(PREDICTION)

# %%
probe = WS / "scripts" / "lab1_gpu_probe.sbatch"
probe.write_text(SBATCH_SMOKE.replace("lab1_smoke", "lab1_gpu_probe"))

gpu = run(["sbatch", "--parsable", "--partition=dc-cpu", "--gres=gpu:1", "scripts/lab1_gpu_probe.sbatch"], cwd=WS)

if gpu.returncode != 0:
    outcome = "refused_at_submission"
    gpu_job = None
else:
    outcome = "accepted"
    gpu_job = gpu.stdout.strip().splitlines()[-1].split("_")[0]
    final_gpu = wait_for_job(gpu_job, timeout_s=240)
    print(f"AllocTRES = {final_gpu.get('AllocTRES')}")
    print(f"State     = {final_gpu.get('State')}  ExitCode = {final_gpu.get('ExitCode')}")
    outcome = f"accepted_and_{final_gpu.get('State', '').strip().lower()}"

print(f"\nOUTCOME: {outcome}")
print(f"you predicted: partition={PREDICTION['gpu_partition']} "
      f"submission={PREDICTION['gpu_submission']} tres={PREDICTION['gpu_tres']}")
print("predicted vs actual — if these disagree, write one sentence on why the "
      "scheduler behaved as it did:")
print("  TODO(you): ...")

# %% [markdown]
# ## 10. Usage guardrails
#
# Say this out loud, because it is the part nobody reads until the allocation is
# gone.
#
# **The compute budget is shared and auditable.** `training2653` has one pool of
# core-hours and GPU-hours for the whole cohort. `sacct` records every job you ever
# submitted with its allocation, wall time and peak memory, and the instructor can
# and does read it. Over-requesting is visible; so is everything you run.
#
# **No unbounded jobs.** Every submission carries `--time`. A job with no wall-time
# limit inherits the partition maximum, which on `dc-cpu` is days: if it hangs, it
# holds those cores until it is cancelled, and it holds them for everyone. Set
# `--time` to roughly twice your measured runtime, not to a number that makes you
# feel safe.
#
# **No unrelated workloads.** This allocation exists for the course's Sentinel-2 /
# CORINE pipeline. Mining, scraping, media transcoding, personal model training, and
# "just a small crypto thing" are project misuse and are detectable in the accounting
# data. JSC can and does revoke training accounts.
#
# **Small bounded test jobs first.** Before a 6-hour training run, submit the same
# script with `--time=00:10:00` and one-tenth of the data. The 2025/26 Lab 6 students
# discovered their checkpoint path was wrong four hours into a 4-GPU job. A 10-minute
# smoke test costs 0.7 GPU-hours; the full run cost 24.
#
# **Clean up large files.** Scratch is not backed up and is deleted after 90 days
# without access, but it is still a shared filesystem with a shared quota. Delete
# intermediate rasters you can regenerate, keep `results.json` and figures in project
# space, and never leave a 40 GB `.npz` in your home directory.
#
# Your own footprint from the last 24 hours:

# %%
acct = run(["sh", "-c", "sacct -X --noheader --parsable2 --starttime=now-1day "
            "--format=JobID,JobName,Partition,AllocCPUS,AllocTRES,State,ExitCode,Elapsed,TotalCPU,MaxRSS "
            "--user=$USER"])
cols = ["jobid", "name", "partition", "cpus", "tres", "state", "exit", "elapsed", "total_cpu", "maxrss"]
rows = [dict(zip(cols, ln.split("|"))) for ln in acct.stdout.strip().splitlines()]
print(f"\n{len(rows)} job(s) of yours in the last 24 h:")
for r in rows:
    print(f"  {r['jobid']:>12s} {r['name'][:16]:16s} {r['partition']:8s} cpus={r['cpus']:>4s} "
          f"{r['state']:10s} {r['exit']:6s} elapsed={r['elapsed']:12s} total_cpu={r['total_cpu']}")
print("\nTotalCPU is what you were charged. Compare it with Elapsed: if TotalCPU is far "
      "larger, you reserved more than you used.")

smoke_row = next((r for r in rows if r["jobid"].split(".")[0] == JOB_ID), {})
SMOKE_TOTAL_CPU = smoke_row.get("total_cpu", "n/a")
print(f"\nyour smoke job {JOB_ID} was charged {SMOKE_TOTAL_CPU} of CPU time "
      f"against {smoke_row.get('elapsed', 'n/a')} of wall time.")

# %% [markdown]
# ## 11. Gate board
#
# `eo_course.gates` raises on failure and prints a board of pass/fail lines. Lab 1's
# gates are access gates, not performance gates: they check the things a student
# cannot fake.

# %%
lines = []


def check(name, ok, detail):
    """Print one gate line and keep it for the gate board and the deliverable.

    eo_course.gates.report() owns the formatting; we mirror its line into `lines`
    so print_gate_board can render the summary verdict.
    """
    gates.report(name, bool(ok), detail)
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


check("on_jureca", paths.on_jureca(), f"host={HOST}")
check("project_root_project1", "project1" in str(project_root), str(project_root))
check("user_scratch_is_subdir", paths.user_scratch() != scratch_root, str(paths.user_scratch()))
check("workspace_layout", all((WS / d).is_dir() for d in DIRS), f"{len(DIRS)} dirs under {WS}")
check("data_root_created", paths.data_root().is_dir(), str(paths.data_root()))
check("partitions_visible", {"dc-cpu", "dc-gpu"} <= seen, f"saw {sorted(seen)}")
check("smoke_job_completed", final.get("State", "").strip() == "COMPLETED",
      f"job {JOB_ID} {final.get('State')} {final.get('ExitCode')}")
check("quota_authoritative", QUOTA_VERIFIED,
      "lfs quota available" if QUOTA_VERIFIED else "only du; record the limit from Judoor")
check("gpu_probe_recorded", outcome.startswith(("refused_at_submission", "accepted_and_")),
      f"outcome={outcome}")

ok = gates.print_gate_board(lines)
n_pass = sum(1 for ln in lines if "[PASS]" in ln)
print(f"\n{n_pass}/{len(lines)} lab-1 access gates passed.")

# quota_authoritative is informational: some JSC login nodes do not expose `lfs`.
# Everything else is a hard requirement for submission.
hard = [ln for ln in lines if "[FAIL]" in ln and "quota_authoritative" not in ln]
assert not hard, "access gate(s) failed:\n  " + "\n  ".join(hard)

# %% [markdown]
# ## 12. Deliverable
#
# `results.json` is the course ledger; every lab appends to it. Lab 1 has no model,
# so there are no accuracy numbers. `record_run` requires four metric keys, so we put
# the honest thing there — the access-gate tally — and say so in `notes` rather than
# inventing a score. Your predictions and their outcomes go in `config`, which is
# what gets graded.

# %%
pass_rate = n_pass / len(lines)
record = results.record_run(
    run_id=f"lab1-{paths.username()}-{time.strftime('%Y%m%dT%H%M%S')}",
    lab="lab1",
    config={
        "host": HOST,
        "on_login_node": paths.on_login_node(),
        "project_root": str(project_root),
        "user_scratch": str(paths.user_scratch()),
        "workspace": str(WS),
        "partitions_seen": sorted(seen),
        "smoke_job_id": JOB_ID,
        "smoke_partition": final.get("Partition"),
        "smoke_alloc_cpus": final.get("AllocCPUS"),
        "smoke_total_cpu": SMOKE_TOTAL_CPU,
        "gpu_probe_outcome": outcome,
        "quota_authoritative": QUOTA_VERIFIED,
        "predictions": {k: v for k, v in PREDICTION.items()},
    },
    split_manifest_hash=None,
    seed=SEED,
    test_metrics={
        "n": len(lines),
        "overall_acc": pass_rate,
        "balanced_acc": pass_rate,
        "macro_f1": pass_rate,
    },
    notes=(
        f"Lab 1 access check. No model: the four metric keys carry the gate tally "
        f"({n_pass}/{len(lines)} passed), not accuracy. Graded content is in config. "
        f"GPU-on-dc-cpu outcome: {outcome}."
    ),
)
out = results.default_results_path()
print(f"appended run_id={record['run_id']} to {out}")
assert out.exists(), f"record_run did not leave a file at {out}"
print(results.summary_table("lab1"))

# %% [markdown]
# ## 13. Submission checklist
#
# Concrete artifacts. Attach or paste each one; do not tick a box that says you
# understand.
#
# 1. **The executed notebook**, with no red cells. If a cell raised, keep the traceback
#    visible and write the diagnosis underneath it — a hidden error is worse than a
#    visible one.
# 2. **Your three prediction blocks** filled in, each with `predicted vs actual` and,
#    where they disagreed, one sentence explaining why.
# 3. **The `sinfo` table** your run printed, showing `dc-cpu` and `dc-gpu`.
# 4. **The `sacct` block for your smoke job**, showing `State=COMPLETED` and
#    `ExitCode=0:0`, plus the job id.
# 5. **The GPU-probe outcome line** (`refused_at_submission` or the state it reached),
#    and the sentence explaining it.
# 6. **`results.json`** from `paths.results_dir()`, containing your `lab1` record.
#
# ### What to bring to Lab 2
#
# Lab 2 launches Jupyter-JSC, clones the course repository, and builds the course
# virtual environment in `paths.user_project("envs")` — which is why `envs/` had to be
# created here. It also spends real time on one idea this lab only touched: each
# `!` line in a notebook is a separate `/bin/sh -c` process, which is why the 2025/26
# Lab 2's clone, its `source activate`, and its git branching demo all silently did
# nothing.
#
# ### Support
#
# * Course Slack channel first, with the exact error text.
# * JSC user support for anything account- or filesystem-level: <https://judoor.fz-juelich.de>.
# * JURECA hardware and filesystem documentation: <https://apps.fz-juelich.de/jsc/hps/jureca/>.
