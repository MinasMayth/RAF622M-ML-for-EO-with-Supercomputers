# Lab 1 — Judoor, JURECA Access, and the Course Kernel

**Notebook:** [lab1_judoor_hpc_access.ipynb](../../../notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb)

## Scope
Get from no account at all to a JupyterLab tab running the course kernel, then validate
a minimal HPC workflow. Lab 1 owns environment setup end to end; Lab 2 only verifies it.

## Learning outcomes
- Activate course access and connect via SSH, including the `from=` access rule.
- Understand `HOME`, `PROJECT`, and `SCRATCH` usage, and why the environment goes where
  it goes.
- Register a Jupyter kernel and prove the notebook is running in it via `sys.prefix`
  rather than the menu name.
- Run core scheduler commands (`squeue`, `sinfo`, `sbatch`) and read `sacct`.

## Suggested flow (2h)
1. Section 0 — account, SSH key, SSH config, Jupyter-JSC launch, clone, and
   `bash scripts/bootstrap_course_env.sh`. **Terminal only, no cells.**
2. Section 1 — register the kernel and switch to it. Standard library only, so it runs
   in the default kernel.
3. Sections 2 onward — filesystem, quotas, `sinfo`, a real `sbatch` job, deliverable.

## Expected outputs
- Working SSH login to the cluster
- Course workspace directory layout
- Environment built in project space, ending `BOOTSTRAP_OK`
- `ML-EO Course` kernel registered with `-E` and selected, verified by `sys.prefix`
- One successful test job submission, confirmed by `sacct`
