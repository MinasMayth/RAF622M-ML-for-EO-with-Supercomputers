# Lab 2 — Git

**Notebook:** [lab2_jupyter_jsc_git.ipynb](../../../notebooks/iceland-ml/lab2_jupyter_jsc_git.ipynb)

## Scope
Git done properly on the cluster. The environment and kernel are Lab 1's deliverable;
Lab 2 opens by verifying them and names the Lab 1 step to re-run if a check fails. It
deliberately does not rebuild them — the 2025/26 edition carried a second, drifting copy
of the whole setup here, and the two contradicted each other.

## Learning outcomes
- Launch a Jupyter-JSC session and open course notebooks in the course kernel.
- Distinguish "the venv exists" from "the venv runs unaided", and why only the second
  one matters to a kernel.
- Set a real Git identity and assert it, rather than shipping a placeholder.
- Clone, branch, commit, merge, and resolve a conflict with state verified after each
  step.
- Keep data artifacts out of Git history.

## Suggested flow (2h)
1. Start Jupyter-JSC session, open the notebook in `ML-EO Course`
2. Section 5 — verify the Lab 1 environment (four assertions)
3. Git identity, workflow, artifacts, merge conflict
4. `pytest tests -q` — this is the Lab 2 gate

## Expected outputs
- Environment checks green: clone path, unaided interpreter, `sys.prefix`, `-E` in argv
- Git configured and verified, with a commit you authored
- A green `pytest tests -q`
