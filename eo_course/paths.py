"""Single source of truth for where things live.

Why this module exists
----------------------
In the 2025/26 edition the same dataset was addressed by four incompatible
spelledings across the labs::

    /p/scratch/training2600/hashim1/training_data/combined_training_data.npz   (lab5_1: someone else's home)
    /p/scratch/training2600/{USER}/training_data                               (lab4_2 writes here)
    Path(os.getenv("SCRATCH")) / USER / "data" / "training_data"               (lab5_2, lab6 read here)
    /p/project/training2600                                                    (lab1 teaches this; it is wrong)

Three of those four do not resolve to the same directory, so lab5_2 and lab6
fail their ``assert TRAINING_DATA_DIR.exists()`` for any student who correctly
followed lab4_2. Everything below is derived once, from environment variables,
and every notebook and script must import it rather than write a literal.

JURECA facts you need
---------------------
* Project space is ``/p/project1/training2600`` -- note the ``1``. There is no
  ``/p/project/training2600``.
* Scratch is fast, large, **not backed up**, and files are deleted after 90
  days without access. Anything you cannot regenerate belongs in project space.
* ``$SCRATCH`` on JURECA points at the *project* scratch root
  (``/p/scratch/training2600``), not at your personal directory. You must
  append ``$USER`` yourself. Forgetting this is how students read each other's
  data and then cannot reproduce their own numbers.
"""

from __future__ import annotations

import getpass
import os
import socket
import tempfile
from pathlib import Path

#: Course SLURM account / project id, used in every sbatch template.
PROJECT_ACCOUNT = "training2600"

#: Canonical JURECA roots, used only when the corresponding env var is absent.
_JURECA_SCRATCH_ROOT = Path("/p/scratch") / PROJECT_ACCOUNT
_JURECA_PROJECT_ROOT = Path("/p/project1") / PROJECT_ACCOUNT

# Env vars JSC may set, in preference order. Different login methods and
# Jupyter-JSC session types populate different subsets of these.
_SCRATCH_VARS = ("EO_COURSE_SCRATCH", "SCRATCH_training2600", "SCRATCH")
_PROJECT_VARS = ("EO_COURSE_PROJECT", "PROJECT_training2600", "PROJECT")

_notice_shown = False


class CoursePathError(RuntimeError):
    """Raised when a course path cannot be resolved.

    This is deliberately an error and not a warning. A silently wrong data
    path is the most expensive failure mode in this course: it produces results
    that look fine and are not.
    """


def username() -> str:
    """The effective JSC username."""
    return os.environ.get("USER") or getpass.getuser()


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:  # pragma: no cover - defensive
        return ""


def on_jureca() -> bool:
    """True if this looks like a JSC login or compute node."""
    return Path("/p/scratch").is_dir() and bool(
        os.environ.get("SLURM_JOB_ID") or "jureca" in _hostname() or "jrlogin" in _hostname()
    )


def on_login_node() -> bool:
    """True if we are on a login node, where heavy I/O does not belong."""
    host = _hostname()
    return "jrlogin" in host or "login" in host


def _first_env(names: tuple[str, ...]) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


def _fallback_root() -> Path:
    """Off-cluster root, so the labs and the test suite also run on a laptop.

    Set ``EO_COURSE_ROOT`` to control it. Without it we use a temp directory and
    print exactly one notice, because a student who means to be on JURECA and is
    not must notice immediately.
    """
    global _notice_shown
    env = os.environ.get("EO_COURSE_ROOT")
    if env:
        return Path(env)
    root = Path(tempfile.gettempdir()) / f"eo_course_{username()}"
    if not _notice_shown and not os.environ.get("EO_COURSE_QUIET"):
        _notice_shown = True
        print(
            f"[eo_course.paths] Not on JURECA; using local root {root}\n"
            "                  Set EO_COURSE_ROOT to override, or the JSC env vars "
            "(SCRATCH/PROJECT) if you expected cluster storage."
        )
    return root


def scratch_root() -> Path:
    """Project scratch root, e.g. ``/p/scratch/training2600``."""
    root = _first_env(_SCRATCH_VARS)
    if root is not None:
        return root
    if _JURECA_SCRATCH_ROOT.is_dir():
        return _JURECA_SCRATCH_ROOT
    return _fallback_root() / "scratch"


def project_root() -> Path:
    """Project shared space root, e.g. ``/p/project1/training2600``."""
    root = _first_env(_PROJECT_VARS)
    if root is not None:
        return root
    if _JURECA_PROJECT_ROOT.is_dir():
        return _JURECA_PROJECT_ROOT
    return _fallback_root() / "project"


def user_scratch(sub: str = "") -> Path:
    """``<scratch>/<user>[/<sub>]`` -- the only scratch path you should use."""
    p = scratch_root() / username()
    return p / sub if sub else p


def user_project(sub: str = "") -> Path:
    """``<project>/<user>[/<sub>]`` -- use for anything not regenerable."""
    p = project_root() / username()
    return p / sub if sub else p


# --- Named locations used across the labs --------------------------------
# Defining them here means lab4_2's output and lab5's input cannot drift apart.


def data_root() -> Path:
    """Raw and intermediate EO data (large, regenerable)."""
    return user_scratch("data")


def aligned_dir() -> Path:
    """Lab 4.1 output: band-stacked S2 + reprojected CORINE GeoTIFFs."""
    return data_root() / "aligned_data"


def training_data_dir() -> Path:
    """Lab 4.2 output: per-scene patch archives. Lab 5+ input."""
    return data_root() / "training_data"


def splits_dir() -> Path:
    """Split manifests written by :mod:`eo_course.splits`."""
    return data_root() / "splits"


def artifacts_dir() -> Path:
    """Checkpoints and run outputs that must survive scratch cleanup."""
    return user_project("artifacts")


def results_dir() -> Path:
    """Graded deliverables: results.json / results.csv / figures."""
    return user_project("results")


def run_dir(run_id: str) -> Path:
    """Per-run output directory. Never reuse one across runs."""
    return artifacts_dir() / "runs" / run_id


def matplotlib_cache_dir() -> str:
    """A writable MPLCONFIGDIR.

    On a shared login node matplotlib defaults to a random dir under /tmp and
    emits a UserWarning; the old notebooks "fixed" this by setting the variable
    *after* importing matplotlib, which does nothing. Set it before the import.
    """
    cache = user_scratch(".cache/matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


def ensure(*paths: Path) -> list[Path]:
    """Create each directory (parents included) and return them."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
    return list(paths)


def require_existing(path, what: str = "input") -> Path:
    """Return ``Path(path)`` or raise with the message a student needs.

    The error names the producing lab, so a missing artifact points at the step
    that was skipped rather than surfacing as a mystery ``FileNotFoundError``.
    """
    p = Path(path)
    if not p.exists():
        raise CoursePathError(
            f"{what} not found: {p}\n"
            f"  resolved from: scratch_root={scratch_root()} user={username()}\n"
            "  If you are on JURECA, check you did not hard-code another user's path.\n"
            "  If a previous lab should have written this, re-run that lab and confirm\n"
            "  it reported 'saved' rather than 'skipped (outputs exist)'."
        )
    return p


def describe() -> str:
    """Human-readable summary; print it at the top of every lab."""
    lines = [
        f"  user            : {username()}",
        f"  on JURECA       : {on_jureca()}",
        f"  host            : {_hostname() or 'unknown'}",
        f"  login node      : {on_login_node()}",
        f"  scratch root    : {scratch_root()}",
        f"  project root    : {project_root()}",
        f"  data root       : {data_root()}",
        f"  training data   : {training_data_dir()}",
        f"  results         : {results_dir()}",
    ]
    return "\n".join(lines)
