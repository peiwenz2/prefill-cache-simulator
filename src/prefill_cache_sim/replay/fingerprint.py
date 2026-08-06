"""Deterministic SHA-256 fingerprints of the sources that produce M10 artifacts.

The M10 generator and this package may be untracked at the commit an artifact
records, so ``git_sha`` alone does not identify the code that produced it: two
artifacts can carry the same commit and dirty flag and still come from different
source bytes. Fingerprinting the sources closes that gap without overclaiming.

M10 differs from M9 in what has to be covered. M9 measured a mock engine, so a
curated list of the modules its generator touched was defensible. M10 replays
through the *whole* simulator -- placement, cache, eviction, lifecycle, SLO --
so almost any module in the package can move a published number. Rather than
curate a list that silently stops covering a module the day someone adds one,
this module walks the package tree and fingerprints every ``.py`` file it finds.

The claim stays narrow. These digests, plus the recorded interpreter, *identify*
the source and runtime a run used. They do not promise that a matching source
tree reproduces the artifact bytes: that additionally requires the same
interpreter, the same third-party dependency versions, and the same input trace,
none of which a source digest can enforce.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ALGORITHM = "sha256"

#: Repository-relative path of the M10 artifact generator.
GENERATOR_PATH = "scripts/run_m10_synthetic.py"

#: Repository-relative root of the simulator package. Every ``.py`` file beneath
#: it is fingerprinted, because M10 replays through the whole simulator and a
#: change anywhere in it can change a published number.
PACKAGE_ROOT = "src/prefill_cache_sim"

#: What the fingerprints do and do not promise, carried in the artifact so the
#: claim travels with the numbers rather than living only in a review comment.
REPRODUCIBILITY_CLAIM = "SOURCE_AND_RUNTIME_IDENTIFIED_REPRODUCTION_CONDITIONAL"

REPRODUCIBILITY_NOTE = (
    "The M10 generator and the simulator package may be untracked at the "
    "recorded commit, so git_sha and git_dirty do not identify the code that "
    "produced this artifact. These digests identify the source files, and the "
    "recorded runtime identifies the interpreter. They do not by themselves "
    "make the artifact reproducible: byte-for-byte reproduction remains "
    "conditional on running that source under a matching Python runtime with "
    "matching third-party dependency versions against the trace whose sha256 "
    "this artifact records. The combined_digest covers the source file set "
    "only -- not the runtime, not installed dependencies, and not the trace."
)


def runtime_identity() -> dict[str, str]:
    """Name the interpreter running the generator.

    Floating-point results and library behaviour are runtime-dependent, so an
    artifact that records only its sources would claim more portability than it
    has. Recording the interpreter keeps the conditional part of the claim
    checkable instead of implicit.
    """
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def m10_source_paths(
    root: Path, generator_path: str = GENERATOR_PATH
) -> tuple[str, ...]:
    """Enumerate the M10 sources by walking the package rather than listing it.

    ``rglob`` rather than a hand-maintained tuple: a list stops covering a module
    on the day someone adds one, which is exactly the change a fingerprint exists
    to catch. ``__pycache__`` is skipped because it is derived, not source.

    ``generator_path`` is a parameter because M10 now has two generators, the
    synthetic replay and the hardware runner, and the manifest should name the
    one that ran rather than the one this module was written for.
    """
    package = root / PACKAGE_ROOT
    if not package.is_dir():
        raise FileNotFoundError(f"simulator package not found at {package}")
    modules = sorted(
        item.relative_to(root).as_posix()
        for item in package.rglob("*.py")
        if "__pycache__" not in item.parts
    )
    if not modules:
        raise FileNotFoundError(f"no modules found under {package}")
    return (generator_path, *modules)


def source_fingerprints(
    root: Path, generator_path: str = GENERATOR_PATH
) -> dict[str, str]:
    """Map each M10 source path to the SHA-256 of its bytes on disk."""
    fingerprints: dict[str, str] = {}
    for relative in m10_source_paths(root, generator_path):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"M10 source {relative} not found at {path}")
        fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def combined_digest(fingerprints: Mapping[str, str]) -> str:
    """Fold per-file digests into one value that also covers the file set.

    Paths are hashed alongside their digests and separated by a byte that cannot
    occur in either, so renaming a file or dropping one from the set changes the
    result instead of colliding with it.
    """
    digest = hashlib.sha256()
    for relative in sorted(fingerprints):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fingerprints[relative].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def source_manifest(root: Path, generator_path: str = GENERATOR_PATH) -> dict[str, Any]:
    """Build the provenance block that records the fingerprinted sources."""
    fingerprints = source_fingerprints(root, generator_path)
    return {
        "algorithm": ALGORITHM,
        "generator": generator_path,
        "reproducibility_claim": REPRODUCIBILITY_CLAIM,
        "note": REPRODUCIBILITY_NOTE,
        "combined_digest": combined_digest(fingerprints),
        "file_count": len(fingerprints),
        "files": fingerprints,
        "runtime": runtime_identity(),
    }
