"""Deterministic SHA-256 fingerprints of the sources that produce M9 artifacts.

The M9 generator and this package are untracked at the commit an artifact
records, so ``git_sha`` alone does not identify the code that produced it: two
artifacts can carry the same commit and dirty flag and still come from different
source bytes. Fingerprinting the sources closes that gap without overclaiming.

The claim is deliberately narrow. These digests, plus the recorded interpreter,
*identify* the source and runtime a run used. They do not promise that a
matching source tree reproduces the artifact bytes: that additionally requires
the same interpreter and the same input trace, neither of which a source digest
can enforce. This module publishes that limitation alongside the digests so it
travels with the numbers.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ALGORITHM = "sha256"

#: Repository-relative path of the M9 artifact generator.
GENERATOR_PATH = "scripts/run_m9_synthetic.py"

#: Repository-relative directory of the calibration package.
PACKAGE_DIR = "src/prefill_cache_sim/calibration"

#: Non-calibration modules the generator loads to produce its published numbers:
#: ``trace`` parses and digests the Mooncake trace, ``identity`` derives the
#: request ids ``trace`` records, and ``config`` supplies the git provenance
#: block. Editing any of them can change an artifact while every calibration
#: module stays byte-identical, so they are fingerprinted too.
#:
#: This is a curated list, not a computed import closure: it names the files it
#: covers and claims nothing about modules reached indirectly from them.
DEPENDENCY_PATHS: tuple[str, ...] = (
    "src/prefill_cache_sim/config.py",
    "src/prefill_cache_sim/identity.py",
    "src/prefill_cache_sim/trace.py",
)

#: What the fingerprints do and do not promise, carried in the artifact so the
#: claim travels with the numbers rather than living only in a review comment.
REPRODUCIBILITY_CLAIM = "SOURCE_AND_RUNTIME_IDENTIFIED_REPRODUCTION_CONDITIONAL"

REPRODUCIBILITY_NOTE = (
    "The M9 generator, the calibration package and its listed dependencies may "
    "be untracked at the recorded commit, so git_sha and git_dirty do not "
    "identify the code that produced this artifact. These digests identify the "
    "source files, and the recorded runtime identifies the interpreter. They do "
    "not by themselves make the artifact reproducible: byte-for-byte "
    "reproduction remains conditional on running that source under a matching "
    "Python runtime against the trace whose sha256 this artifact records. The "
    "combined_digest covers the source file set only, not the runtime or the "
    "trace, both of which are recorded separately."
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


def m9_source_paths(
    root: Path, generator_path: str = GENERATOR_PATH
) -> tuple[str, ...]:
    """Enumerate the M9 sources, discovering package modules rather than listing them.

    A hand-maintained list silently stops covering a module on the day someone
    adds one, which is exactly the change a fingerprint is supposed to catch, so
    the calibration package is globbed. The out-of-package dependencies in
    :data:`DEPENDENCY_PATHS` cannot be discovered the same way without dragging
    in the whole simulator, so they stay explicit.

    ``generator_path`` is a parameter rather than a constant because M9 now has
    more than one generator: the synthetic sweep and the hardware runner produce
    different artifacts from the same package. Recording whichever one ran keeps
    the manifest a description of the run instead of a description of the
    package.
    """
    package = root / PACKAGE_DIR
    if not package.is_dir():
        raise FileNotFoundError(f"calibration package not found at {package}")
    modules = sorted(item.name for item in package.glob("*.py"))
    return (
        generator_path,
        *(f"{PACKAGE_DIR}/{name}" for name in modules),
        *DEPENDENCY_PATHS,
    )


def source_fingerprints(
    root: Path, generator_path: str = GENERATOR_PATH
) -> dict[str, str]:
    """Map each M9 source path to the SHA-256 of its bytes on disk."""
    fingerprints: dict[str, str] = {}
    for relative in m9_source_paths(root, generator_path):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"M9 source {relative} not found at {path}")
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
        "files": fingerprints,
        "runtime": runtime_identity(),
    }
