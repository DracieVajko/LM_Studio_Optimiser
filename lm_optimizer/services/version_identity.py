"""Version identity for revalidation decisions.

The pre-fingerprint code stored the first non-empty line of ``lms --version``
in a field named ``lm_studio_version``. That line is a build identifier of the
``lms`` CLI binary (e.g. ``CLI commit: dcefd5a``), not an LM Studio semantic
version, and it moves on every CLI rebuild. Fingerprinting it directly caused
spurious revalidation: a documentation change in the CLI looked like a change
in the inference host.

This module separates the three things that were conflated:

``raw``
    The exact string, kept for diagnostics and shown in reports.
``semantic``
    An x.y.z version, extracted ONLY when the source really provides one and
    the extraction is unambiguous. Never fabricated.
``build``
    A commit/build identifier when that is all the source offers.

``compatibility_key`` is what revalidation actually compares. A real version
change always matters. A bare build identifier matters only when the build
actually performs loads for us (the ``lms`` load channel), because otherwise a
CLI rebuild cannot change a measurement taken over REST.

An unknown or malformed version yields ``unknown``, which is deliberately
stable: an unreadable version must not trigger endless revalidation, and it
must not be invented into a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# x.y.z with optional pre-release/build metadata.
_SEMVER = re.compile(r"(?<!\d)(\d{1,4})\.(\d{1,4})\.(\d{1,4})(?:[-+][0-9A-Za-z.\-+]+)?")

# 7+ hex chars, optionally prefixed with a word (commit/sha/build).
_BUILD_HASH = re.compile(r"(?i)\b(?:commit|sha|build|revision)\b[:\s#]*([0-9a-f]{7,40})\b")
_BARE_HASH = re.compile(r"(?<![0-9a-f])([0-9a-f]{7,40})(?![0-9a-f])")

UNKNOWN = "unknown"


@dataclass(frozen=True)
class VersionIdentity:
    """Normalized identity of a version-ish string.

    ``compatibility_key`` is the revalidation input; ``raw`` is diagnostics.
    """

    raw: str
    semantic: str | None
    build: str | None
    source: str
    compatibility_key: str

    @property
    def is_semantic(self) -> bool:
        return self.semantic is not None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "semantic": self.semantic,
            "build": self.build,
            "source": self.source,
            "compatibility_key": self.compatibility_key,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "VersionIdentity | None":
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                raw=str(data.get("raw", UNKNOWN)),
                semantic=data.get("semantic"),
                build=data.get("build"),
                source=str(data.get("source", UNKNOWN)),
                compatibility_key=str(data.get("compatibility_key", UNKNOWN)),
            )
        except Exception:
            return None


def identity_from_version_string(
    version_text: str | None,
    *,
    source: str = "lms-cli",
    cli_performs_loads: bool = False,
) -> VersionIdentity:
    """Normalize a raw version string into a :class:`VersionIdentity`.

    ``cli_performs_loads`` reflects whether the ``lms`` CLI is used to load
    models for this workload. Only then can a CLI build change a measurement,
    so only then does a commit-only identity take part in revalidation.
    """
    raw = (version_text or "").strip() or UNKNOWN

    if raw.startswith("unknown"):
        return VersionIdentity(
            raw=raw,
            semantic=None,
            build=None,
            source=source,
            compatibility_key=UNKNOWN,
        )

    sem = None
    matches = _SEMVER.findall(raw)
    full_matches = _SEMVER.finditer(raw)
    distinct = {m.group(0) for m in full_matches}
    if len(distinct) == 1:
        m = _SEMVER.search(raw)
        if m:
            sem = m.group(0)

    build = None
    bm = _BUILD_HASH.search(raw)
    if bm:
        build = bm.group(1).lower()
    else:
        for m in _BARE_HASH.finditer(raw):
            token = m.group(1)
            if len(token) >= 7:
                build = token.lower()
                break

    if sem:
        compat = f"semver:{sem}"
    elif build and cli_performs_loads:
        compat = f"build:{build}"
    elif build:
        # Build metadata for a CLI we do not load through: recorded, but not a
        # reason to invalidate measurements.
        compat = "build:non-loading"
    else:
        compat = f"unversioned:{_BARE_HASH.sub('', raw).strip()[:40] or UNKNOWN}"

    return VersionIdentity(
        raw=raw[:120],
        semantic=sem,
        build=build,
        source=source,
        compatibility_key=compat,
    )


def lms_identity(cli_performs_loads: bool = False) -> VersionIdentity:
    """Normalized identity of the local ``lms`` CLI."""
    from lm_optimizer.services import lms_cli

    try:
        text = lms_cli.lms_version()
    except Exception as e:  # pragma: no cover - defensive
        return VersionIdentity(
            raw=f"unknown (error: {type(e).__name__})",
            semantic=None,
            build=None,
            source="lms-cli",
            compatibility_key=UNKNOWN,
        )
    return identity_from_version_string(
        text, source="lms-cli", cli_performs_loads=cli_performs_loads
    )