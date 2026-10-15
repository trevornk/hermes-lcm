"""Regression coverage: hardening SQLite artifacts must not drop SQLite's locks.

POSIX advisory locks belong to the *process*, not the descriptor: closing ANY
descriptor that refers to a file silently releases every lock the process holds
on it (https://www.sqlite.org/howtocorrupt.html, section 2.2). The artifact
hardening helpers used to ``os.open``/``os.close`` the database and its
``-wal``/``-shm`` sidecars on every store construction. Inside a process that
already held SQLite connections (a long-lived gateway), that cancelled the
connections' locks; the next sibling process to close its own connection then
took the exclusive lock, checkpointed, and *unlinked* the ``-wal``/``-shm`` the
gateway was still using. New openers recreated them, so two generations wrote
one database file and the file's page versions diverged.

These tests assert the observable invariant (a second process opening and
closing the database must not unlink a live holder's sidecars) plus the
mechanism (an already-private artifact is never opened at all).
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

import hermes_lcm.sqlite_util as sqlite_util_module
from hermes_lcm.store import MessageStore

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX fcntl lock semantics")

_SIBLING = (
    "import sqlite3, sys\n"
    "c = sqlite3.connect(sys.argv[1])\n"
    "c.execute('PRAGMA journal_mode=WAL')\n"
    "c.execute('SELECT count(*) FROM sqlite_master').fetchone()\n"
    "c.close()\n"
)


def _sibling_opens_and_closes(db_path: Path) -> None:
    """Model a short-lived CLI/health-probe process touching the same database."""
    subprocess.run([sys.executable, "-c", _SIBLING, str(db_path)], check=True, timeout=60)


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino)


def _wal_holder(db_path: Path) -> sqlite3.Connection:
    """A connection with live WAL state, like the gateway's long-lived store."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT)")
    conn.execute("INSERT OR REPLACE INTO t VALUES ('a', '1')")
    conn.commit()
    conn.execute("SELECT count(*) FROM t").fetchone()
    for artifact in (db_path, db_path.with_name(db_path.name + "-wal")):
        os.chmod(artifact, 0o600)
    return conn


def _assert_generation_survives_sibling(db_path: Path) -> None:
    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    before = (_identity(wal), _identity(shm))
    assert None not in before, "holder must have live -wal/-shm before the probe"

    _sibling_opens_and_closes(db_path)

    after = (_identity(wal), _identity(shm))
    assert after == before, (
        "a sibling process opening/closing the database unlinked or replaced the "
        "holder's -wal/-shm: the holder's SQLite locks were dropped"
    )


def test_control_sibling_close_leaves_live_wal_generation_alone(tmp_path):
    """Sanity: with no stray descriptor work, the sibling must not disturb the holder."""
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path)
    try:
        _assert_generation_survives_sibling(db_path)
    finally:
        holder.close()


@pytest.mark.parametrize(
    "helper",
    ["_restrict_existing_sqlite_artifacts", "_prepare_private_sqlite_file"],
)
def test_artifact_hardening_does_not_strand_live_wal_generation(tmp_path, helper):
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path)
    try:
        getattr(sqlite_util_module, helper)(db_path)
        _assert_generation_survives_sibling(db_path)
    finally:
        holder.close()


def test_second_store_in_same_process_does_not_strand_first_stores_wal(tmp_path):
    """The production shape: gateway holds a store, a cron job builds another."""
    db_path = tmp_path / "lcm.db"
    first = MessageStore(db_path)
    try:
        first.append("s", {"role": "user", "content": "alpha"})
        first.commit()
        second = MessageStore(db_path)  # _init_db re-runs the artifact hardening
        try:
            _assert_generation_survives_sibling(db_path)
        finally:
            second.close()
    finally:
        first.close()


@pytest.mark.parametrize(
    "helper",
    ["_restrict_existing_sqlite_artifacts", "_prepare_private_sqlite_file"],
)
def test_already_private_artifacts_are_never_opened(tmp_path, monkeypatch, helper):
    """Mechanism: no descriptor on a db/-wal/-shm means no lock can be dropped."""
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path)
    try:
        artifacts = {
            db_path.name,
            db_path.name + "-wal",
            db_path.name + "-shm",
        }
        for name in artifacts:
            os.chmod(tmp_path / name, 0o600)

        opened: list[str] = []
        real_open = os.open

        def recording_open(path, flags, mode=0o777, *, dir_fd=None):
            opened.append(os.fspath(path))
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(sqlite_util_module.os, "open", recording_open)
        getattr(sqlite_util_module, helper)(db_path)

        assert not artifacts.intersection(opened), (
            f"{helper} opened SQLite artifacts that were already mode 0600: {sorted(artifacts.intersection(opened))}"
        )
    finally:
        holder.close()


def test_loose_artifacts_are_still_tightened(tmp_path):
    """The skip must not weaken hardening: wrong modes are still corrected."""
    db_path = tmp_path / "lcm.db"
    holder = _wal_holder(db_path)
    try:
        wal = db_path.with_name(db_path.name + "-wal")
        for artifact in (db_path, wal):
            os.chmod(artifact, 0o644)
        sqlite_util_module._restrict_existing_sqlite_artifacts(db_path)
        assert {p.name: p.stat().st_mode & 0o777 for p in (db_path, wal)} == {
            db_path.name: 0o600,
            wal.name: 0o600,
        }
    finally:
        holder.close()
