from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

VERSION = "1.0.0"


@dataclass
class WorkBudget:
    max_jobs: int | None = None
    max_minutes: float | None = None
    _used: int = field(init=False, default=0)
    _deadline: float | None = field(init=False, default=None)

    def __post_init__(self):
        if self.max_jobs is not None and self.max_jobs < 1:
            raise ValueError("--max-jobs must be at least 1")
        if self.max_minutes is not None:
            if not isfinite(self.max_minutes) or self.max_minutes <= 0:
                raise ValueError("--max-minutes must be positive and finite")
            self._deadline = monotonic() + 60 * self.max_minutes

    def take(self):
        if self.max_jobs is not None and self._used >= self.max_jobs:
            return False
        if self._deadline is not None and monotonic() >= self._deadline:
            return False
        self._used += 1
        return True


@contextmanager
def work_lock(root, stage, split):
    import fcntl

    path = Path(root) / ".locks" / f"{stage}-{split}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A {stage} runner is already active for {split}") from exc
        yield


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def fingerprint(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        from hashlib import file_digest
        return file_digest(stream, "sha256").hexdigest()


@contextmanager
def atomic_file(path, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if "b" in mode else {"encoding": "utf-8", "newline": "\n"}
    with NamedTemporaryFile(mode, dir=path.parent, delete=False, **options) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path) as stream:
        stream.write(canonical_json(value) + "\n")


def write_jsonl(path, rows):
    with atomic_file(path) as stream:
        for row in rows:
            stream.write(canonical_json({"schema_version": VERSION, **row}) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("schema_version") != VERSION:
                raise ValueError(f"Unsupported schema in {path}")
            yield row
