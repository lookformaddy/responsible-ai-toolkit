"""Append-only, hash-chained audit log (tamper-evident, not tamper-proof).

Each entry stores the SHA-256 hash of the previous entry's hash plus its
own payload, so any edit, deletion, or reordering after the fact breaks
verification at exactly one index. This is a local governance control:
it detects tampering by anyone with file access, it does not stop a
party who can rewrite the whole file (for that, anchor hashes externally
or use an append-only store -- see the README trade-offs).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

GENESIS_HASH = "0" * 64


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def hash_entry(
    index: int,
    timestamp: str,
    event_type: str,
    payload: dict,
    prev_hash: str,
) -> str:
    """SHA-256 over the canonical entry contents (excluding its own hash)."""
    material = f"{index}|{timestamp}|{event_type}|{_canonical(payload)}|{prev_hash}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditEntry:
    index: int
    timestamp: str
    event_type: str
    payload: dict
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuditEntry":
        return cls(
            index=int(data["index"]),
            timestamp=str(data["timestamp"]),
            event_type=str(data["event_type"]),
            payload=dict(data["payload"]),
            prev_hash=str(data["prev_hash"]),
            entry_hash=str(data["entry_hash"]),
        )


class AuditLog:
    """In-memory hash chain with JSONL persistence."""

    def __init__(self, entries: list[AuditEntry] | None = None) -> None:
        self._entries: list[AuditEntry] = list(entries) if entries else []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def append(
        self,
        event_type: str,
        payload: dict,
        timestamp: str | None = None,
    ) -> AuditEntry:
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        index = len(self._entries)
        entry_hash = hash_entry(index, timestamp, event_type, payload, prev_hash)
        entry = AuditEntry(index, timestamp, event_type, payload, prev_hash, entry_hash)
        self._entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, int | None]:
        """Return ``(ok, first_bad_index)``; ``first_bad_index`` is None if ok."""
        prev_hash = GENESIS_HASH
        for entry in self._entries:
            expected = hash_entry(
                entry.index, entry.timestamp, entry.event_type, entry.payload, prev_hash
            )
            if entry.prev_hash != prev_hash or entry.entry_hash != expected:
                return False, entry.index
            prev_hash = entry.entry_hash
        # indices must be contiguous from 0
        if [e.index for e in self._entries] != list(range(len(self._entries))):
            for i, e in enumerate(self._entries):
                if e.index != i:
                    return False, i
        return True, None

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(e.to_dict(), sort_keys=True) for e in self._entries
        ) + ("\n" if self._entries else "")

    @classmethod
    def from_jsonl(cls, text: str) -> "AuditLog":
        entries = [
            AuditEntry.from_dict(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]
        return cls(entries)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_jsonl())

    @classmethod
    def load(cls, path: str) -> "AuditLog":
        with open(path, encoding="utf-8") as fh:
            return cls.from_jsonl(fh.read())
