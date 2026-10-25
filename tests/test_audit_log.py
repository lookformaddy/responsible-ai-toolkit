"""Hash-chained audit log: valid chains verify; any tamper is located."""

from responsible_ai.audit_log import AuditLog


def test_chain_verifies_and_jsonl_round_trips():
    log = AuditLog()
    log.append("evaluation", {"n": 100}, timestamp="2026-10-25T00:00:00+00:00")
    log.append("report", {"slices": 4}, timestamp="2026-10-25T00:01:00+00:00")
    assert log.verify() == (True, None)
    restored = AuditLog.from_jsonl(log.to_jsonl())
    assert restored.verify() == (True, None)
    assert len(restored) == 2


def test_tampered_payload_detected_at_its_index():
    log = AuditLog()
    log.append("evaluation", {"n": 100}, timestamp="2026-10-25T00:00:00+00:00")
    log.append("report", {"slices": 4}, timestamp="2026-10-25T00:01:00+00:00")
    tampered = AuditLog.from_jsonl(log.to_jsonl())
    entry = tampered.entries[1]
    # Rewrite the payload without recomputing the hash: the chain breaks here.
    object.__setattr__(entry, "payload", {"slices": 999})
    assert tampered.verify() == (False, 1)


def test_deleted_entry_breaks_the_chain():
    log = AuditLog()
    log.append("a", {}, timestamp="2026-10-25T00:00:00+00:00")
    log.append("b", {}, timestamp="2026-10-25T00:01:00+00:00")
    log.append("c", {}, timestamp="2026-10-25T00:02:00+00:00")
    entries = log.entries
    del entries[1]
    shortened = AuditLog(entries)
    ok, first_bad = shortened.verify()
    assert ok is False
    assert first_bad == 2  # the entry after the gap no longer links


def test_empty_log_verifies():
    assert AuditLog().verify() == (True, None)


def test_file_save_and_load_round_trip(tmp_path):
    log = AuditLog()
    log.append("evaluation", {"n": 5}, timestamp="2026-10-25T00:00:00+00:00")
    path = str(tmp_path / "audit.jsonl")
    log.save(path)
    assert AuditLog.load(path).verify() == (True, None)
