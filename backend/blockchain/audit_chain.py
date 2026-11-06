"""
Step 9 — Tamper-evident audit trail (SHA-256 hash chain).

This is not a blockchain: there's no network, no consensus, no distributed
nodes. It's an append-only, hash-linked ledger — each block references the
previous block's hash, so altering any past record breaks the chain and is
detectable. That property is what the project brief actually needs (proof
that an investigation record wasn't quietly edited after the fact), so this
module implements exactly that, honestly labelled.

Only a hash of each investigation event is stored here — never raw customer
or meter data. The chain is persisted to backend/data/audit_chain.json so it
survives process restarts (this file is now the system of record for audit
anchors; there is no separate database behind it).

  1. AuditEvent — canonical, order-independent serialization of an event.
  2. Block      — SHA-256 hash of the event + link to the previous block.
  3. LocalChainClient — append (`submit`) and verify (`verify`) against the
     persisted chain. If you ever needed a real permissioned blockchain
     (e.g. Hyperledger Besu) instead, `submit`/`verify` are the only two
     methods the rest of the app calls, so that's the integration surface.
"""

from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class AuditEvent:
    alert_id: int
    investigation_id: int
    risk_score: float
    model_version: str
    investigation_outcome: str
    timestamp: str  # ISO-8601, set by the caller for determinism

    def canonical_payload(self) -> str:
        """Deterministic JSON (sorted keys, no whitespace) so the same
        event always hashes to the same digest regardless of field order."""
        return json.dumps(
            {
                "alert_id": self.alert_id,
                "investigation_id": self.investigation_id,
                "risk_score": self.risk_score,
                "model_version": self.model_version,
                "investigation_outcome": self.investigation_outcome,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()


@dataclass
class Block:
    index: int
    event_hash: str
    previous_hash: str
    created_at: float = field(default_factory=time.time)

    def block_hash(self) -> str:
        raw = f"{self.index}|{self.event_hash}|{self.previous_hash}|{self.created_at}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEFAULT_CHAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audit_chain.json")


class LocalChainClient:
    """
    Append-only, JSON-persisted hash chain. This is the actual audit
    ledger for the project (not a stand-in for a database) — it lives at
    `chain_path` and is loaded/saved on every write so it survives restarts.
    """

    def __init__(self, chain_path: str = DEFAULT_CHAIN_PATH):
        self.chain_path = chain_path
        self._chain: list[Block] = self._load()

    def _load(self) -> list[Block]:
        if os.path.exists(self.chain_path):
            with open(self.chain_path) as f:
                raw = json.load(f)
            return [Block(**b) for b in raw]
        genesis = Block(index=0, event_hash="0" * 64, previous_hash="0" * 64)
        return [genesis]

    def _save(self):
        os.makedirs(os.path.dirname(self.chain_path), exist_ok=True)
        with open(self.chain_path, "w") as f:
            json.dump([asdict(b) for b in self._chain], f, indent=2)

    def submit(self, event: AuditEvent) -> dict:
        """Anchor an audit event. Returns what gets stored alongside the
        investigation record (see app/datastore.py investigations.json)."""
        previous = self._chain[-1]
        block = Block(
            index=previous.index + 1,
            event_hash=event.hash(),
            previous_hash=previous.block_hash(),
        )
        self._chain.append(block)
        self._save()
        return {
            "payload_hash": event.hash(),
            "chain_tx_ref": f"block#{block.index}",
            # Hash of the previous block, so the dashboard can draw the real link.
            "previous_hash": block.previous_hash,
            "anchored_at": block.created_at,
        }

    def verify(self, event: AuditEvent, chain_tx_ref: str) -> bool:
        """Recompute the event's hash and compare it against what's anchored.
        A mismatch means the investigation record was altered after anchoring."""
        try:
            index = int(chain_tx_ref.replace("block#", ""))
            block = self._chain[index]
        except (ValueError, IndexError):
            return False
        return block.event_hash == event.hash()

    def is_chain_intact(self) -> bool:
        """Checks the hash-chain links themselves haven't been broken."""
        return self.first_broken_index() is None

    def first_broken_index(self):
        """Index of the first block whose previous_hash doesn't match the
        block before it, or None if every link holds."""
        for i in range(1, len(self._chain)):
            if self._chain[i].previous_hash != self._chain[i - 1].block_hash():
                return i
        return None

    def blocks(self) -> list[dict]:
        """Every block in append order, including the block's own hash and
        whether its link to the previous block holds. The dashboard draws
        the chain from this, so it always shows every block that exists
        (including earlier saves that were later superseded)."""
        out = []
        for i, b in enumerate(self._chain):
            link_ok = True if i == 0 else b.previous_hash == self._chain[i - 1].block_hash()
            out.append({
                "index": b.index,
                "event_hash": b.event_hash,
                "previous_hash": b.previous_hash,
                "block_hash": b.block_hash(),
                "created_at": b.created_at,
                "link_ok": link_ok,
            })
        return out


if __name__ == "__main__":
    # Minimal demo mirroring what the dashboard's "Audit chain" view shows.
    # Uses a throwaway path so running this file doesn't touch real data.
    client = LocalChainClient(chain_path="/tmp/demo_audit_chain.json")

    event = AuditEvent(
        alert_id=4821,
        investigation_id=2201,
        risk_score=0.87,
        model_version="rf-v2.3",
        investigation_outcome="confirmed",
        timestamp="2026-09-15T10:30:00Z",
    )
    anchor = client.submit(event)
    print("Anchored:", anchor)
    print("Verifies against original event:", client.verify(event, anchor["chain_tx_ref"]))

    # Simulate someone editing the investigation record after the fact
    tampered_event = AuditEvent(
        alert_id=4821,
        investigation_id=2201,
        risk_score=0.87,
        model_version="rf-v2.3",
        investigation_outcome="not_confirmed",  # outcome silently changed
        timestamp="2026-09-15T10:30:00Z",
    )
    print("Verifies after tampering:", client.verify(tampered_event, anchor["chain_tx_ref"]))
    print("Chain link integrity intact:", client.is_chain_intact())

