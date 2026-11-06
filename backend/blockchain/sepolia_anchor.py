"""Publish one investigation hash to the AuditAnchor contract on Sepolia.

The local hash chain in audit_chain.py is still the full record. This module
only sends that same SHA-256 to Ethereum's public testnet, so a third party
can see the hash existed at that time. Raw meter readings are never in the
transaction.

Config, all in backend/.env (never commit that file):
    SEPOLIA_RPC_URL
    SEPOLIA_PRIVATE_KEY
    ANCHOR_CONTRACT_ADDRESS
"""

import os

from dotenv import load_dotenv
from web3 import Web3

SEPOLIA_CHAIN_ID = 11155111

ABI = [
    {
        "inputs": [{"name": "payloadHash", "type": "bytes32"}],
        "name": "anchor",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "", "type": "bytes32"}],
        "name": "anchoredAt",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


class SepoliaNotConfigured(Exception):
    pass


def _settings():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    url = os.getenv("SEPOLIA_RPC_URL", "").strip()
    key = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
    address = os.getenv("ANCHOR_CONTRACT_ADDRESS", "").strip()
    if not url or not key or not address:
        raise SepoliaNotConfigured(
            "Set SEPOLIA_RPC_URL, SEPOLIA_PRIVATE_KEY, and ANCHOR_CONTRACT_ADDRESS in backend/.env"
        )
    return url, key, address


def _web3():
    url, key, address = _settings()
    w3 = Web3(Web3.HTTPProvider(url))
    if not w3.is_connected():
        raise ConnectionError("Could not reach the Sepolia RPC URL")
    chain_id = w3.eth.chain_id
    if chain_id != SEPOLIA_CHAIN_ID:
        raise ConnectionError(f"RPC is chain {chain_id}, expected Sepolia ({SEPOLIA_CHAIN_ID})")
    account = w3.eth.account.from_key(key)
    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=ABI)
    return w3, account, contract


def _hash_bytes(payload_hash: str) -> bytes:
    text = payload_hash.removeprefix("0x")
    if len(text) != 64:
        raise ValueError("payload hash must be 32 bytes of hex")
    return bytes.fromhex(text)


def anchor_hash(payload_hash: str) -> str:
    """Send the hash. Returns the transaction hash once Sepolia accepts it."""
    w3, account, contract = _web3()
    digest = _hash_bytes(payload_hash)
    tx = contract.functions.anchor(digest).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": SEPOLIA_CHAIN_ID,
    })
    signed = account.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    sent = w3.eth.send_raw_transaction(raw)
    hx = sent.hex()
    return hx if hx.startswith("0x") else "0x" + hx


def is_anchored(payload_hash: str) -> bool:
    """True when this hash is already stored in the contract."""
    _, _, contract = _web3()
    return contract.functions.anchoredAt(_hash_bytes(payload_hash)).call() > 0
