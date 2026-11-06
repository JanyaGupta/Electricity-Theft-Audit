"""Deploy AuditAnchor.sol to Sepolia and write the address into backend/.env.

Run from backend/, after SEPOLIA_RPC_URL and SEPOLIA_PRIVATE_KEY are set:

    python -m blockchain.deploy_anchor
"""

import os

from dotenv import load_dotenv
from solcx import compile_source, install_solc, set_solc_version
from solcx import install as solcx_install
from web3 import Web3

from blockchain.sepolia_anchor import SEPOLIA_CHAIN_ID

# py-solc-x still downloads from solc-bin.ethereum.org, which does not resolve
# on this network. binaries.soliditylang.org is the current official host.
solcx_install.BINARY_DOWNLOAD_BASE = "https://binaries.soliditylang.org/{}-amd64/{}"

HERE = os.path.dirname(__file__)
ENV_PATH = os.path.join(HERE, "..", ".env")
SOURCE_PATH = os.path.join(HERE, "AuditAnchor.sol")


def main():
    load_dotenv(ENV_PATH)
    url = os.getenv("SEPOLIA_RPC_URL", "").strip()
    key = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
    if not url or not key:
        raise SystemExit("Put SEPOLIA_RPC_URL and SEPOLIA_PRIVATE_KEY in backend/.env first.")

    w3 = Web3(Web3.HTTPProvider(url))
    if not w3.is_connected():
        raise SystemExit("Could not reach SEPOLIA_RPC_URL")
    if w3.eth.chain_id != SEPOLIA_CHAIN_ID:
        raise SystemExit(f"That URL is chain {w3.eth.chain_id}, not Sepolia.")

    install_solc("0.8.20")
    set_solc_version("0.8.20")
    with open(SOURCE_PATH, encoding="utf-8") as f:
        compiled = compile_source(f.read(), output_values=["abi", "bin"])
    contract = compiled["<stdin>:AuditAnchor"]

    account = w3.eth.account.from_key(key)
    print("Deploying from", account.address)
    artifact = w3.eth.contract(abi=contract["abi"], bytecode=contract["bin"])
    tx = artifact.constructor().build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": SEPOLIA_CHAIN_ID,
    })
    signed = account.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    sent = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(sent, timeout=180)
    if receipt.status != 1:
        raise SystemExit("Deploy transaction failed")
    address = receipt.contractAddress
    print("AuditAnchor deployed at", address)
    print("Transaction", sent.hex())

    _write_address(address)
    print("Saved ANCHOR_CONTRACT_ADDRESS in backend/.env")


def _write_address(address: str):
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            lines = f.read().splitlines()
    kept = [line for line in lines if not line.startswith("ANCHOR_CONTRACT_ADDRESS=")]
    kept.append(f"ANCHOR_CONTRACT_ADDRESS={address}")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")


if __name__ == "__main__":
    main()
