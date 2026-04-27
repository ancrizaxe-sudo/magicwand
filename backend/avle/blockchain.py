"""Local EVM blockchain logging via eth-tester + py-evm.

A real in-process Ethereum Virtual Machine (py-evm) is spun up via
`web3.EthereumTesterProvider`.  Each analysis record is committed as a
transaction whose `input` field contains the hex-encoded payload.  The
transaction hash is the immutable proof.

No wallet interaction, no external node, no API keys.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List

from eth_tester import EthereumTester
from web3 import EthereumTesterProvider, Web3

log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {}


def _get_w3() -> Web3:
    if "w3" in _STATE:
        return _STATE["w3"]
    tester = EthereumTester()
    w3 = Web3(EthereumTesterProvider(tester))
    # Default account, pre-funded by eth-tester
    w3.eth.default_account = w3.eth.accounts[0]
    _STATE["tester"] = tester
    _STATE["w3"] = w3
    _STATE["records"]: List[Dict[str, Any]] = []
    _STATE["genesis_block"] = int(w3.eth.block_number)
    log.info("Local EVM chain started, chain_id=%s, genesis_block=%s",
             w3.eth.chain_id, _STATE["genesis_block"])
    return w3


def log_analysis(record: Dict[str, Any]) -> Dict[str, Any]:
    """Commit `record` to the local chain as a zero-value self-transaction
    whose data field is the JSON payload.  Returns on-chain proof."""
    with _LOCK:
        w3 = _get_w3()
        payload = json.dumps(record, separators=(",", ":"), default=str).encode()
        tx_hash = w3.eth.send_transaction({
            "from":  w3.eth.default_account,
            "to":    w3.eth.default_account,  # self-transfer
            "value": 0,
            "gas":   200_000 + len(payload) * 68,
            "data":  "0x" + payload.hex(),
        })
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        proof = {
            "tx_hash":      receipt["transactionHash"].hex(),
            "block_number": int(receipt["blockNumber"]),
            "block_hash":   receipt["blockHash"].hex(),
            "gas_used":     int(receipt["gasUsed"]),
            "from":         receipt["from"],
            "payload_size": len(payload),
            "chain_id":     int(w3.eth.chain_id),
        }
        _STATE["records"].append({"proof": proof, "record": record})
        return proof


def list_records(limit: int = 50) -> List[Dict[str, Any]]:
    _get_w3()
    return list(reversed(_STATE.get("records", [])[-limit:]))


def chain_status() -> Dict[str, Any]:
    w3 = _get_w3()
    return {
        "chain_id":       int(w3.eth.chain_id),
        "latest_block":   int(w3.eth.block_number),
        "genesis_block":  int(_STATE.get("genesis_block", 0)),
        "account":        w3.eth.default_account,
        "balance_wei":    int(w3.eth.get_balance(w3.eth.default_account)),
        "record_count":   len(_STATE.get("records", [])),
        "provider":       "eth-tester (py-evm, in-process)",
    }


def verify(tx_hash: str) -> Dict[str, Any]:
    w3 = _get_w3()
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        tx = w3.eth.get_transaction(tx_hash)
    except Exception as e:
        return {"valid": False, "error": str(e)}
    data_hex = tx["input"].hex() if hasattr(tx["input"], "hex") else str(tx["input"])
    # input is '0x..'; strip prefix and decode
    raw = data_hex[2:] if data_hex.startswith("0x") else data_hex
    try:
        payload = bytes.fromhex(raw).decode()
        record = json.loads(payload)
    except Exception:
        record = None
    return {
        "valid":        True,
        "block_number": int(receipt["blockNumber"]),
        "block_hash":   receipt["blockHash"].hex(),
        "gas_used":     int(receipt["gasUsed"]),
        "record":       record,
    }
