// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// Stores the SHA-256 of an investigation outcome. The reading itself stays off-chain.
contract AuditAnchor {
    event Anchored(bytes32 indexed payloadHash, address indexed sender, uint256 timestamp);

    mapping(bytes32 => uint256) public anchoredAt;

    function anchor(bytes32 payloadHash) external {
        require(anchoredAt[payloadHash] == 0, "already anchored");
        anchoredAt[payloadHash] = block.timestamp;
        emit Anchored(payloadHash, msg.sender, block.timestamp);
    }
}
