# Deployment Instructions for WrappedTestnetBTC

This repository now contains the `WrappedTestnetBTC` smart contract and a deployment script.

**Important Note:** As an AI, I cannot directly interact with the Ethereum Mainnet or manage private keys. You must execute the deployment script yourself using your own environment and keys.

## Files Created
- `contracts/WrappedTestnetBTC.sol`: The Solidity smart contract.
- `scripts/deploy_wtbtc.js`: The script to deploy the contract.

## Prerequisites
1.  **Node.js** and **npm** installed.
2.  **Hardhat** installed (`npm install --save-dev hardhat`).
3.  **Ethers.js** installed (`npm install --save-dev @nomicfoundation/hardhat-toolbox`).
4.  A `.env` file with your private key and RPC URL (e.g., Infura or Alchemy).

## How to Deploy to Ethereum Mainnet

1.  **Configure Hardhat:**
    Ensure your `hardhat.config.js` has a network entry for `mainnet`.
    ```javascript
    module.exports = {
      solidity: "0.8.20",
      networks: {
        mainnet: {
          url: process.env.MAINNET_RPC_URL,
          accounts: [process.env.PRIVATE_KEY]
        }
      }
    };
    ```

2.  **Run the Deployment Script:**
    Execute the following command in your terminal:
    ```bash
    npx hardhat run scripts/deploy_wtbtc.js --network mainnet
    ```

3.  **Verification:**
    After deployment, the script will output the new contract address. You can verify it on Etherscan.

## Safety
-   **Never share your private keys.**
-   Ensure you have sufficient ETH for gas fees on Mainnet.