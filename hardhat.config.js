require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

// Helper to validate private key
function getAccounts() {
  const privateKey = process.env.PRIVATE_KEY;

  // Check if private key is set and valid (64 characters for 32 bytes)
  if (privateKey && privateKey.length === 64) {
    return [privateKey];
  }

  // Return empty array if no valid private key (will use default Hardhat accounts)
  return [];
}

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    mainnet: {
      url: process.env.MAINNET_RPC_URL || "https://ethereum-rpc.publicnode.com",
      accounts: getAccounts(),
    },
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "https://ethereum-sepolia-rpc.publicnode.com",
      accounts: getAccounts(),
    },
    polygon: {
      url: process.env.POLYGON_RPC_URL || "https://polygon-rpc.com",
      accounts: getAccounts(),
      chainId: 137,
    },
    polygonMumbai: {
      url: process.env.POLYGON_MUMBAI_RPC_URL || "https://rpc-mumbai.maticvigil.com",
      accounts: getAccounts(),
      chainId: 80001,
    },
    hardhat: {
      // Local development network
      chainId: 31337,
    },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};
