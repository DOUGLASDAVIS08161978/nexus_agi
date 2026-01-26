
// hardhat.config.js
require("@nomiclabs/hardhat-waffle");
require("@nomiclabs/hardhat-etherscan");
require("dotenv").config();

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
    linea: {
      url: process.env.LINEA_RPC_URL,
      accounts: [process.env.PRIVATE_KEY],
      chainId: 59144
    },
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL,
      accounts: [process.env.PRIVATE_KEY],
      chainId: 11155111
    }
  },
  etherscan: {
    apiKey: {
      linea: process.env.LINEASCAN_API_KEY,
      sepolia: process.env.ETHERSCAN_API_KEY
    }
  }
};
