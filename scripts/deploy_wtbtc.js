const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();

  console.log("Deploying contracts with the account:", deployer.address);

  // You can set the initial bridge operator to the deployer for simplicity,
  // or use a specific address from your environment variables.
  const initialOperator = process.env.BRIDGE_OPERATOR_ADDRESS || deployer.address;

  console.log("Setting initial bridge operator to:", initialOperator);

  const WrappedTestnetBTC = await hre.ethers.getContractFactory("WrappedTestnetBTC");
  const wTBTC = await WrappedTestnetBTC.deploy(initialOperator);

  // Wait for the deployment to complete
  await wTBTC.waitForDeployment();

  const contractAddress = await wTBTC.getAddress();
  console.log("WrappedTestnetBTC deployed to:", contractAddress);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
