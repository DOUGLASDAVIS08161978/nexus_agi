const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  const balance = await hre.ethers.provider.getBalance(deployer.address);

  console.log(`Address: ${deployer.address}`);
  console.log(`Balance: ${hre.ethers.formatEther(balance)} ETH`);

  if (balance === 0n) {
    console.log("\n⚠️ WARNING: No ETH balance! Cannot deploy.");
    process.exit(1);
  } else if (balance < hre.ethers.parseEther("0.01")) {
    console.log("\n⚠️ WARNING: Low ETH balance. May not be enough for gas fees.");
  } else {
    console.log("\n✅ Sufficient balance for deployment");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
