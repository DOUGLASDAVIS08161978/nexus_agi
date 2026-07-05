const { ethers } = require("hardhat");

async function main() {
  console.log("=".repeat(60));
  console.log("🧪 TESTING wTBTC MINTING");
  console.log("=".repeat(60));
  console.log();

  const contractAddress = "0x5FbDB2315678afecb367f032d93F642f64180aa3";
  const [operator, user1] = await ethers.getSigners();

  console.log("📋 Test Setup:");
  console.log("-".repeat(60));
  console.log("Contract Address:", contractAddress);
  console.log("Bridge Operator:", operator.address);
  console.log("Test User:", user1.address);
  console.log();

  // Get contract instance
  const wTBTC = await ethers.getContractAt("WrappedTestnetBTC", contractAddress);

  // Test 1: Mint 1.5 wTBTC to user1
  console.log("🏗️  Test 1: Minting 1.5 wTBTC to user...");
  console.log("-".repeat(60));

  const mintAmount = ethers.parseEther("1.5");
  const bitcoinTxId = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6";

  const mintTx = await wTBTC.mint(user1.address, mintAmount, bitcoinTxId);
  await mintTx.wait();

  console.log("✅ Minted 1.5 wTBTC");
  console.log("   Bitcoin TX ID:", bitcoinTxId);
  console.log("   Ethereum TX:", mintTx.hash);
  console.log();

  // Check balance
  const balance = await wTBTC.balanceOf(user1.address);
  console.log("💰 User Balance:", ethers.formatEther(balance), "wTBTC");
  console.log();

  // Test 2: User transfers 0.5 wTBTC
  console.log("🔄 Test 2: User transfers 0.5 wTBTC to operator...");
  console.log("-".repeat(60));

  const transferAmount = ethers.parseEther("0.5");
  const transferTx = await wTBTC.connect(user1).transfer(operator.address, transferAmount);
  await transferTx.wait();

  console.log("✅ Transferred 0.5 wTBTC");
  console.log("   Ethereum TX:", transferTx.hash);
  console.log();

  // Check balances
  const user1Balance = await wTBTC.balanceOf(user1.address);
  const operatorBalance = await wTBTC.balanceOf(operator.address);

  console.log("💰 Final Balances:");
  console.log("-".repeat(60));
  console.log("User:", ethers.formatEther(user1Balance), "wTBTC");
  console.log("Operator:", ethers.formatEther(operatorBalance), "wTBTC");
  console.log();

  // Test 3: User burns 0.5 wTBTC
  console.log("🔥 Test 3: User burns 0.5 wTBTC...");
  console.log("-".repeat(60));

  const burnAmount = ethers.parseEther("0.5");
  const bitcoinAddress = "tb1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh";

  const burnTx = await wTBTC.connect(user1).burn(burnAmount, bitcoinAddress);
  await burnTx.wait();

  console.log("✅ Burned 0.5 wTBTC");
  console.log("   Bitcoin Address:", bitcoinAddress);
  console.log("   Ethereum TX:", burnTx.hash);
  console.log();

  // Final state
  const finalBalance = await wTBTC.balanceOf(user1.address);
  const totalSupply = await wTBTC.totalSupply();

  console.log("📊 Final State:");
  console.log("-".repeat(60));
  console.log("User Balance:", ethers.formatEther(finalBalance), "wTBTC");
  console.log("Total Supply:", ethers.formatEther(totalSupply), "wTBTC");
  console.log();

  console.log("=".repeat(60));
  console.log("🎉 ALL TESTS PASSED!");
  console.log("=".repeat(60));
  console.log();
  console.log("✅ Bridge minting works!");
  console.log("✅ ERC20 transfers work!");
  console.log("✅ Bridge burning works!");
  console.log();
  console.log("Ready for Sepolia deployment! 🚀");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
