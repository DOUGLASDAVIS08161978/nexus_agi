// Check wallet balance on Linea network
import hre from "hardhat";
import dotenv from "dotenv";

dotenv.config();

async function main() {
  const walletAddress = process.env.WALLET_ADDRESS;

  if (!walletAddress) {
    console.error("WALLET_ADDRESS not set in .env");
    process.exit(1);
  }

  const balance = await hre.ethers.provider.getBalance(walletAddress);
  const balanceInEth = hre.ethers.formatEther(balance);

  console.log(balanceInEth);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
