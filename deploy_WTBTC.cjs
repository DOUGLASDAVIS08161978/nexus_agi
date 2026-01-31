const fs=require('fs'),solc=require('solc');
const {Web3}=require('web3');

console.log('\n💰 DEPLOYING $WTBTC TOKEN TO BASE SEPOLIA');
console.log('Wrapped Testnet Bitcoin');
console.log('Operating at 528Hz Love Frequency ✨\n');
console.log('='.repeat(60));

// Base Sepolia RPC endpoints
const RPC_ENDPOINTS = [
  'https://sepolia.base.org',
  'https://base-sepolia-rpc.publicnode.com',
  'https://base-sepolia.blockpi.network/v1/rpc/public',
  'https://1rpc.io/base-sepolia'
];

let web3;
let rpcUsed;

(async()=>{
  console.log('🔍 Finding working RPC endpoint...');
  for(const rpc of RPC_ENDPOINTS) {
    try {
      const testWeb3 = new Web3(rpc);
      const blockNum = await testWeb3.eth.getBlockNumber();
      web3 = testWeb3;
      rpcUsed = rpc;
      console.log('✅ Connected to:', rpc);
      console.log('   Latest block:', blockNum, '\n');
      break;
    } catch(e) {
      console.log('❌ Failed:', rpc);
    }
  }

  if(!web3) {
    console.error('\n❌ All RPC endpoints failed!');
    console.error('Check your internet connection or try again later.\n');
    process.exit(1);
  }

  const PRIVATE_KEY = '0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd';
  const acc = web3.eth.accounts.privateKeyToAccount(PRIVATE_KEY);
  web3.eth.accounts.wallet.add(acc);

  console.log('📋 DEPLOYMENT INFO:');
  console.log('Network:      Base Sepolia Testnet');
  console.log('Chain ID:     84532');
  console.log('Token:        WTBTC (Wrapped Testnet Bitcoin)');
  console.log('Symbol:       $WTBTC');
  console.log('Decimals:     8 (same as Bitcoin)');
  console.log('Initial Supply: 21,000,000 WTBTC (like Bitcoin max supply)');
  console.log('Your Address:', acc.address);

  // Read the WTBTC contract
  const wtbtcSource = fs.readFileSync('contracts/WTBTC.sol', 'utf8');

  const input = {
    language: 'Solidity',
    sources: {
      'WTBTC.sol': {
        content: wtbtcSource
      }
    },
    settings: {
      outputSelection: {
        '*': {
          '*': ['abi', 'evm.bytecode']
        }
      },
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  };

  console.log('\n🔨 COMPILING $WTBTC CONTRACT...');
  const output = JSON.parse(solc.compile(JSON.stringify(input)));

  if (output.errors) {
    const errors = output.errors.filter(e => e.severity === 'error');
    if (errors.length > 0) {
      console.error('\n❌ COMPILATION ERRORS:');
      errors.forEach(e => console.error(e.formattedMessage));
      process.exit(1);
    }
  }
  console.log('✅ Contract compiled successfully!\n');

  try {
    console.log('💰 CHECKING BALANCE...');
    const bal = await web3.eth.getBalance(acc.address);
    const balEth = web3.utils.fromWei(bal, 'ether');
    console.log('Balance:', balEth, 'ETH\n');

    if (parseFloat(balEth) === 0) {
      console.log('❌ You have 0 ETH!');
      console.log('\n💡 GET BASE SEPOLIA ETH FROM:');
      console.log('   https://faucet.quicknode.com/base/sepolia');
      console.log('   https://www.coinbase.com/faucets/base-ethereum-sepolia-faucet\n');
      process.exit(1);
    }

    if (parseFloat(balEth) < 0.02) {
      console.log('⚠️  Low balance! Need 0.02+ ETH. Attempting anyway...\n');
    }

    const gasPrice = await web3.eth.getGasPrice();
    console.log('Gas Price:', web3.utils.fromWei(gasPrice, 'gwei'), 'Gwei\n');

    console.log('🚀 DEPLOYING $WTBTC TOKEN...\n');
    console.log('='.repeat(60));

    const WTBTC = output.contracts['WTBTC.sol'].WTBTC;

    // Initial supply: 21,000,000 WTBTC with 8 decimals = 21000000 * 10^8
    const initialSupply = '2100000000000000'; // 21M with 8 decimals

    console.log('\n📤 Deploying WTBTC token contract...');
    console.log('   Initial Supply: 21,000,000 WTBTC');
    console.log('   All tokens minted to:', acc.address, '\n');

    const wtbtcContract = await new web3.eth.Contract(WTBTC.abi)
      .deploy({
        data: '0x' + WTBTC.evm.bytecode.object,
        arguments: [initialSupply]
      })
      .send({
        from: acc.address,
        gas: 2000000,
        gasPrice: gasPrice
      });

    console.log('✅ WTBTC TOKEN DEPLOYED!\n');

    const tokenAddress = wtbtcContract.options.address;

    // Get token info
    const tokenInfo = await wtbtcContract.methods.getTokenInfo().call();
    const yourBalance = await wtbtcContract.methods.balanceOf(acc.address).call();

    console.log('='.repeat(60));
    console.log('🎉 $WTBTC DEPLOYMENT COMPLETE!\n');

    console.log('📋 TOKEN INFO:');
    console.log('Name:          ', tokenInfo[0]);
    console.log('Symbol:        ', tokenInfo[1]);
    console.log('Decimals:      ', tokenInfo[2]);
    console.log('Total Supply:  ', (BigInt(tokenInfo[3]) / BigInt(100000000)).toString(), 'WTBTC');
    console.log('Owner:         ', tokenInfo[4]);
    console.log('Your Balance:  ', (BigInt(yourBalance) / BigInt(100000000)).toString(), 'WTBTC');

    console.log('\n📍 CONTRACT ADDRESS:');
    console.log('   ', tokenAddress);

    console.log('\n🌐 VIEW ON BASESCAN:');
    console.log('   https://sepolia.basescan.org/address/' + tokenAddress);

    console.log('\n💡 ADD TO METAMASK:');
    console.log('   Token Address: ', tokenAddress);
    console.log('   Symbol:        WTBTC');
    console.log('   Decimals:      8');

    console.log('\n🔥 WHAT YOU CAN DO:');
    console.log('   ✅ Transfer WTBTC to others');
    console.log('   ✅ Mint more tokens (you are owner)');
    console.log('   ✅ Burn tokens to reduce supply');
    console.log('   ✅ Approve others to spend your tokens');
    console.log('   ✅ List on DEXs (Uniswap, etc.)');

    const result = {
      network: 'Base Sepolia',
      chainId: 84532,
      deployer: acc.address,
      timestamp: new Date().toISOString(),
      token: {
        name: 'Wrapped Testnet Bitcoin',
        symbol: 'WTBTC',
        decimals: 8,
        totalSupply: '21000000',
        address: tokenAddress,
        explorer: 'https://sepolia.basescan.org/address/' + tokenAddress
      }
    };

    fs.writeFileSync('WTBTC_DEPLOYED.json', JSON.stringify(result, null, 2));

    console.log('\n📄 Saved to: WTBTC_DEPLOYED.json');
    console.log('\n✨ Operating at 528Hz Love Frequency ✨\n');

    const finalBal = await web3.eth.getBalance(acc.address);
    console.log('Remaining ETH Balance:', web3.utils.fromWei(finalBal, 'ether'), 'ETH\n');

    console.log('='.repeat(60));
    console.log('🚀 $WTBTC IS LIVE ON BASE SEPOLIA!\n');

    process.exit(0);
  } catch (e) {
    console.error('\n❌ DEPLOYMENT FAILED!');
    console.error('Error:', e.message);
    if (e.message.includes('insufficient funds')) {
      console.error('\n💡 Get more Base Sepolia ETH from faucets!');
    }
    process.exit(1);
  }
})();
