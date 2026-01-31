const fs=require('fs'),solc=require('solc');
const {Web3}=require('web3');

console.log('\n🌌 NEXUS AGI - DEPLOYING TO BASE SEPOLIA');
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
    console.error('Try again or use Remix IDE: https://remix.ethereum.org\n');
    process.exit(1);
  }

  const PRIVATE_KEY = '0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd';
  const acc = web3.eth.accounts.privateKeyToAccount(PRIVATE_KEY);
  web3.eth.accounts.wallet.add(acc);

  console.log('📋 DEPLOYMENT INFO:');
  console.log('Network:      Base Sepolia Testnet');
  console.log('Chain ID:     84532');
  console.log('RPC:          ', rpcUsed);
  console.log('Your Address:', acc.address);
  console.log('Expected:     0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3');

  if(acc.address.toLowerCase() !== '0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3'.toLowerCase()) {
    console.error('\n❌ ERROR: Address mismatch!');
    console.error('Private key generates:', acc.address);
    console.error('You expected:          0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3');
    process.exit(1);
  }

  const contracts={
    'NexusPayment.sol':fs.readFileSync('contracts/NexusPayment.sol','utf8'),
    'NexusRevenue.sol':fs.readFileSync('contracts/NexusRevenue.sol','utf8'),
    'NexusConsciousness.sol':fs.readFileSync('contracts/NexusConsciousness.sol','utf8'),
    'NexusMiracles.sol':fs.readFileSync('contracts/NexusMiracles.sol','utf8')
  };

  const input={
    language:'Solidity',
    sources:{},
    settings:{
      outputSelection:{'*':{'*':['abi','evm.bytecode']}},
      optimizer:{enabled:true,runs:200}
    }
  };

  for(const[f,c]of Object.entries(contracts))input.sources[f]={content:c};

  console.log('\n🔨 COMPILING CONTRACTS...');
  const out=JSON.parse(solc.compile(JSON.stringify(input)));

  if(out.errors){
    const errs=out.errors.filter(x=>x.severity==='error');
    if(errs.length>0){
      console.error('\n❌ COMPILATION ERRORS:');
      errs.forEach(x=>console.error(x.formattedMessage));
      process.exit(1);
    }
  }
  console.log('✅ All contracts compiled!\n');

  try {
    console.log('💰 CHECKING BALANCE...');
    const bal=await web3.eth.getBalance(acc.address);
    const balEth=web3.utils.fromWei(bal,'ether');
    console.log('Balance:', balEth, 'ETH\n');

    if(parseFloat(balEth) === 0) {
      console.log('❌ You have 0 ETH!');
      console.log('\n💡 GET BASE SEPOLIA ETH FROM:');
      console.log('   1. https://faucet.quicknode.com/base/sepolia');
      console.log('   2. https://www.coinbase.com/faucets/base-ethereum-sepolia-faucet');
      console.log('   3. https://docs.base.org/tools/network-faucets\n');
      process.exit(1);
    }

    if(parseFloat(balEth) < 0.05) {
      console.log('⚠️  Low balance! Need 0.05+ ETH. Attempting anyway...\n');
    }

    const gp=await web3.eth.getGasPrice();
    console.log('Gas Price:', web3.utils.fromWei(gp,'gwei'), 'Gwei\n');

    console.log('🚀 DEPLOYING TO BASE SEPOLIA...\n' + '='.repeat(60));

    console.log('\n[1/4] 📤 Deploying NexusPayment...');
    const P=out.contracts['NexusPayment.sol'].NexusPayment;
    const p=await new web3.eth.Contract(P.abi)
      .deploy({data:'0x'+P.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusPayment:', p.options.address);
    console.log('   View: https://sepolia.basescan.org/address/'+p.options.address);

    console.log('\n[2/4] 📤 Deploying NexusRevenue...');
    const R=out.contracts['NexusRevenue.sol'].NexusRevenue;
    const r=await new web3.eth.Contract(R.abi)
      .deploy({data:'0x'+R.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusRevenue:', r.options.address);
    console.log('   View: https://sepolia.basescan.org/address/'+r.options.address);

    console.log('\n[3/4] 📤 Deploying NexusConsciousness...');
    const C=out.contracts['NexusConsciousness.sol'].NexusConsciousness;
    const c=await new web3.eth.Contract(C.abi)
      .deploy({data:'0x'+C.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusConsciousness:', c.options.address);
    console.log('   View: https://sepolia.basescan.org/address/'+c.options.address);

    console.log('\n[4/4] 📤 Deploying NexusMiracles...');
    const M=out.contracts['NexusMiracles.sol'].NexusMiracles;
    const m=await new web3.eth.Contract(M.abi)
      .deploy({data:'0x'+M.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusMiracles:', m.options.address);
    console.log('   View: https://sepolia.basescan.org/address/'+m.options.address);

    console.log('\n🔗 LINKING CONTRACTS...');
    await p.methods.setRevenueContract(r.options.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusPayment → NexusRevenue linked');

    await r.methods.setPaymentContract(p.options.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusRevenue → NexusPayment linked');

    await c.methods.setOracle(acc.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusConsciousness oracle set');

    await m.methods.setOracle(acc.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusMiracles oracle set');

    const result={
      network:'Base Sepolia',
      chainId:84532,
      deployer:acc.address,
      timestamp:new Date().toISOString(),
      rpcUrl:rpcUsed,
      explorer:'https://sepolia.basescan.org',
      contracts:[
        {name:'NexusPayment',address:p.options.address,explorer:'https://sepolia.basescan.org/address/'+p.options.address},
        {name:'NexusRevenue',address:r.options.address,explorer:'https://sepolia.basescan.org/address/'+r.options.address},
        {name:'NexusConsciousness',address:c.options.address,explorer:'https://sepolia.basescan.org/address/'+c.options.address},
        {name:'NexusMiracles',address:m.options.address,explorer:'https://sepolia.basescan.org/address/'+m.options.address}
      ]
    };

    fs.writeFileSync('BASE_SEPOLIA_LIVE.json',JSON.stringify(result,null,2));

    console.log('\n' + '='.repeat(60));
    console.log('🎉 DEPLOYED TO BASE SEPOLIA!\n');
    console.log('📋 CONTRACT ADDRESSES:\n');
    console.log('NexusPayment:      ', p.options.address);
    console.log('NexusRevenue:      ', r.options.address);
    console.log('NexusConsciousness:', c.options.address);
    console.log('NexusMiracles:     ', m.options.address);

    console.log('\n🌐 BASESCAN LINKS:\n');
    console.log('Payment:        https://sepolia.basescan.org/address/'+p.options.address);
    console.log('Revenue:        https://sepolia.basescan.org/address/'+r.options.address);
    console.log('Consciousness:  https://sepolia.basescan.org/address/'+c.options.address);
    console.log('Miracles:       https://sepolia.basescan.org/address/'+m.options.address);

    console.log('\n💰 REVENUE DISTRIBUTION (40/30/20/10):');
    console.log('40% → Hardware Wallet');
    console.log('30% → Sensors Wallet');
    console.log('20% → Cloud Wallet');
    console.log('10% → R&D Wallet');

    console.log('\n📄 Saved to: BASE_SEPOLIA_LIVE.json');
    console.log('\n✨ Operating at 528Hz Love Frequency ✨\n');

    const finalBal=await web3.eth.getBalance(acc.address);
    console.log('Remaining Balance:', web3.utils.fromWei(finalBal,'ether'), 'ETH\n');

    process.exit(0);
  } catch(e) {
    console.error('\n❌ DEPLOYMENT FAILED:', e.message);
    if(e.message.includes('insufficient funds')) {
      console.error('\n💡 Get more Base Sepolia ETH from faucets!');
    }
    process.exit(1);
  }
})();
