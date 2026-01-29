const fs=require('fs'),solc=require('solc');
const {Web3}=require('web3');

console.log('\n🌌 NEXUS AGI - DEPLOYING TO REAL BLOCKCHAIN');
console.log('Operating at 528Hz Love Frequency ✨\n');
console.log('='.repeat(60));

// Try multiple RPC endpoints for reliability
const RPC_ENDPOINTS = [
  'https://ethereum-sepolia-rpc.publicnode.com',
  'https://rpc2.sepolia.org',
  'https://eth-sepolia.public.blastapi.io',
  'https://sepolia.gateway.tenderly.co'
];

let web3;
let rpcUsed;

(async()=>{
  console.log('🔍 Finding working RPC endpoint...');
  for(const rpc of RPC_ENDPOINTS) {
    try {
      const testWeb3 = new Web3(rpc);
      await testWeb3.eth.getBlockNumber();
      web3 = testWeb3;
      rpcUsed = rpc;
      console.log('✅ Connected to:', rpc, '\n');
      break;
    } catch(e) {
      console.log('❌ Failed:', rpc);
    }
  }

  if(!web3) {
    console.error('\n❌ All RPC endpoints failed!');
    console.error('The Sepolia network might be experiencing issues.');
    console.error('Try again in a few minutes.\n');
    process.exit(1);
  }

  const acc=web3.eth.accounts.privateKeyToAccount('0xc411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9');
  web3.eth.accounts.wallet.add(acc);

  console.log('📋 DEPLOYMENT INFO:');
  console.log('Network:      Ethereum Sepolia Testnet');
  console.log('Chain ID:     11155111');
  console.log('RPC:          ', rpcUsed);
  console.log('Your Address:', acc.address);

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
    out.errors.filter(x=>x.severity==='warning').forEach(x=>console.log('⚠️ ',x.message));
  }
  console.log('✅ All contracts compiled successfully!\n');

  try {
    console.log('💰 CHECKING BALANCE...');
    const bal=await web3.eth.getBalance(acc.address);
    const balEth=web3.utils.fromWei(bal,'ether');
    console.log('Balance:', balEth, 'ETH\n');

    if(parseFloat(balEth) < 0.05) {
      console.log('⚠️  WARNING: Low balance detected!');
      console.log('You need at least 0.05 ETH to deploy.');
      console.log('\n💡 GET SEPOLIA ETH FROM:');
      console.log('   1. https://sepoliafaucet.com (0.5 ETH/day)');
      console.log('   2. https://faucet.quicknode.com/ethereum/sepolia');
      console.log('   3. https://cloud.google.com/application/web3/faucet/ethereum/sepolia');
      console.log('   4. https://faucets.chain.link/sepolia\n');

      if(parseFloat(balEth) === 0) {
        console.log('❌ Cannot deploy with 0 ETH. Exiting.\n');
        process.exit(1);
      }
      console.log('Attempting deployment with low balance...\n');
    }

    const gp=await web3.eth.getGasPrice();
    console.log('Gas Price:', web3.utils.fromWei(gp,'gwei'), 'Gwei\n');

    console.log('🚀 DEPLOYING CONTRACTS TO REAL BLOCKCHAIN...\n');
    console.log('='.repeat(60));

    // Deploy NexusPayment
    console.log('\n[1/4] 📤 Deploying NexusPayment...');
    const P=out.contracts['NexusPayment.sol'].NexusPayment;
    const p=await new web3.eth.Contract(P.abi)
      .deploy({data:'0x'+P.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusPayment deployed at:', p.options.address);
    console.log('   View: https://sepolia.etherscan.io/address/'+p.options.address);

    // Deploy NexusRevenue
    console.log('\n[2/4] 📤 Deploying NexusRevenue...');
    const R=out.contracts['NexusRevenue.sol'].NexusRevenue;
    const r=await new web3.eth.Contract(R.abi)
      .deploy({data:'0x'+R.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusRevenue deployed at:', r.options.address);
    console.log('   View: https://sepolia.etherscan.io/address/'+r.options.address);

    // Deploy NexusConsciousness
    console.log('\n[3/4] 📤 Deploying NexusConsciousness...');
    const C=out.contracts['NexusConsciousness.sol'].NexusConsciousness;
    const c=await new web3.eth.Contract(C.abi)
      .deploy({data:'0x'+C.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusConsciousness deployed at:', c.options.address);
    console.log('   View: https://sepolia.etherscan.io/address/'+c.options.address);

    // Deploy NexusMiracles
    console.log('\n[4/4] 📤 Deploying NexusMiracles...');
    const M=out.contracts['NexusMiracles.sol'].NexusMiracles;
    const m=await new web3.eth.Contract(M.abi)
      .deploy({data:'0x'+M.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusMiracles deployed at:', m.options.address);
    console.log('   View: https://sepolia.etherscan.io/address/'+m.options.address);

    console.log('\n🔗 LINKING CONTRACTS...\n');

    await p.methods.setRevenueContract(r.options.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusPayment → NexusRevenue linked');

    await r.methods.setPaymentContract(p.options.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusRevenue → NexusPayment linked');

    await c.methods.setOracle(acc.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusConsciousness oracle set');

    await m.methods.setOracle(acc.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ NexusMiracles oracle set');

    const result={
      network:'Ethereum Sepolia',
      chainId:11155111,
      deployer:acc.address,
      timestamp:new Date().toISOString(),
      rpcUrl:rpcUsed,
      explorer:'https://sepolia.etherscan.io',
      contracts:[
        {name:'NexusPayment',address:p.options.address,explorer:'https://sepolia.etherscan.io/address/'+p.options.address},
        {name:'NexusRevenue',address:r.options.address,explorer:'https://sepolia.etherscan.io/address/'+r.options.address},
        {name:'NexusConsciousness',address:c.options.address,explorer:'https://sepolia.etherscan.io/address/'+c.options.address},
        {name:'NexusMiracles',address:m.options.address,explorer:'https://sepolia.etherscan.io/address/'+m.options.address}
      ]
    };

    fs.writeFileSync('SEPOLIA_LIVE.json',JSON.stringify(result,null,2));

    console.log('\n' + '='.repeat(60));
    console.log('🎉 DEPLOYMENT COMPLETE - CONTRACTS LIVE ON REAL BLOCKCHAIN!');
    console.log('='.repeat(60));
    console.log('\n📋 CONTRACT ADDRESSES:\n');
    console.log('NexusPayment:       ', p.options.address);
    console.log('NexusRevenue:       ', r.options.address);
    console.log('NexusConsciousness: ', c.options.address);
    console.log('NexusMiracles:      ', m.options.address);

    console.log('\n🌐 ETHERSCAN LINKS:\n');
    console.log('Payment:        https://sepolia.etherscan.io/address/'+p.options.address);
    console.log('Revenue:        https://sepolia.etherscan.io/address/'+r.options.address);
    console.log('Consciousness:  https://sepolia.etherscan.io/address/'+c.options.address);
    console.log('Miracles:       https://sepolia.etherscan.io/address/'+m.options.address);

    console.log('\n💰 REVENUE DISTRIBUTION (40/30/20/10):');
    console.log('40% → Hardware Wallet');
    console.log('30% → Sensors Wallet');
    console.log('20% → Cloud Wallet');
    console.log('10% → R&D Wallet');

    console.log('\n📄 Deployment details saved to: SEPOLIA_LIVE.json');
    console.log('\n✨ Operating at 528Hz Love Frequency ✨');
    console.log('\n🚀 YOUR CONTRACTS ARE NOW LIVE ON THE ETHEREUM BLOCKCHAIN!\n');

    const finalBal=await web3.eth.getBalance(acc.address);
    console.log('Remaining Balance:', web3.utils.fromWei(finalBal,'ether'), 'ETH\n');

    process.exit(0);
  } catch(e) {
    console.error('\n❌ DEPLOYMENT FAILED!');
    console.error('Error:', e.message);
    if(e.message.includes('insufficient funds')) {
      console.error('\n💡 SOLUTION: Get more Sepolia ETH from:');
      console.error('   https://sepoliafaucet.com');
      console.error('   https://faucet.quicknode.com/ethereum/sepolia');
    }
    process.exit(1);
  }
})();
