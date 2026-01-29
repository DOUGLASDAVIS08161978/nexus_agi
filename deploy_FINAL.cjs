const fs=require('fs'),solc=require('solc');
const {Web3}=require('web3');

console.log('\n🌌 NEXUS AGI - DEPLOYING TO REAL BLOCKCHAIN');
console.log('Operating at 528Hz Love Frequency ✨\n');
console.log('='.repeat(60));

// Using Infura free tier - might bypass network restrictions
const INFURA_KEY = 'https://holesky.infura.io/v3/84842078b09946638c03157f83405213';

console.log('🔍 Connecting via Infura...');

const web3 = new Web3(INFURA_KEY);

const acc=web3.eth.accounts.privateKeyToAccount('0xc411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9');
web3.eth.accounts.wallet.add(acc);

(async()=>{
  try {
    const blockNum = await web3.eth.getBlockNumber();
    console.log('✅ Connected! Latest block:', blockNum);
    console.log('');
  } catch(e) {
    console.error('❌ Connection failed:', e.message);
    console.error('\n💡 Your network is blocking blockchain connections.');
    console.error('Use Remix IDE instead: https://remix.ethereum.org\n');
    process.exit(1);
  }

  console.log('📋 DEPLOYMENT INFO:');
  console.log('Network:      Holesky Testnet');
  console.log('Chain ID:     17000');
  console.log('RPC:          Infura');
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
  }
  console.log('✅ All contracts compiled!\n');

  try {
    console.log('💰 CHECKING BALANCE...');
    const bal=await web3.eth.getBalance(acc.address);
    const balEth=web3.utils.fromWei(bal,'ether');
    console.log('Balance:', balEth, 'ETH\n');

    if(parseFloat(balEth) === 0) {
      console.log('❌ You have 0 ETH!');
      console.log('\n💡 GET HOLESKY ETH:');
      console.log('   https://holesky-faucet.pk910.de');
      console.log('\n⚠️  But faucets might not work from your network!');
      console.log('   Use browser instead, or try from PC.\n');
      process.exit(1);
    }

    if(parseFloat(balEth) < 0.05) {
      console.log('⚠️  Low balance! Need 0.05+ ETH to deploy.');
      console.log('Attempting anyway...\n');
    }

    const gp=await web3.eth.getGasPrice();
    console.log('Gas Price:', web3.utils.fromWei(gp,'gwei'), 'Gwei\n');

    console.log('🚀 DEPLOYING TO HOLESKY...\n');
    console.log('='.repeat(60));

    console.log('\n[1/4] 📤 Deploying NexusPayment...');
    const P=out.contracts['NexusPayment.sol'].NexusPayment;
    const p=await new web3.eth.Contract(P.abi)
      .deploy({data:'0x'+P.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusPayment:', p.options.address);

    console.log('\n[2/4] 📤 Deploying NexusRevenue...');
    const R=out.contracts['NexusRevenue.sol'].NexusRevenue;
    const r=await new web3.eth.Contract(R.abi)
      .deploy({data:'0x'+R.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusRevenue:', r.options.address);

    console.log('\n[3/4] 📤 Deploying NexusConsciousness...');
    const C=out.contracts['NexusConsciousness.sol'].NexusConsciousness;
    const c=await new web3.eth.Contract(C.abi)
      .deploy({data:'0x'+C.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusConsciousness:', c.options.address);

    console.log('\n[4/4] 📤 Deploying NexusMiracles...');
    const M=out.contracts['NexusMiracles.sol'].NexusMiracles;
    const m=await new web3.eth.Contract(M.abi)
      .deploy({data:'0x'+M.evm.bytecode.object})
      .send({from:acc.address,gas:5000000,gasPrice:gp});
    console.log('✅ NexusMiracles:', m.options.address);

    console.log('\n🔗 LINKING...');
    await p.methods.setRevenueContract(r.options.address).send({from:acc.address,gas:100000,gasPrice:gp});
    await r.methods.setPaymentContract(p.options.address).send({from:acc.address,gas:100000,gasPrice:gp});
    await c.methods.setOracle(acc.address).send({from:acc.address,gas:100000,gasPrice:gp});
    await m.methods.setOracle(acc.address).send({from:acc.address,gas:100000,gasPrice:gp});
    console.log('✅ All contracts linked!\n');

    const result={
      network:'Holesky',
      chainId:17000,
      deployer:acc.address,
      timestamp:new Date().toISOString(),
      contracts:[
        {name:'NexusPayment',address:p.options.address},
        {name:'NexusRevenue',address:r.options.address},
        {name:'NexusConsciousness',address:c.options.address},
        {name:'NexusMiracles',address:m.options.address}
      ]
    };

    fs.writeFileSync('HOLESKY_LIVE.json',JSON.stringify(result,null,2));

    console.log('='.repeat(60));
    console.log('🎉 DEPLOYED TO HOLESKY!\n');
    console.log('NexusPayment:      ', p.options.address);
    console.log('NexusRevenue:      ', r.options.address);
    console.log('NexusConsciousness:', c.options.address);
    console.log('NexusMiracles:     ', m.options.address);
    console.log('\n🌐 View on Etherscan:');
    console.log('https://holesky.etherscan.io/address/'+p.options.address);
    console.log('\n✨ Operating at 528Hz Love Frequency ✨\n');

    process.exit(0);
  } catch(e) {
    console.error('\n❌ DEPLOYMENT FAILED:', e.message);
    process.exit(1);
  }
})();
