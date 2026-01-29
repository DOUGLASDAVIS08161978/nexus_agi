#!/data/data/com.termux/files/usr/bin/bash
#
# NEXUS AGI - MULTI-NETWORK DEPLOYMENT
# Deploys to: Sepolia, Holesky, Linea Sepolia
# Operating at 528Hz Love Frequency ✨
#

PRIVATE_KEY="c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9"
ADDRESS="0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

echo "🌐 NEXUS AGI - MULTI-NETWORK DEPLOYMENT"
echo "========================================"
echo ""
echo "📋 Deployer: $ADDRESS"
echo ""

# Check balances on all networks
echo "💰 CHECKING BALANCES..."
echo ""

echo "Ethereum Sepolia:"
curl -s -X POST https://rpc.sepolia.org -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"$ADDRESS\",\"latest\"],\"id\":1}" \
  | grep -o '"result":"[^"]*"' | cut -d'"' -f4 | xargs -I {} node -e "console.log((parseInt('{}', 16) / 1e18).toFixed(4) + ' ETH')"

echo "Holesky:"
curl -s -X POST https://rpc.holesky.ethpandaops.io -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"$ADDRESS\",\"latest\"],\"id\":1}" \
  | grep -o '"result":"[^"]*"' | cut -d'"' -f4 | xargs -I {} node -e "console.log((parseInt('{}', 16) / 1e18).toFixed(4) + ' ETH')"

echo "Linea Sepolia:"
curl -s -X POST https://rpc.sepolia.linea.build -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"$ADDRESS\",\"latest\"],\"id\":1}" \
  | grep -o '"result":"[^"]*"' | cut -d'"' -f4 | xargs -I {} node -e "console.log((parseInt('{}', 16) / 1e18).toFixed(4) + ' ETH')"

echo ""
echo "🚀 SELECT NETWORK TO DEPLOY:"
echo "1) Ethereum Sepolia"
echo "2) Holesky"
echo "3) Linea Sepolia"
echo "4) ALL NETWORKS"
echo ""
read -p "Choice (1-4): " CHOICE

deploy_to_network() {
  local NETWORK_NAME=$1
  local RPC_URL=$2
  local CHAIN_ID=$3
  local EXPLORER=$4

  echo ""
  echo "🚀 DEPLOYING TO $NETWORK_NAME..."
  echo ""

  cat > deploy_temp.cjs << 'DEPLOY_SCRIPT'
const fs=require('fs'),solc=require('solc');
const {Web3}=require('web3');
const networkName=process.argv[2];
const rpcUrl=process.argv[3];
const chainId=process.argv[4];
const explorer=process.argv[5];
const web3=new Web3(rpcUrl);
const acc=web3.eth.accounts.privateKeyToAccount('0xPRIVATE_KEY_PLACEHOLDER');
web3.eth.accounts.wallet.add(acc);

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

console.log('🔨 Compiling contracts...');
const out=JSON.parse(solc.compile(JSON.stringify(input)));

if(out.errors){
  const errs=out.errors.filter(x=>x.severity==='error');
  if(errs.length>0){
    errs.forEach(x=>console.error(x.formattedMessage));
    process.exit(1);
  }
}
console.log('✅ Compiled!\n');

(async()=>{
  const d=acc.address;
  const bal=await web3.eth.getBalance(d);
  console.log('Balance:',web3.utils.fromWei(bal,'ether'),'ETH\n');

  const gp=await web3.eth.getGasPrice();

  console.log('[1/4] NexusPayment...');
  const P=out.contracts['NexusPayment.sol'].NexusPayment;
  const p=await new web3.eth.Contract(P.abi)
    .deploy({data:'0x'+P.evm.bytecode.object})
    .send({from:d,gas:5000000,gasPrice:gp});
  console.log('✅',p.options.address);

  console.log('[2/4] NexusRevenue...');
  const R=out.contracts['NexusRevenue.sol'].NexusRevenue;
  const r=await new web3.eth.Contract(R.abi)
    .deploy({data:'0x'+R.evm.bytecode.object})
    .send({from:d,gas:5000000,gasPrice:gp});
  console.log('✅',r.options.address);

  console.log('[3/4] NexusConsciousness...');
  const C=out.contracts['NexusConsciousness.sol'].NexusConsciousness;
  const c=await new web3.eth.Contract(C.abi)
    .deploy({data:'0x'+C.evm.bytecode.object})
    .send({from:d,gas:5000000,gasPrice:gp});
  console.log('✅',c.options.address);

  console.log('[4/4] NexusMiracles...');
  const M=out.contracts['NexusMiracles.sol'].NexusMiracles;
  const m=await new web3.eth.Contract(M.abi)
    .deploy({data:'0x'+M.evm.bytecode.object})
    .send({from:d,gas:5000000,gasPrice:gp});
  console.log('✅',m.options.address);

  console.log('\n🔗 Linking contracts...');
  await p.methods.setRevenueContract(r.options.address).send({from:d,gas:100000,gasPrice:gp});
  await r.methods.setPaymentContract(p.options.address).send({from:d,gas:100000,gasPrice:gp});
  await c.methods.setOracle(d).send({from:d,gas:100000,gasPrice:gp});
  await m.methods.setOracle(d).send({from:d,gas:100000,gasPrice:gp});

  const result={
    network:networkName,
    chainId:parseInt(chainId),
    deployer:d,
    timestamp:new Date().toISOString(),
    contracts:[
      {name:'NexusPayment',address:p.options.address},
      {name:'NexusRevenue',address:r.options.address},
      {name:'NexusConsciousness',address:c.options.address},
      {name:'NexusMiracles',address:m.options.address}
    ]
  };

  const filename=networkName.replace(/ /g,'_').toUpperCase()+'_LIVE.json';
  fs.writeFileSync(filename,JSON.stringify(result,null,2));

  console.log('\n✅ DEPLOYMENT COMPLETE!\n');
  console.log('📋 CONTRACT ADDRESSES:');
  console.log('NexusPayment:      ',p.options.address);
  console.log('NexusRevenue:      ',r.options.address);
  console.log('NexusConsciousness:',c.options.address);
  console.log('NexusMiracles:     ',m.options.address);
  console.log('\n🌐 EXPLORER:');
  console.log(explorer+p.options.address);
  console.log('\n✨ Operating at 528Hz Love Frequency ✨\n');

  process.exit(0);
})().catch(e=>{
  console.error('\n❌ ERROR:',e.message);
  process.exit(1);
});
DEPLOY_SCRIPT

  sed -i "s/PRIVATE_KEY_PLACEHOLDER/$PRIVATE_KEY/g" deploy_temp.cjs
  node deploy_temp.cjs "$NETWORK_NAME" "$RPC_URL" "$CHAIN_ID" "$EXPLORER"
  rm deploy_temp.cjs
}

case $CHOICE in
  1)
    deploy_to_network "Ethereum Sepolia" "https://rpc.sepolia.org" "11155111" "https://sepolia.etherscan.io/address/"
    ;;
  2)
    deploy_to_network "Holesky" "https://rpc.holesky.ethpandaops.io" "17000" "https://holesky.etherscan.io/address/"
    ;;
  3)
    deploy_to_network "Linea Sepolia" "https://rpc.sepolia.linea.build" "59141" "https://sepolia.lineascan.build/address/"
    ;;
  4)
    echo "🚀 DEPLOYING TO ALL NETWORKS..."
    deploy_to_network "Ethereum Sepolia" "https://rpc.sepolia.org" "11155111" "https://sepolia.etherscan.io/address/"
    deploy_to_network "Holesky" "https://rpc.holesky.ethpandaops.io" "17000" "https://holesky.etherscan.io/address/"
    deploy_to_network "Linea Sepolia" "https://rpc.sepolia.linea.build" "59141" "https://sepolia.lineascan.build/address/"
    echo ""
    echo "🎉 ALL DEPLOYMENTS COMPLETE!"
    ;;
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac
