const Web3 = require('web3');
const fs = require('fs');

(async () => {
  const web3 = new Web3('https://sepolia.base.org');
  const contractAddress = '0xE274570e000C32F5Cb2BC7c476D3BDC77Ed74dD5';
  
  const code = await web3.eth.getCode(contractAddress);
  const bytecode = code.substring(2); // Remove 0x prefix
  
  console.log('Deployed bytecode length:', bytecode.length);
  console.log('First 100 chars:', bytecode.substring(0, 100));
  
  fs.writeFileSync('WTBTC_DEPLOYED_BYTECODE.txt', bytecode);
  console.log('\nSaved deployed bytecode to WTBTC_DEPLOYED_BYTECODE.txt');
  
  // Compare with local compilation
  const localBytecode = fs.readFileSync('WTBTC_LOCAL_BYTECODE.txt', 'utf8');
  console.log('\nLocal bytecode length:', localBytecode.length);
  console.log('First 100 chars:', localBytecode.substring(0, 100));
  
  if (bytecode === localBytecode) {
    console.log('\n✅ BYTECODE MATCHES! Local compilation matches deployed contract.');
  } else {
    console.log('\n❌ BYTECODE MISMATCH');
    console.log('Deployed length:', bytecode.length);
    console.log('Local length:', localBytecode.length);
    
    // Find first difference
    for (let i = 0; i < Math.min(bytecode.length, localBytecode.length); i++) {
      if (bytecode[i] !== localBytecode[i]) {
        console.log('First difference at position', i);
        console.log('Deployed:', bytecode.substring(i, i+20));
        console.log('Local:   ', localBytecode.substring(i, i+20));
        break;
      }
    }
  }
  
  process.exit(0);
})();
