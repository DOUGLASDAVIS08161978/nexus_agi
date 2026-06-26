#!/bin/bash
#
# Deploy WTBTC to Base Mainnet using CreateX for deterministic address
# Same address across all chains where CreateX is deployed
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
PRIVATE_KEY="0x0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
WALLET="0x9FE74D9D6f1Ae0Ce1fb3B51d4a82c05b74e280f3"
BASE_RPC="https://mainnet.base.org"
CHAIN_ID=8453
CREATEX_ADDRESS="0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed"

# WTBTC bytecode
WTBTC_BYTECODE="0x608060405234801561001057600080fd5b506040518060400160405280601681526020017f5772617070656420546573746e6574204254432000000000000000000000000081525060405180604001604052806005815260200164575442544360d81b8152508160039081610075919061047d565b50600461008282826104d2565b5050506100a361009760201b60201c565b6005546001600160a01b0316565b6100b96100ae60201b60201c565b6100c160201b60201c565b610699565b600033905090565b6000600560009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905081600560006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055508173ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff167f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e060405160405180910390a35050565b600081519050919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806102185750607f821691505b60208210810361022b5761022a6101d1565b5b50919050565b60008190508160005260206000209050919050565b60006020601f8301049050919050565b600082821b905092915050565b6000600883026102937fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff82610256565b61029d8683610256565b95508019841693508086168417925050509392505050565b6000819050919050565b6000819050919050565b60006102e46102df6102da846102b5565b6102bf565b6102b5565b9050919050565b6000819050919050565b6102fe836102c9565b610312610304826102eb565b848454610263565b825550505050565b600090565b61032761031a565b6103328184846102f5565b505050565b5b818110156103565761034b60008261031f565b600181019050610338565b5050565b601f82111561039b5761036c81610231565b61037584610246565b81016020851015610384578190505b61039861039085610246565b830182610337565b50505b505050565b600082821c905092915050565b60006103be600019846008026103a0565b1980831691505092915050565b60006103d783836103ad565b9150826002028217905092915050565b6103f082610194565b67ffffffffffffffff8111156104095761040861019f565b5b6104138254610200565b61041e82828561035a565b600060209050601f831160018114610451576000841561043f578287015190505b61044985826103cb565b8655506104b1565b601f19841661045f86610231565b60005b8281101561048757848901518255600182019150602085019450602081019050610462565b868310156104a457848901516104a0601f8916826103ad565b8355505b6001600288020188555050505b505050505050565b610de9806104c86000396000f3fe608060405234801561001057600080fd5b50600436106100a95760003560e01c8063715018a611610071578063715018a6146101425780638da5cb5b1461014c57806395d89b411461016a578063a9059cbb14610188578063dd62ed3e146101b8576100a9565b806306fdde03146100ae578063095ea7b3146100cc57806318160ddd146100fc57806370a082311461011a578063715018a614610140575b600080fd5b6100b66101e8565b6040516100c39190610af3565b60405180910390f35b6100e660048036038101906100e19190610bae565b61027a565b6040516100f39190610c09565b60405180910390f35b61010461029d565b6040516101119190610c33565b60405180910390f35b610134600480360381019061012f9190610c4e565b6102a7565b6040516101419190610c33565b60405180910390f35b61014a6102ef565b005b610154610303565b6040516101619190610c8a565b60405180910390f35b61017261032d565b60405161017f9190610af3565b60405180910390f35b6101a2600480360381019061019d9190610bae565b6103bf565b6040516101af9190610c09565b60405180910390f35b6101d260048036038101906101cd9190610ca5565b6103e2565b6040516101df9190610c33565b60405180910390f35b6060600380546101f790610d14565b80601f016020809104026020016040519081016040528092919081815260200182805461022390610d14565b80156102705780601f1061024557610100808354040283529160200191610270565b820191906000526020600020905b81548152906001019060200180831161025357829003601f168201915b5050505050905090565b600080610285610469565b9050610292818585610471565b600191505092915050565b6000600254905090565b60008060008373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020549050919050565b6102f761063a565b61030160006106b8565b565b6000600560009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b60606004805461033c90610d14565b80601f016020809104026020016040519081016040528092919081815260200182805461036890610d14565b80156103b55780601f1061038a576101008083540402835291602001916103b5565b820191906000526020600020905b81548152906001019060200180831161039857829003601f168201915b5050505050905090565b6000806103ca610469565b90506103d781858561077e565b600191505092915050565b6000600160008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002054905092915050565b600033905090565b600073ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff16036104e0576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016104d790610db7565b60405180910390fd5b600073ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff160361054f576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161054690610e49565b60405180910390fd5b80600160008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055508173ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff167f8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b9258360405161062d9190610c33565b60405180910390a3505050565b610642610469565b73ffffffffffffffffffffffffffffffffffffffff16610660610303565b73ffffffffffffffffffffffffffffffffffffffff16146106b6576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016106ad90610eb5565b60405180910390fd5b565b6000600560009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905081600560006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055508173ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff167f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e060405160405180910390a35050565b505050565b600081519050919050565b600082825260208201905092915050565b60005b838110156107bd5780820151818401526020810190506107a2565b60008484015250505050565b6000601f19601f8301169050919050565b60006107e582610783565b6107ef818561078e565b93506107ff81856020860161079f565b610808816107c9565b840191505092915050565b6000602082019050818103600083015261082d81846107da565b905092915050565b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b60006108658261083a565b9050919050565b6108758161085a565b811461088057600080fd5b50565b6000813590506108928161086c565b92915050565b6000819050919050565b6108ab81610898565b81146108b657600080fd5b50565b6000813590506108c8816108a2565b92915050565b600080604083850312156108e5576108e4610835565b5b60006108f385828601610883565b9250506020610904858286016108b9565b9150509250929050565b60008115159050919050565b6109238161090e565b82525050565b600060208201905061093e600083018461091a565b92915050565b61094d81610898565b82525050565b60006020820190506109686000830184610944565b92915050565b60006020828403121561098457610983610835565b5b600061099284828501610883565b91505092915050565b6109a48161085a565b82525050565b60006020820190506109bf600083018461099b565b92915050565b600080604083850312156109dc576109db610835565b5b60006109ea85828601610883565b92505060206109fb85828601610883565b9150509250929050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b60006002820490506001821680610a4c57607f821691505b602082108103610a5f57610a5e610a05565b5b50919050565b7f45524332303a20617070726f76652066726f6d20746865207a65726f2061646460008201527f7265737300000000000000000000000000000000000000000000000000000000602082015250565b6000610ac160248361078e565b9150610acc82610a65565b604082019050919050565b60006020820190508181036000830152610af081610ab4565b9050919050565b7f45524332303a20617070726f766520746f20746865207a65726f20616464726560008201527f7373000000000000000000000000000000000000000000000000000000000000602082015250565b6000610b5360228361078e565b9150610b5e82610af7565b604082019050919050565b60006020820190508181036000830152610b8281610b46565b9050919050565b7f4f776e61626c653a2063616c6c6572206973206e6f7420746865206f776e6572600082015250565b6000610bbf60208361078e565b9150610bca82610b89565b602082019050919050565b60006020820190508181036000830152610bee81610bb2565b905091905056fea26469706673582212201234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef64736f6c63430008170033"

echo ""
echo -e "${BOLD}================================================================================${NC}"
echo -e "${BOLD}║  🚀 BASE MAINNET - CREATEX DETERMINISTIC DEPLOYMENT                         ║${NC}"
echo -e "${BOLD}================================================================================${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: MAINNET DEPLOYMENT WITH REAL ETH! ⚠️${NC}"
echo ""
read -p "Deploy WTBTC to Base Mainnet via CreateX? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${YELLOW}Deployment cancelled.${NC}"
    exit 0
fi

# Function to make JSON-RPC calls
rpc_call() {
    local method="$1"
    local params="$2"
    curl -s -X POST "$BASE_RPC" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":$params,\"id\":1}" \
        --connect-timeout 30 \
        --max-time 60
}

# Verify CreateX deployment
echo -e "${CYAN}🔍 Verifying CreateX on Base Mainnet...${NC}"
CREATEX_CODE=$(rpc_call "eth_getCode" "[\"$CREATEX_ADDRESS\",\"latest\"]")

if [[ "$CREATEX_CODE" == *'"result":"0x"'* ]] || [[ "$CREATEX_CODE" == *'"result":"0x0"'* ]]; then
    echo -e "${RED}❌ CreateX not deployed at $CREATEX_ADDRESS on Base${NC}"
    echo -e "${YELLOW}   Using direct deployment instead...${NC}"
    exec ./deploy_base_mainnet.sh
fi

echo -e "${GREEN}✅ CreateX found at $CREATEX_ADDRESS${NC}"

# Check connection
echo -e "${CYAN}📡 Connecting to Base Mainnet...${NC}"
CHAIN_ID_RESULT=$(rpc_call "eth_chainId" "[]")
CHAIN_ID_HEX=$(echo "$CHAIN_ID_RESULT" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)

if [ -z "$CHAIN_ID_HEX" ]; then
    echo -e "${RED}❌ Failed to connect${NC}"
    exit 1
fi

CHAIN_ID_DEC=$(printf "%d" "$CHAIN_ID_HEX")

if [ "$CHAIN_ID_DEC" != "8453" ]; then
    echo -e "${RED}❌ Wrong chain! Expected 8453, got $CHAIN_ID_DEC${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Connected to Base Mainnet (Chain ID: $CHAIN_ID_DEC)${NC}"

# Check balance
echo -e "${CYAN}💼 Deployer: $WALLET${NC}"

BALANCE_RESULT=$(rpc_call "eth_getBalance" "[\"$WALLET\",\"latest\"]")
BALANCE_HEX=$(echo "$BALANCE_RESULT" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)

if [ -z "$BALANCE_HEX" ] || [ "$BALANCE_HEX" = "0x0" ] || [ "$BALANCE_HEX" = "0x" ]; then
    echo -e "${RED}❌ No ETH for gas!${NC}"
    exit 1
fi

BALANCE_WEI=$(printf "%d" "$BALANCE_HEX")
BALANCE_ETH=$(awk "BEGIN {printf \"%.6f\", $BALANCE_WEI / 1000000000000000000}")
echo -e "${GREEN}💰 Balance: $BALANCE_ETH ETH${NC}"

# Get gas parameters
NONCE_RESULT=$(rpc_call "eth_getTransactionCount" "[\"$WALLET\",\"latest\"]")
NONCE_HEX=$(echo "$NONCE_RESULT" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)

GAS_PRICE_RESULT=$(rpc_call "eth_gasPrice" "[]")
GAS_PRICE_HEX=$(echo "$GAS_PRICE_RESULT" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}║  DEPLOYING VIA CREATEX                                        ║${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Create deployment script
cat > /tmp/deploy_base_createx.js << 'EOFJS'
const { ethers } = require('ethers');

async function main() {
    const privateKey = process.argv[2];
    const nonce = process.argv[3];
    const gasPrice = process.argv[4];
    const chainId = parseInt(process.argv[5]);
    const bytecode = process.argv[6];

    const wallet = new ethers.Wallet(privateKey);

    // Generate salt: <20 bytes deployer><1 byte cross-chain flag><11 bytes random>
    const deployerBytes = wallet.address.slice(2).toLowerCase();
    const crossChainByte = "01"; // Enable cross-chain protection
    const randomBytes = "00000000000000000000";
    const salt = "0x" + deployerBytes + crossChainByte + randomBytes;

    // CreateX interface
    const createxInterface = new ethers.Interface([
        "function deployCreate2(bytes32 salt, bytes memory initCode) payable returns (address)"
    ]);

    const calldata = createxInterface.encodeFunctionData("deployCreate2", [salt, bytecode]);

    const tx = {
        to: "0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed",
        data: calldata,
        nonce: parseInt(nonce, 16),
        gasLimit: 3000000,
        gasPrice: gasPrice,
        chainId: chainId,
        value: 0
    };

    const signedTx = await wallet.signTransaction(tx);

    // Calculate expected address
    const abiCoder = ethers.AbiCoder.defaultAbiCoder();
    const guardedSalt = ethers.keccak256(
        abiCoder.encode(['address', 'uint256', 'bytes32'], [wallet.address, chainId, salt])
    );

    const initCodeHash = ethers.keccak256(bytecode);
    const expectedAddress = ethers.getCreate2Address(
        "0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed",
        guardedSalt,
        initCodeHash
    );

    console.log("SALT:" + salt);
    console.log("EXPECTED:" + expectedAddress);
    console.log("SIGNED:" + signedTx);
}

main().catch(err => {
    console.error("ERROR:" + err.message);
    process.exit(1);
});
EOFJS

echo -e "${CYAN}🔐 Computing deterministic address...${NC}"

DEPLOY_OUTPUT=$(node /tmp/deploy_base_createx.js "$PRIVATE_KEY" "$NONCE_HEX" "$GAS_PRICE_HEX" "$CHAIN_ID" "$WTBTC_BYTECODE" 2>&1)

if [[ "$DEPLOY_OUTPUT" == *"ERROR:"* ]]; then
    echo -e "${RED}❌ Failed to prepare transaction${NC}"
    echo "$DEPLOY_OUTPUT"
    exit 1
fi

SALT=$(echo "$DEPLOY_OUTPUT" | grep "^SALT:" | cut -d':' -f2)
EXPECTED_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep "^EXPECTED:" | cut -d':' -f2)
SIGNED_TX=$(echo "$DEPLOY_OUTPUT" | grep "^SIGNED:" | cut -d':' -f2)

if [ -z "$SIGNED_TX" ] || [ -z "$EXPECTED_ADDRESS" ]; then
    echo -e "${RED}❌ Failed to build transaction${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Transaction prepared${NC}"
echo -e "${CYAN}   Salt: $SALT${NC}"
echo -e "${YELLOW}   Deterministic Address: $EXPECTED_ADDRESS${NC}"
echo ""

echo -e "${CYAN}📤 Broadcasting to Base Mainnet...${NC}"

SEND_RESULT=$(rpc_call "eth_sendRawTransaction" "[\"$SIGNED_TX\"]")
TX_HASH=$(echo "$SEND_RESULT" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)

if [ -z "$TX_HASH" ] || [[ "$TX_HASH" == *"error"* ]]; then
    echo -e "${RED}❌ Transaction failed${NC}"
    echo "$SEND_RESULT"
    exit 1
fi

echo -e "${YELLOW}   TX Hash: $TX_HASH${NC}"
echo ""
echo -e "${CYAN}⏳ Waiting for confirmation...${NC}"

TIMEOUT=40
COUNTER=0

while [ $COUNTER -lt $TIMEOUT ]; do
    sleep 3
    COUNTER=$((COUNTER + 1))

    RECEIPT_RESULT=$(rpc_call "eth_getTransactionReceipt" "[\"$TX_HASH\"]")

    if [[ "$RECEIPT_RESULT" != *"\"result\":null"* ]]; then
        STATUS=$(echo "$RECEIPT_RESULT" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

        if [ "$STATUS" = "0x1" ]; then
            echo ""
            echo -e "${BOLD}================================================================================${NC}"
            echo -e "${GREEN}║  ✅ CREATEX DEPLOYMENT SUCCESSFUL ON BASE MAINNET!                            ║${NC}"
            echo -e "${BOLD}================================================================================${NC}"
            echo ""
            echo -e "${GREEN}📍 WTBTC Contract: $EXPECTED_ADDRESS${NC}"
            echo -e "${GREEN}🏭 Deployed via: CreateX CREATE2${NC}"
            echo -e "${GREEN}🔗 TX Hash: $TX_HASH${NC}"
            echo -e "${GREEN}🔐 Salt: $SALT${NC}"
            echo ""
            echo -e "${CYAN}🌐 BaseScan:${NC}"
            echo -e "${CYAN}   https://basescan.org/address/$EXPECTED_ADDRESS${NC}"
            echo -e "${CYAN}   https://basescan.org/tx/$TX_HASH${NC}"
            echo ""
            echo -e "${BOLD}✨ WTBTC with deterministic addressing! ✨${NC}"
            echo -e "${YELLOW}   Use the same salt to deploy to other chains at the same address.${NC}"
            echo ""

            cat > wtbtc_base_mainnet_createx.json << EOF
{
  "network": "base-mainnet",
  "chainId": $CHAIN_ID,
  "deploymentMethod": "CreateX CREATE2",
  "createxAddress": "$CREATEX_ADDRESS",
  "salt": "$SALT",
  "contracts": {
    "WTBTC": {
      "address": "$EXPECTED_ADDRESS",
      "deploymentTx": "$TX_HASH"
    }
  },
  "explorer": "https://basescan.org/address/$EXPECTED_ADDRESS",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
            echo -e "${CYAN}💾 Saved: wtbtc_base_mainnet_createx.json${NC}"
            echo ""
            exit 0
        else
            echo -e "${RED}❌ Transaction reverted${NC}"
            exit 1
        fi
    fi

    if [ $((COUNTER % 5)) -eq 0 ]; then
        echo -e "${CYAN}   Waiting... ($((COUNTER * 3))s)${NC}"
    fi
done

echo -e "${RED}❌ Timeout${NC}"
echo -e "${YELLOW}Check: https://basescan.org/tx/$TX_HASH${NC}"
exit 1
