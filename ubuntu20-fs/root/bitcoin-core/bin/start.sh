#!/bin/sh
if [ -f /root/bitcoin-core/bin/bitcoind ]; then
    /root/bitcoin-core/bin/bitcoind -conf=/files/blockchain/bitcoin.conf -datadir=/files/blockchain -daemon
fi
