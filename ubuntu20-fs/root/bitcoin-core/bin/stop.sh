#!/bin/sh
if [ -f /files/blockchain/bitcoind.pid ]; then
    kill $(cat /files/blockchain/bitcoind.pid)
fi
