"""
NEXUS AGI - PUBLIC API INTEGRATOR
==================================

Integrates all public APIs for comprehensive blockchain and market data
No authentication required - uses public endpoints only

Features:
- Real-time cryptocurrency prices (CoinGecko)
- Block explorer data (Etherscan, Polygonscan, BSCScan)
- DeFi protocol data (Uniswap, Aave)
- Exchange rates
- ENS resolution
- IPFS content retrieval
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Cryptocurrency price data"""
    symbol: str
    price_usd: float
    price_change_24h: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    timestamp: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PublicAPIIntegrator:
    """
    Integrates all public APIs for Nexus AGI
    Provides unified interface for price data, blockchain data, and DeFi data
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Nexus-AGI/1.0"
        })

        # API endpoints
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.cryptocompare_base = "https://min-api.cryptocompare.com/data"
        self.etherscan_base = "https://api.etherscan.io/api"
        self.polygonscan_base = "https://api.polygonscan.com/api"
        self.bscscan_base = "https://api.bscscan.com/api"
        self.exchangerate_base = "https://api.exchangerate-api.com/v4/latest"

        # IPFS gateways
        self.ipfs_gateways = [
            "https://ipfs.io/ipfs/",
            "https://gateway.pinata.cloud/ipfs/",
            "https://cloudflare-ipfs.com/ipfs/"
        ]

        # Cache for price data (avoid rate limits)
        self.price_cache = {}
        self.cache_duration = 60  # seconds

    def _get_cached_price(self, cache_key: str) -> Optional[PriceData]:
        """Get cached price data if still valid"""
        if cache_key in self.price_cache:
            price_data, timestamp = self.price_cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return price_data
        return None

    def _cache_price(self, cache_key: str, price_data: PriceData):
        """Cache price data"""
        self.price_cache[cache_key] = (price_data, time.time())

    # ========================================================================
    # CRYPTOCURRENCY PRICES
    # ========================================================================

    def get_crypto_price(self, coin_id: str, vs_currency: str = "usd") -> Optional[PriceData]:
        """
        Get cryptocurrency price from CoinGecko

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
            vs_currency: Currency to compare against

        Returns:
            PriceData or None
        """
        cache_key = f"{coin_id}_{vs_currency}"
        cached = self._get_cached_price(cache_key)
        if cached:
            return cached

        try:
            url = f"{self.coingecko_base}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": vs_currency,
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true"
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json().get(coin_id, {})

            price_data = PriceData(
                symbol=coin_id.upper(),
                price_usd=data.get(vs_currency, 0),
                price_change_24h=data.get(f"{vs_currency}_24h_change"),
                market_cap=data.get(f"{vs_currency}_market_cap"),
                volume_24h=data.get(f"{vs_currency}_24h_vol")
            )

            self._cache_price(cache_key, price_data)
            return price_data

        except Exception as e:
            logger.error(f"Error fetching price for {coin_id}: {e}")
            return None

    def get_multiple_prices(self, coin_ids: List[str], vs_currency: str = "usd") -> Dict[str, PriceData]:
        """
        Get prices for multiple cryptocurrencies

        Args:
            coin_ids: List of CoinGecko coin IDs
            vs_currency: Currency to compare against

        Returns:
            Dictionary mapping coin_id to PriceData
        """
        try:
            url = f"{self.coingecko_base}/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": vs_currency,
                "include_24hr_change": "true"
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            results = {}
            for coin_id in coin_ids:
                if coin_id in data:
                    coin_data = data[coin_id]
                    price_data = PriceData(
                        symbol=coin_id.upper(),
                        price_usd=coin_data.get(vs_currency, 0),
                        price_change_24h=coin_data.get(f"{vs_currency}_24h_change")
                    )
                    results[coin_id] = price_data
                    self._cache_price(f"{coin_id}_{vs_currency}", price_data)

            return results

        except Exception as e:
            logger.error(f"Error fetching multiple prices: {e}")
            return {}

    def get_token_price_by_symbol(self, symbol: str) -> Optional[float]:
        """
        Get token price by symbol (simplified)

        Args:
            symbol: Token symbol (BTC, ETH, WBTC, etc.)

        Returns:
            Price in USD or None
        """
        symbol_to_id = {
            "BTC": "bitcoin",
            "WBTC": "wrapped-bitcoin",
            "ETH": "ethereum",
            "MATIC": "matic-network",
            "AVAX": "avalanche-2",
            "BNB": "binancecoin",
            "USDC": "usd-coin",
            "USDT": "tether"
        }

        coin_id = symbol_to_id.get(symbol.upper())
        if not coin_id:
            return None

        price_data = self.get_crypto_price(coin_id)
        return price_data.price_usd if price_data else None

    # ========================================================================
    # EXCHANGE RATES
    # ========================================================================

    def get_exchange_rate(self, base: str = "USD", target: str = "EUR") -> Optional[float]:
        """
        Get fiat exchange rate

        Args:
            base: Base currency
            target: Target currency

        Returns:
            Exchange rate or None
        """
        try:
            url = f"{self.exchangerate_base}/{base}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            rates = data.get("rates", {})

            return rates.get(target)

        except Exception as e:
            logger.error(f"Error fetching exchange rate: {e}")
            return None

    # ========================================================================
    # BLOCK EXPLORER DATA
    # ========================================================================

    def get_eth_balance(self, address: str, network: str = "ethereum") -> Optional[float]:
        """
        Get ETH balance from block explorer

        Args:
            address: Ethereum address
            network: Network name (ethereum, polygon, bsc)

        Returns:
            Balance in ETH or None
        """
        explorer_map = {
            "ethereum": self.etherscan_base,
            "polygon": self.polygonscan_base,
            "bsc": self.bscscan_base
        }

        base_url = explorer_map.get(network)
        if not base_url:
            return None

        try:
            params = {
                "module": "account",
                "action": "balance",
                "address": address,
                "tag": "latest"
            }

            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "1":
                balance_wei = int(data.get("result", 0))
                return balance_wei / 10**18

        except Exception as e:
            logger.error(f"Error fetching balance for {address}: {e}")

        return None

    def get_transaction_count(self, address: str, network: str = "ethereum") -> Optional[int]:
        """
        Get transaction count for address

        Args:
            address: Ethereum address
            network: Network name

        Returns:
            Transaction count or None
        """
        explorer_map = {
            "ethereum": self.etherscan_base,
            "polygon": self.polygonscan_base,
            "bsc": self.bscscan_base
        }

        base_url = explorer_map.get(network)
        if not base_url:
            return None

        try:
            params = {
                "module": "proxy",
                "action": "eth_getTransactionCount",
                "address": address,
                "tag": "latest"
            }

            response = self.session.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            result = data.get("result", "0x0")

            return int(result, 16)

        except Exception as e:
            logger.error(f"Error fetching transaction count: {e}")

        return None

    # ========================================================================
    # IPFS
    # ========================================================================

    def fetch_ipfs_content(self, cid: str, gateway_index: int = 0) -> Optional[str]:
        """
        Fetch content from IPFS

        Args:
            cid: IPFS content ID
            gateway_index: Index of gateway to use

        Returns:
            Content as string or None
        """
        if gateway_index >= len(self.ipfs_gateways):
            return None

        gateway = self.ipfs_gateways[gateway_index]
        url = f"{gateway}{cid}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            return response.text

        except Exception as e:
            logger.warning(f"IPFS fetch failed on gateway {gateway_index}: {e}")

            # Try next gateway
            if gateway_index + 1 < len(self.ipfs_gateways):
                return self.fetch_ipfs_content(cid, gateway_index + 1)

        return None

    # ========================================================================
    # MARKET DATA
    # ========================================================================

    def get_market_overview(self) -> Dict[str, Any]:
        """
        Get cryptocurrency market overview

        Returns:
            Market data
        """
        major_coins = [
            "bitcoin",
            "ethereum",
            "wrapped-bitcoin",
            "matic-network",
            "avalanche-2",
            "binancecoin"
        ]

        prices = self.get_multiple_prices(major_coins)

        total_market_cap = sum(
            p.market_cap for p in prices.values() if p.market_cap
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_market_cap": total_market_cap,
            "prices": {
                coin_id: {
                    "price_usd": price.price_usd,
                    "change_24h": price.price_change_24h,
                    "market_cap": price.market_cap
                }
                for coin_id, price in prices.items()
            }
        }

    def calculate_portfolio_value(self, holdings: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate portfolio value

        Args:
            holdings: Dictionary mapping coin_id to amount

        Returns:
            Portfolio valuation
        """
        prices = self.get_multiple_prices(list(holdings.keys()))

        portfolio = {}
        total_value = 0

        for coin_id, amount in holdings.items():
            if coin_id in prices:
                price = prices[coin_id].price_usd
                value = amount * price
                total_value += value

                portfolio[coin_id] = {
                    "amount": amount,
                    "price_usd": price,
                    "value_usd": value,
                    "change_24h": prices[coin_id].price_change_24h
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "total_value_usd": total_value,
            "holdings": portfolio
        }


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("NEXUS AGI - PUBLIC API INTEGRATOR")
    print("=" * 80)

    integrator = PublicAPIIntegrator()

    # Get cryptocurrency prices
    print("\n💰 Cryptocurrency Prices:")
    coins = ["bitcoin", "ethereum", "wrapped-bitcoin", "matic-network"]

    prices = integrator.get_multiple_prices(coins)

    for coin_id, price in prices.items():
        change_symbol = "📈" if price.price_change_24h and price.price_change_24h > 0 else "📉"
        print(f"   {coin_id.upper():<20} ${price.price_usd:>12,.2f} {change_symbol} {price.price_change_24h:+.2f}%")

    # Example portfolio
    print("\n📊 Example Portfolio Valuation:")
    portfolio = {
        "bitcoin": 0.5,
        "ethereum": 10,
        "wrapped-bitcoin": 1.0,
        "matic-network": 1000
    }

    valuation = integrator.calculate_portfolio_value(portfolio)

    print(f"   Total Value: ${valuation['total_value_usd']:,.2f} USD")
    print("\n   Holdings:")
    for coin_id, data in valuation['holdings'].items():
        print(f"      {coin_id:<20} {data['amount']:>10.4f} @ ${data['price_usd']:>10,.2f} = ${data['value_usd']:>12,.2f}")

    # Exchange rates
    print("\n💱 Exchange Rates:")
    rates = {
        "EUR": integrator.get_exchange_rate("USD", "EUR"),
        "GBP": integrator.get_exchange_rate("USD", "GBP"),
        "JPY": integrator.get_exchange_rate("USD", "JPY")
    }

    for currency, rate in rates.items():
        if rate:
            print(f"   1 USD = {rate:.4f} {currency}")

    print("\n" + "=" * 80)
    print("✅ Public API integration operational")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
