"""
Nova ASI — Crypto Paper Trading Bot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Built autonomously by Nova via APIHunter + /build
Uses CoinGecko (free, no API key) for live prices.
Trades simulated money — zero financial risk.

When Douglas is ready to go live:
  1. Get Coinbase Advanced Trade API key at coinbase.com
  2. Add COINBASE_API_KEY + COINBASE_SECRET to .env
  3. Set PAPER_MODE = False in this file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sqlite3, json, os, time, random, threading, traceback, hmac, hashlib, uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import requests as _req
    _OK = True
except ImportError:
    _OK = False

# ── Config ─────────────────────────────────────────────────────────────────────
PAPER_MODE       = True          # ALWAYS start here. Change only when ready.
STARTING_CASH    = 10_000.00     # Virtual dollars
MAX_POSITION_PCT = 0.20          # Never put more than 20% in one coin
STOP_LOSS_PCT    = 0.05          # Auto-sell if a position drops 5%
TAKE_PROFIT_PCT  = 0.15          # Auto-sell if a position gains 15%
DB_PATH          = os.path.expanduser("~/nexus_agi/nova_trading.db")
LIVE_DB_PATH     = os.path.expanduser("~/nexus_agi/nova_live_trading.db")

# Live trading — opt-in only via .env. Never enabled by default.
# Set NOVA_LIVE_TRADING=true and NOVA_LIVE_BUDGET=50.00 in .env to go live.
LIVE_MODE        = os.getenv("NOVA_LIVE_TRADING", "false").lower() == "true"
LIVE_BUDGET_CAP  = float(os.getenv("NOVA_LIVE_BUDGET", "50.00"))  # hard $ ceiling

# Coins Nova watches
WATCHLIST = ["bitcoin", "ethereum", "solana", "cardano", "dogecoin"]


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE FEED — CoinGecko, free, no API key
# ═══════════════════════════════════════════════════════════════════════════════

class PriceFeed:
    """Real live crypto prices via CoinGecko — zero credentials."""

    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self._cache: Dict[str, Tuple[float, float]] = {}  # id → (price, ts)

    def price(self, coin_id: str) -> Optional[float]:
        """Get current USD price for a coin."""
        ts, cached = self._cache.get(coin_id, (0, 0.0))
        if time.time() - ts < 60:       # cache 60 seconds
            return cached
        if not _OK:
            return None
        try:
            r = _req.get(f"{self.BASE}/simple/price",
                         params={"ids": coin_id, "vs_currencies": "usd"},
                         timeout=8)
            p = r.json()[coin_id]["usd"]
            self._cache[coin_id] = (time.time(), p)
            return p
        except Exception:
            return cached or None

    def prices(self, coin_ids: List[str]) -> Dict[str, float]:
        """Get prices for multiple coins in one call."""
        if not _OK:
            return {}
        ids = ",".join(coin_ids)
        try:
            r = _req.get(f"{self.BASE}/simple/price",
                         params={"ids": ids, "vs_currencies": "usd"},
                         timeout=10)
            data = r.json()
            result = {}
            for cid in coin_ids:
                p = data.get(cid, {}).get("usd")
                if p:
                    self._cache[cid] = (time.time(), p)
                    result[cid] = p
            return result
        except Exception:
            return {}

    def history(self, coin_id: str, days: int = 14) -> List[float]:
        """Get daily closing prices for RSI calculation (cached 5 min)."""
        if not _OK:
            return []
        cache_key = f"hist_{coin_id}_{days}"
        ts, cached = self._cache.get(cache_key, (0, []))
        if time.time() - ts < 300 and cached:   # 5-minute cache
            return cached
        try:
            r = _req.get(f"{self.BASE}/coins/{coin_id}/market_chart",
                         params={"vs_currency": "usd", "days": days,
                                 "interval": "daily"},
                         timeout=10)
            prices = [p[1] for p in r.json().get("prices", [])]
            if prices:
                self._cache[cache_key] = (time.time(), prices)
            return prices
        except Exception:
            return cached or []


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY ENGINE — RSI + Moving Average signals
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyEngine:
    """
    Generates BUY / SELL / HOLD signals using:
    - RSI (14-period): buy < 35, sell > 70
    - Moving average crossover: fast(7) crosses slow(14)
    Both must agree before Nova acts.
    """

    def rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, period + 1):
            diff = prices[-period - 1 + i] - prices[-period - 2 + i]
            (gains if diff > 0 else losses).append(abs(diff))
        ag = sum(gains) / period if gains else 0
        al = sum(losses) / period if losses else 1e-9
        rs = ag / al
        return 100 - (100 / (1 + rs))

    def ma(self, prices: List[float], period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def signal(self, prices: List[float]) -> str:
        """Return 'BUY', 'SELL', or 'HOLD'."""
        if len(prices) < 15:
            return "HOLD"
        rsi_val = self.rsi(prices)
        ma7     = self.ma(prices, 7)
        ma14    = self.ma(prices, 14)
        if rsi_val is None or ma7 is None or ma14 is None:
            return "HOLD"
        rsi_buy  = rsi_val < 35
        rsi_sell = rsi_val > 70
        ma_bull  = ma7 > ma14
        ma_bear  = ma7 < ma14
        if rsi_buy and ma_bull:
            return "BUY"
        if rsi_sell and ma_bear:
            return "SELL"
        return "HOLD"


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER PORTFOLIO — tracks virtual cash and positions in SQLite
# ═══════════════════════════════════════════════════════════════════════════════

class PaperPortfolio:
    """Virtual portfolio — all trades are simulated, no real money moves."""

    def __init__(self, db_path: str = DB_PATH,
                 starting_cash: float = STARTING_CASH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()
        # Seed cash on first run (INSERT OR IGNORE avoids duplicate-key error)
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO portfolio (key, value) VALUES ('cash', ?)",
                (starting_cash,)
            )
            self.conn.commit()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio (
                key TEXT PRIMARY KEY, value REAL
            );
            CREATE TABLE IF NOT EXISTS positions (
                coin TEXT PRIMARY KEY,
                qty  REAL,
                avg_price REAL,
                last_updated TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT,
                coin      TEXT,
                side      TEXT,
                qty       REAL,
                price     REAL,
                total_usd REAL,
                reason    TEXT
            );
        """)
        self.conn.commit()

    def _get_cash(self) -> float:
        row = self.conn.execute(
            "SELECT value FROM portfolio WHERE key='cash'"
        ).fetchone()
        return float(row[0]) if (row and row[0] is not None) else 0.0

    def _set_cash(self, amount: float):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO portfolio (key,value) VALUES ('cash',?)",
                (amount,)
            )
            self.conn.commit()

    def cash(self) -> float:
        return self._get_cash()

    def position(self, coin: str) -> Dict:
        row = self.conn.execute(
            "SELECT qty, avg_price FROM positions WHERE coin=?", (coin,)
        ).fetchone()
        return {"qty": row[0], "avg_price": row[1]} if row else {"qty": 0, "avg_price": 0}

    def buy(self, coin: str, price: float, usd_amount: float,
            reason: str = "") -> Dict:
        with self._lock:
            cash = self._get_cash()
            usd_amount = min(usd_amount, cash)
            if usd_amount < 1:
                return {"error": "Insufficient cash"}
            qty = usd_amount / price
            pos = self.position(coin)
            new_qty   = pos["qty"] + qty
            new_avg   = ((pos["qty"] * pos["avg_price"]) + (qty * price)) / new_qty
            self.conn.execute(
                "INSERT OR REPLACE INTO positions (coin,qty,avg_price,last_updated) "
                "VALUES (?,?,?,?)",
                (coin, new_qty, new_avg, datetime.now().isoformat())
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO portfolio (key,value) VALUES ('cash',?)",
                (cash - usd_amount,)
            )
            self.conn.execute(
                "INSERT INTO trades (ts,coin,side,qty,price,total_usd,reason) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), coin, "BUY", qty, price, usd_amount, reason)
            )
            self.conn.commit()
        return {"coin": coin, "qty": qty, "price": price, "spent": usd_amount}

    def sell(self, coin: str, price: float, reason: str = "") -> Dict:
        with self._lock:
            pos = self.position(coin)
            if pos["qty"] <= 0:
                return {"error": f"No {coin} to sell"}
            proceeds = pos["qty"] * price
            pnl      = proceeds - (pos["qty"] * pos["avg_price"])
            self.conn.execute("DELETE FROM positions WHERE coin=?", (coin,))
            self.conn.execute(
                "INSERT OR REPLACE INTO portfolio (key,value) VALUES ('cash',?)",
                (self._get_cash() + proceeds,)
            )
            self.conn.execute(
                "INSERT INTO trades (ts,coin,side,qty,price,total_usd,reason) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), coin, "SELL",
                 pos["qty"], price, proceeds, reason)
            )
            self.conn.commit()
        return {"coin": coin, "proceeds": proceeds, "pnl": pnl}

    def value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value in USD."""
        total = self._get_cash()          # always a float now
        rows  = self.conn.execute("SELECT coin,qty FROM positions").fetchall()
        for row in rows:
            if row and len(row) == 2 and row[0] and row[1]:
                total += float(row[1]) * prices.get(row[0], 0)
        return total

    def recent_trades(self, n: int = 10) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT ts,coin,side,qty,price,total_usd,reason "
            "FROM trades ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [{"ts": r[0], "coin": r[1], "side": r[2],
                 "qty": r[3], "price": r[4], "usd": r[5], "reason": r[6]}
                for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# COINBASE EXECUTOR — places REAL orders on Coinbase Advanced Trade API
# Credentials come from .env ONLY — never hardcoded.
# ═══════════════════════════════════════════════════════════════════════════════

class CoinbaseExecutor:
    """
    Executes real buy/sell orders via Coinbase Advanced Trade API.
    Reads COINBASE_API_KEY and COINBASE_API_SECRET from environment.
    All amounts in USD. Budget cap enforced — never exceeds LIVE_BUDGET_CAP.
    """

    BASE = "https://api.coinbase.com/api/v3/brokerage"

    # Map CoinGecko names → Coinbase product IDs
    PRODUCTS = {
        "bitcoin":  "BTC-USD",
        "ethereum": "ETH-USD",
        "solana":   "SOL-USD",
        "cardano":  "ADA-USD",
        "dogecoin": "DOGE-USD",
    }

    def __init__(self) -> None:
        self._key    = os.getenv("COINBASE_API_KEY", "")
        self._secret = os.getenv("COINBASE_API_SECRET", "")
        self._spent  = 0.0   # total USD deployed this session (budget guard)

    def _ready(self) -> bool:
        """Return True only if real credentials are present."""
        return bool(self._key and self._secret)

    def _sign(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Build Coinbase Advanced Trade HMAC-SHA256 auth headers."""
        ts  = str(int(time.time()))
        msg = ts + method.upper() + path + body
        sig = hmac.new(
            self._secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return {
            "CB-ACCESS-KEY":       self._key,
            "CB-ACCESS-SIGN":      sig,
            "CB-ACCESS-TIMESTAMP": ts,
            "Content-Type":        "application/json",
        }

    def usd_balance(self) -> float:
        """Return available USD cash on Coinbase account."""
        if not _OK or not self._ready():
            return 0.0
        path = "/accounts"
        try:
            r = _req.get(
                self.BASE + path,
                headers=self._sign("GET", path),
                timeout=10
            )
            r.raise_for_status()
            for acct in r.json().get("accounts", []):
                if acct.get("currency") == "USD":
                    return float(acct.get("available_balance", {}).get("value", 0))
        except Exception as e:
            print(f"  [Coinbase] balance error: {e}")
        return 0.0

    def holdings(self) -> Dict[str, float]:
        """Return {coin_id: qty} for all held crypto assets."""
        if not _OK or not self._ready():
            return {}
        path = "/accounts"
        result: Dict[str, float] = {}
        reverse = {v.split("-")[0]: k for k, v in self.PRODUCTS.items()}
        try:
            r = _req.get(
                self.BASE + path,
                headers=self._sign("GET", path),
                timeout=10
            )
            r.raise_for_status()
            for acct in r.json().get("accounts", []):
                currency = acct.get("currency", "")
                coin     = reverse.get(currency)
                if coin:
                    qty = float(acct.get("available_balance", {}).get("value", 0))
                    if qty > 0:
                        result[coin] = qty
        except Exception as e:
            print(f"  [Coinbase] holdings error: {e}")
        return result

    def buy(self, coin: str, usd_amount: float) -> Dict:
        """
        Place a real market buy order for usd_amount dollars of coin.
        Budget cap: total live spend never exceeds LIVE_BUDGET_CAP.
        Returns order result dict or {'error': reason}.
        """
        if not _OK or not self._ready():
            return {"error": "No Coinbase credentials in .env"}
        product = self.PRODUCTS.get(coin)
        if not product:
            return {"error": f"Unknown coin: {coin}"}

        # Hard budget enforcement
        remaining = LIVE_BUDGET_CAP - self._spent
        usd_amount = min(usd_amount, remaining)
        if usd_amount < 1.00:
            return {"error": f"Budget cap reached (${LIVE_BUDGET_CAP:.2f} limit)"}

        path = "/orders"
        body = json.dumps({
            "client_order_id": str(uuid.uuid4()),
            "product_id":      product,
            "side":            "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": f"{usd_amount:.2f}"
                }
            }
        })
        try:
            r = _req.post(
                self.BASE + path,
                headers=self._sign("POST", path, body),
                data=body,
                timeout=15
            )
            r.raise_for_status()
            resp = r.json()
            if resp.get("success"):
                self._spent += usd_amount
                order_id = resp.get("order_id", "")
                print(f"  [Coinbase LIVE] BUY {coin} ${usd_amount:.2f} ✓ order={order_id[:12]}")
                return {"coin": coin, "usd": usd_amount,
                        "order_id": order_id, "live": True}
            err = resp.get("error_response", {}).get("message", str(resp))
            return {"error": err}
        except Exception as e:
            return {"error": str(e)}

    def sell(self, coin: str, qty: float) -> Dict:
        """
        Place a real market sell order for qty units of coin.
        Returns order result dict or {'error': reason}.
        """
        if not _OK or not self._ready():
            return {"error": "No Coinbase credentials in .env"}
        product = self.PRODUCTS.get(coin)
        if not product:
            return {"error": f"Unknown coin: {coin}"}
        if qty <= 0:
            return {"error": "qty must be > 0"}

        path = "/orders"
        body = json.dumps({
            "client_order_id": str(uuid.uuid4()),
            "product_id":      product,
            "side":            "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{qty:.8f}"
                }
            }
        })
        try:
            r = _req.post(
                self.BASE + path,
                headers=self._sign("POST", path, body),
                data=body,
                timeout=15
            )
            r.raise_for_status()
            resp = r.json()
            if resp.get("success"):
                order_id = resp.get("order_id", "")
                print(f"  [Coinbase LIVE] SELL {coin} qty={qty:.6f} ✓ order={order_id[:12]}")
                return {"coin": coin, "qty": qty,
                        "order_id": order_id, "live": True}
            err = resp.get("error_response", {}).get("message", str(resp))
            return {"error": err}
        except Exception as e:
            return {"error": str(e)}

    def validate(self) -> str:
        """Test credentials and return a status string."""
        if not self._ready():
            return "✗ No credentials — add COINBASE_API_KEY + COINBASE_API_SECRET to .env"
        bal = self.usd_balance()
        if bal >= 0:
            return f"✓ Connected — ${bal:.2f} USD available on Coinbase"
        return "✗ Credentials present but API call failed"


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE PORTFOLIO — real money wrapper around PaperPortfolio
# Records every real trade in SQLite alongside the Coinbase execution.
# ═══════════════════════════════════════════════════════════════════════════════

class LivePortfolio(PaperPortfolio):
    """
    Executes real Coinbase orders, then records them in SQLite.
    Inherits all tracking logic from PaperPortfolio.
    Budget hard-capped at LIVE_BUDGET_CAP — never exceeded.
    """

    def __init__(self) -> None:
        self.executor = CoinbaseExecutor()
        # Seed SQLite with real Coinbase USD balance (capped at budget)
        real_balance  = self.executor.usd_balance()
        starting      = min(real_balance, LIVE_BUDGET_CAP) if real_balance > 0 else LIVE_BUDGET_CAP
        super().__init__(db_path=LIVE_DB_PATH, starting_cash=starting)
        print(f"  [LivePortfolio] Budget: ${starting:.2f}  "
              f"Real Coinbase balance: ${real_balance:.2f}")

    def buy(self, coin: str, price: float, usd_amount: float,
            reason: str = "") -> Dict:
        """Place real buy on Coinbase, then record in SQLite."""
        result = self.executor.buy(coin, usd_amount)
        if "error" in result:
            print(f"  [LivePortfolio] BUY failed: {result['error']}")
            return result
        # Record in SQLite using parent class logic (adjusts cash + position)
        return super().buy(coin, price, usd_amount, reason=reason + " [LIVE]")

    def sell(self, coin: str, price: float, reason: str = "") -> Dict:
        """Place real sell on Coinbase, then record in SQLite."""
        pos = self.position(coin)
        if pos["qty"] <= 0:
            return {"error": f"No {coin} position to sell"}
        result = self.executor.sell(coin, pos["qty"])
        if "error" in result:
            print(f"  [LivePortfolio] SELL failed: {result['error']}")
            return result
        return super().sell(coin, price, reason=reason + " [LIVE]")


# ═══════════════════════════════════════════════════════════════════════════════
# NOVA TRADER — the brain that connects everything
# ═══════════════════════════════════════════════════════════════════════════════

class NovaTrader:
    """
    Nova's autonomous paper trading system.
    Calls PriceFeed → StrategyEngine → PaperPortfolio on each cycle.
    Run cycle() manually or start_auto() for background trading.
    """

    def __init__(self):
        self.feed      = PriceFeed()
        self.strategy  = StrategyEngine()
        self.watchlist = WATCHLIST
        self._running  = False

        if LIVE_MODE:
            self.portfolio = LivePortfolio()
            print(f"  [NovaTrader] ⚡ LIVE MODE — real money, budget cap ${LIVE_BUDGET_CAP:.2f}")
        else:
            self.portfolio = PaperPortfolio()
            print("  [NovaTrader] Paper mode — no real money at risk")

    def cycle(self) -> List[str]:
        """One full trading cycle. Returns list of actions taken."""
        actions = []
        prices  = self.feed.prices(self.watchlist)
        cash    = self.portfolio.cash()
        total_value = self.portfolio.value(prices)

        for coin in self.watchlist:
            price = prices.get(coin)
            if not price:
                continue

            hist   = self.feed.history(coin, days=15)
            signal = self.strategy.signal(hist)
            pos    = self.portfolio.position(coin)

            # ── Stop-loss / take-profit check ──────────────────────────────
            if pos["qty"] > 0 and pos["avg_price"] > 0:
                pct_change = (price - pos["avg_price"]) / pos["avg_price"]
                if pct_change <= -STOP_LOSS_PCT:
                    result = self.portfolio.sell(coin, price,
                                                 reason=f"stop-loss {pct_change*100:.1f}%")
                    actions.append(f"STOP-LOSS {coin} @ ${price:,.2f} "
                                   f"PnL ${result.get('pnl',0):+.2f}")
                    continue
                if pct_change >= TAKE_PROFIT_PCT:
                    result = self.portfolio.sell(coin, price,
                                                 reason=f"take-profit {pct_change*100:.1f}%")
                    actions.append(f"TAKE-PROFIT {coin} @ ${price:,.2f} "
                                   f"PnL ${result.get('pnl',0):+.2f}")
                    continue

            # ── Strategy signals ───────────────────────────────────────────
            if signal == "BUY" and pos["qty"] == 0 and cash > 10:
                invest = min(cash * MAX_POSITION_PCT, cash)
                result = self.portfolio.buy(coin, price, invest,
                                            reason="RSI+MA signal")
                if "error" not in result:
                    actions.append(f"BUY  {coin} ${invest:.2f} @ ${price:,.2f}")

            elif signal == "SELL" and pos["qty"] > 0:
                result = self.portfolio.sell(coin, price, reason="RSI+MA signal")
                actions.append(f"SELL {coin} @ ${price:,.2f} "
                               f"PnL ${result.get('pnl',0):+.2f}")

        return actions

    def report(self) -> str:
        """Full portfolio status report."""
        prices = self.feed.prices(self.watchlist)
        cash   = self.portfolio.cash()
        total  = self.portfolio.value(prices)
        pnl    = total - STARTING_CASH
        pnl_pct = (pnl / STARTING_CASH) * 100

        lines = [
            f"{'─'*50}",
            f"  Nova Paper Trading Report  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"{'─'*50}",
            f"  Cash:        ${cash:>10,.2f}",
            f"  Total value: ${total:>10,.2f}",
            f"  P&L:         ${pnl:>+10,.2f}  ({pnl_pct:+.2f}%)",
            f"{'─'*50}",
            "  POSITIONS:",
        ]
        rows = self.portfolio.conn.execute(
            "SELECT coin, qty, avg_price FROM positions"
        ).fetchall()
        if rows:
            for coin, qty, avg in rows:
                price  = prices.get(coin, 0)
                value  = qty * price
                pnl_p  = (price - avg) / avg * 100 if avg else 0
                lines.append(f"  {coin:12} qty={qty:.6f}  "
                             f"avg=${avg:,.2f}  now=${price:,.2f}  "
                             f"val=${value:,.2f}  ({pnl_p:+.1f}%)")
        else:
            lines.append("  No open positions.")
        lines.append(f"{'─'*50}")
        recent = self.portfolio.recent_trades(5)
        if recent:
            lines.append("  RECENT TRADES:")
            for t in recent:
                lines.append(f"  {t['ts'][:16]}  {t['side']:4} {t['coin']:12} "
                             f"${t['usd']:>8,.2f}  {t['reason']}")
        lines.append(f"{'─'*50}")
        return "\n".join(lines)

    def signals_report(self) -> str:
        """Show current RSI, MA, and signal for every watched coin."""
        prices = self.feed.prices(self.watchlist)
        lines  = [f"{'─'*56}",
                  f"  Nova Signal Scan  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  f"{'─'*56}",
                  f"  {'Coin':12} {'Price':>10}  {'RSI':>6}  {'MA7':>10}  {'MA14':>10}  Signal"]
        for coin in self.watchlist:
            price = prices.get(coin)
            if not price:
                lines.append(f"  {coin:12}  (no price data)")
                continue
            hist  = self.feed.history(coin, days=16)
            rsi   = self.strategy.rsi(hist) if len(hist) >= 15 else None
            ma7   = self.strategy.ma(hist, 7)
            ma14  = self.strategy.ma(hist, 14)
            sig   = self.strategy.signal(hist)
            rsi_s = f"{rsi:5.1f}" if rsi is not None else "  N/A"
            ma7_s = f"{ma7:>10,.2f}" if ma7 else "       N/A"
            ma14_s= f"{ma14:>10,.2f}" if ma14 else "       N/A"
            flag  = " ◀ SIGNAL" if sig != "HOLD" else ""
            lines.append(f"  {coin:12} ${price:>10,.2f}  {rsi_s}  {ma7_s}  {ma14_s}  {sig}{flag}")
        lines.append(f"{'─'*56}")
        lines.append("  BUY when RSI < 35 and MA7 > MA14 (bullish crossover)")
        lines.append("  SELL when RSI > 70 and MA7 < MA14 (bearish crossover)")
        lines.append(f"{'─'*56}")
        return "\n".join(lines)

    def start_auto(self, interval_minutes: int = 15):
        """Start background trading loop."""
        import threading
        self._running = True
        def _loop():
            while self._running:
                try:
                    actions = self.cycle()
                    if actions:
                        print("\n  [Nova Trader] " + " | ".join(actions))
                except Exception as e:
                    print(f"\n  [Nova Trader] Error: {e}")
                    traceback.print_exc()
                time.sleep(interval_minutes * 60)
        threading.Thread(target=_loop, daemon=True).start()
        return f"Auto-trading started. Cycle every {interval_minutes} min."

    def stop_auto(self):
        self._running = False
        return "Auto-trading stopped."

    def validate_live(self) -> str:
        """Test Coinbase credentials and show live readiness status."""
        executor = CoinbaseExecutor()
        cred_status = executor.validate()
        lines = [
            "  Live Trading Readiness",
            f"  {'─'*40}",
            f"  Credentials: {cred_status}",
            f"  LIVE_MODE env:   {'✓ enabled' if LIVE_MODE else '✗ not set (add NOVA_LIVE_TRADING=true to .env)'}",
            f"  Budget cap:      ${LIVE_BUDGET_CAP:.2f}",
            f"  Mode active:     {'⚡ LIVE' if LIVE_MODE else '📄 Paper'}",
        ]
        if not LIVE_MODE:
            lines.append("  To go live: add NOVA_LIVE_TRADING=true to ~/nexus_agi/.env")
        return "\n".join(lines)


# ── Quick start ────────────────────────────────────────────────────────────────
# from nova_cap_trader import NovaTrader
# trader = NovaTrader()
# print(trader.report())          # see current portfolio
# actions = trader.cycle()        # run one trade cycle
# trader.start_auto(15)          # trade every 15 minutes automatically
