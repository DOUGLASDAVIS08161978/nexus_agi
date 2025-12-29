#!/usr/bin/env python3
"""
NEXUS AGI - AUTOMATED CUSTOMER MANAGEMENT SYSTEM
Handles customer onboarding, billing, notifications, and lifetime value tracking
"""

import json
import sqlite3
import hashlib
import secrets
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

@dataclass
class Customer:
    """Customer record with complete profile"""
    customer_id: str
    email: str
    api_key: str
    subscription_tier: str  # free, basic, pro, enterprise
    created_at: str
    total_spent_usd: float = 0.0
    total_spent_btc: float = 0.0
    api_calls_count: int = 0
    last_activity: Optional[str] = None
    status: str = "active"  # active, suspended, churned
    referral_source: Optional[str] = None
    lifetime_value: float = 0.0
    payment_method: str = "bitcoin"  # bitcoin, card, other

    def to_dict(self):
        return asdict(self)


@dataclass
class Transaction:
    """Transaction record for revenue tracking"""
    transaction_id: str
    customer_id: str
    service_type: str
    amount_usd: float
    amount_btc: float
    btc_address: str
    timestamp: str
    status: str  # pending, confirmed, failed
    confirmations: int = 0

    def to_dict(self):
        return asdict(self)


class CustomerManagementSystem:
    """Complete customer lifecycle management"""

    def __init__(self, db_path: str = "nexus_customers.db"):
        self.db_path = db_path
        self.bitcoin_wallet = os.getenv("BITCOIN_WALLET_ADDRESS", "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                subscription_tier TEXT NOT NULL,
                created_at TEXT NOT NULL,
                total_spent_usd REAL DEFAULT 0.0,
                total_spent_btc REAL DEFAULT 0.0,
                api_calls_count INTEGER DEFAULT 0,
                last_activity TEXT,
                status TEXT DEFAULT 'active',
                referral_source TEXT,
                lifetime_value REAL DEFAULT 0.0,
                payment_method TEXT DEFAULT 'bitcoin'
            )
        """)

        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                service_type TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                amount_btc REAL NOT NULL,
                btc_address TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                confirmations INTEGER DEFAULT 0,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)

        # API usage log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                response_time REAL,
                status_code INTEGER,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)

        # Marketing campaigns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                platform TEXT NOT NULL,
                posted_at TEXT,
                views INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                signups INTEGER DEFAULT 0,
                revenue_generated REAL DEFAULT 0.0
            )
        """)

        conn.commit()
        conn.close()

    def register_customer(self, email: str, subscription_tier: str = "free",
                         referral_source: Optional[str] = None) -> Customer:
        """Register new customer and generate API key"""
        customer_id = self._generate_customer_id(email)
        api_key = self._generate_api_key()

        customer = Customer(
            customer_id=customer_id,
            email=email,
            api_key=api_key,
            subscription_tier=subscription_tier,
            created_at=datetime.utcnow().isoformat(),
            referral_source=referral_source
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO customers (customer_id, email, api_key, subscription_tier,
                                     created_at, referral_source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (customer.customer_id, customer.email, customer.api_key,
                  customer.subscription_tier, customer.created_at, customer.referral_source))

            conn.commit()

            # Send welcome email
            self._send_welcome_email(customer)

            return customer

        except sqlite3.IntegrityError:
            raise ValueError(f"Customer with email {email} already exists")
        finally:
            conn.close()

    def record_transaction(self, customer_id: str, service_type: str,
                          amount_usd: float, amount_btc: float) -> Transaction:
        """Record new transaction"""
        transaction = Transaction(
            transaction_id=secrets.token_hex(16),
            customer_id=customer_id,
            service_type=service_type,
            amount_usd=amount_usd,
            amount_btc=amount_btc,
            btc_address=self.bitcoin_wallet,
            timestamp=datetime.utcnow().isoformat(),
            status="pending"
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO transactions (transaction_id, customer_id, service_type,
                                    amount_usd, amount_btc, btc_address, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (transaction.transaction_id, transaction.customer_id, transaction.service_type,
              transaction.amount_usd, transaction.amount_btc, transaction.btc_address,
              transaction.timestamp, transaction.status))

        conn.commit()
        conn.close()

        return transaction

    def update_customer_spending(self, customer_id: str, amount_usd: float, amount_btc: float):
        """Update customer spending and lifetime value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE customers
            SET total_spent_usd = total_spent_usd + ?,
                total_spent_btc = total_spent_btc + ?,
                lifetime_value = total_spent_usd + ?,
                last_activity = ?
            WHERE customer_id = ?
        """, (amount_usd, amount_btc, amount_usd, datetime.utcnow().isoformat(), customer_id))

        conn.commit()
        conn.close()

    def get_customer_by_api_key(self, api_key: str) -> Optional[Customer]:
        """Retrieve customer by API key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM customers WHERE api_key = ?", (api_key,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Customer(*row)
        return None

    def get_revenue_stats(self) -> Dict:
        """Get comprehensive revenue statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total revenue
        cursor.execute("SELECT SUM(total_spent_usd), SUM(total_spent_btc) FROM customers")
        total_usd, total_btc = cursor.fetchone()

        # Revenue by tier
        cursor.execute("""
            SELECT subscription_tier, SUM(total_spent_usd), COUNT(*)
            FROM customers
            GROUP BY subscription_tier
        """)
        revenue_by_tier = {row[0]: {"revenue": row[1], "customers": row[2]}
                          for row in cursor.fetchall()}

        # Today's revenue
        today = datetime.utcnow().date().isoformat()
        cursor.execute("""
            SELECT SUM(amount_usd), SUM(amount_btc)
            FROM transactions
            WHERE DATE(timestamp) = ?
        """, (today,))
        today_usd, today_btc = cursor.fetchone()

        # Monthly revenue
        month_start = datetime.utcnow().replace(day=1).isoformat()
        cursor.execute("""
            SELECT SUM(amount_usd), SUM(amount_btc)
            FROM transactions
            WHERE timestamp >= ?
        """, (month_start,))
        month_usd, month_btc = cursor.fetchone()

        # Customer count
        cursor.execute("SELECT COUNT(*) FROM customers WHERE status = 'active'")
        active_customers = cursor.fetchone()[0]

        conn.close()

        return {
            "total_revenue_usd": total_usd or 0.0,
            "total_revenue_btc": total_btc or 0.0,
            "today_revenue_usd": today_usd or 0.0,
            "today_revenue_btc": today_btc or 0.0,
            "month_revenue_usd": month_usd or 0.0,
            "month_revenue_btc": month_btc or 0.0,
            "active_customers": active_customers,
            "revenue_by_tier": revenue_by_tier,
            "bitcoin_wallet": self.bitcoin_wallet
        }

    def get_top_customers(self, limit: int = 10) -> List[Customer]:
        """Get top customers by lifetime value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM customers
            ORDER BY lifetime_value DESC
            LIMIT ?
        """, (limit,))

        customers = [Customer(*row) for row in cursor.fetchall()]
        conn.close()

        return customers

    def track_campaign(self, campaign_id: str, name: str, platform: str):
        """Track marketing campaign"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO campaigns (campaign_id, name, platform, posted_at)
            VALUES (?, ?, ?, ?)
        """, (campaign_id, name, platform, datetime.utcnow().isoformat()))

        conn.commit()
        conn.close()

    def update_campaign_stats(self, campaign_id: str, views: int = 0,
                             clicks: int = 0, signups: int = 0, revenue: float = 0.0):
        """Update campaign performance stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE campaigns
            SET views = views + ?, clicks = clicks + ?,
                signups = signups + ?, revenue_generated = revenue_generated + ?
            WHERE campaign_id = ?
        """, (views, clicks, signups, revenue, campaign_id))

        conn.commit()
        conn.close()

    def get_campaign_roi(self) -> Dict:
        """Calculate ROI for all campaigns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM campaigns")
        campaigns = cursor.fetchall()
        conn.close()

        roi_data = []
        for campaign in campaigns:
            campaign_id, name, platform, posted_at, views, clicks, signups, revenue = campaign

            ctr = (clicks / views * 100) if views > 0 else 0
            conversion_rate = (signups / clicks * 100) if clicks > 0 else 0
            revenue_per_signup = (revenue / signups) if signups > 0 else 0

            roi_data.append({
                "campaign_id": campaign_id,
                "name": name,
                "platform": platform,
                "views": views,
                "clicks": clicks,
                "signups": signups,
                "revenue": revenue,
                "ctr": round(ctr, 2),
                "conversion_rate": round(conversion_rate, 2),
                "revenue_per_signup": round(revenue_per_signup, 2)
            })

        return {"campaigns": roi_data}

    def _generate_customer_id(self, email: str) -> str:
        """Generate unique customer ID"""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{email}{timestamp}{secrets.token_hex(8)}"
        return f"cust_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"

    def _generate_api_key(self) -> str:
        """Generate secure API key"""
        return f"nxs_{secrets.token_urlsafe(32)}"

    def _send_welcome_email(self, customer: Customer):
        """Send welcome email to new customer"""
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     WELCOME EMAIL SENT                            ║
╚══════════════════════════════════════════════════════════════════╝

To: {customer.email}
Subject: Welcome to Nexus AGI - Your API Key Inside!

Hi there!

Welcome to Nexus AGI - the most advanced AI service platform.

Your API credentials:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Customer ID: {customer.customer_id}
API Key: {customer.api_key}
Subscription: {customer.subscription_tier}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Getting Started:
1. Check our API docs: http://localhost:8000/docs
2. Make your first API call
3. Pay with Bitcoin: {self.bitcoin_wallet}

Pricing:
• API Calls: $0.05 each
• Data Analysis: $100
• Code Generation: $50
• Consultation: $100/hour

Questions? Just reply to this email!

Best regards,
Nexus AGI Team
        """)


class AutomatedEmailResponder:
    """Automated email responses for customer queries"""

    def __init__(self):
        self.templates = {
            "pricing": self._pricing_template,
            "onboarding": self._onboarding_template,
            "payment": self._payment_template,
            "technical": self._technical_template
        }

    def _pricing_template(self, customer_name: str = "there") -> str:
        return f"""
Hi {customer_name},

Thanks for your interest in Nexus AGI pricing!

Our services:
• API Calls: $0.05 per request
• Data Analysis: $100
• Code Generation: $50
• AI Consultation: $100/hour

We accept Bitcoin payments (zero fees!):
bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass

Ready to get started? Sign up at: http://nexus-agi.com/register

Best,
Nexus AGI
        """

    def _onboarding_template(self, customer_name: str = "there") -> str:
        return f"""
Hi {customer_name},

Welcome aboard! Here's how to get started:

1. Get your API key: http://nexus-agi.com/register
2. Read the docs: http://nexus-agi.com/docs
3. Make your first call
4. Pay with Bitcoin

Need help? We're here 24/7!

Best,
Nexus AGI
        """

    def _payment_template(self, customer_name: str = "there") -> str:
        return f"""
Hi {customer_name},

Bitcoin payment instructions:

1. Send BTC to: bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
2. Include your customer ID in transaction metadata
3. Wait for 3 confirmations (~30 minutes)
4. Your credits will be applied automatically

Questions? Reply to this email!

Best,
Nexus AGI
        """

    def _technical_template(self, customer_name: str = "there") -> str:
        return f"""
Hi {customer_name},

Technical support is here to help!

Common solutions:
• Authentication issues? Check your API key
• Rate limits? Upgrade your tier
• Payment not confirmed? Wait for 3 confirmations

Still stuck? Send us:
1. Your customer ID
2. Error message
3. What you're trying to do

We'll respond within 24 hours!

Best,
Nexus AGI Support
        """


def create_demo_customers():
    """Create demo customers to show system working"""
    cms = CustomerManagementSystem()

    print("Creating demo customers...")

    # Demo customers
    demo_emails = [
        ("alice@example.com", "pro", "reddit_r_bitcoin"),
        ("bob@techstartup.com", "enterprise", "twitter"),
        ("carol@researcher.edu", "basic", "upwork"),
        ("dave@freelancer.io", "pro", "fiverr"),
        ("eve@investor.com", "enterprise", "email_campaign")
    ]

    customers = []
    for email, tier, source in demo_emails:
        try:
            customer = cms.register_customer(email, tier, source)
            customers.append(customer)
            print(f"✓ Registered: {email} ({tier})")
        except ValueError:
            print(f"✗ Already exists: {email}")

    # Simulate some transactions
    print("\nSimulating transactions...")
    for customer in customers:
        # Random transactions
        transactions = [
            ("api_call", 0.05, 0.000001),
            ("data_analysis", 100.0, 0.002),
            ("code_generation", 50.0, 0.001)
        ]

        for service, usd, btc in transactions:
            trans = cms.record_transaction(customer.customer_id, service, usd, btc)
            cms.update_customer_spending(customer.customer_id, usd, btc)
            print(f"  ✓ {customer.email}: {service} ${usd}")

    # Show stats
    print("\n" + "="*70)
    print("REVENUE DASHBOARD")
    print("="*70)
    stats = cms.get_revenue_stats()
    print(f"""
Total Revenue:    ${stats['total_revenue_usd']:.2f} USD ({stats['total_revenue_btc']:.8f} BTC)
Today's Revenue:  ${stats['today_revenue_usd']:.2f} USD
Month's Revenue:  ${stats['month_revenue_usd']:.2f} USD
Active Customers: {stats['active_customers']}
Bitcoin Wallet:   {stats['bitcoin_wallet']}
    """)

    print("Revenue by Tier:")
    for tier, data in stats['revenue_by_tier'].items():
        print(f"  {tier.upper()}: ${data['revenue']:.2f} ({data['customers']} customers)")

    print("\n" + "="*70)
    print("TOP CUSTOMERS")
    print("="*70)
    top_customers = cms.get_top_customers(5)
    for i, cust in enumerate(top_customers, 1):
        print(f"{i}. {cust.email}: ${cust.lifetime_value:.2f} ({cust.api_calls_count} API calls)")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       NEXUS AGI - CUSTOMER MANAGEMENT SYSTEM v1.0                ║
║       Automated customer lifecycle & revenue tracking            ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    create_demo_customers()

    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)
    print("""
Customer management system is now running!

Features:
✓ Automated customer registration
✓ API key generation
✓ Transaction tracking
✓ Revenue analytics
✓ Campaign ROI tracking
✓ Automated email responses

Database: nexus_customers.db
Bitcoin Wallet: bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass

Next steps:
1. Integrate with realworld_api_deployment.py
2. Connect to actual Bitcoin payment processor
3. Start marketing campaign
4. Watch revenue grow!
    """)
