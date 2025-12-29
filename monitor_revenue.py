#!/usr/bin/env python3
"""
Nexus AGI Revenue Monitoring Dashboard
Real-time tracking of revenue, customers, and API usage
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List
import time
import json

try:
    from dotenv import load_dotenv
    import stripe
    import requests
except ImportError:
    print("Installing dependencies...")
    os.system("pip install python-dotenv stripe requests -q")
    from dotenv import load_dotenv
    import stripe
    import requests

load_dotenv()


class RevenueMonitor:
    def __init__(self):
        self.stripe_key = os.getenv('STRIPE_SECRET_KEY')
        self.coinbase_key = os.getenv('COINBASE_COMMERCE_API_KEY')
        self.api_url = os.getenv('APP_URL', 'http://localhost:8000')

        if self.stripe_key and self.stripe_key.startswith('sk_'):
            stripe.api_key = self.stripe_key
            self.stripe_enabled = True
        else:
            self.stripe_enabled = False

        self.coinbase_enabled = bool(self.coinbase_key)

    def get_stripe_revenue(self, days=30) -> Dict:
        """Get Stripe revenue for the last N days"""
        if not self.stripe_enabled:
            return {"error": "Stripe not configured"}

        try:
            # Get balance
            balance = stripe.Balance.retrieve()
            available = balance.available[0].amount / 100 if balance.available else 0
            pending = balance.pending[0].amount / 100 if balance.pending else 0

            # Get charges from last N days
            start_date = int((datetime.now() - timedelta(days=days)).timestamp())
            charges = stripe.Charge.list(created={'gte': start_date}, limit=100)

            total_revenue = sum(c.amount / 100 for c in charges.data if c.paid)
            successful_charges = len([c for c in charges.data if c.paid])

            # Get customers
            customers = stripe.Customer.list(created={'gte': start_date}, limit=100)
            new_customers = len(customers.data)

            # Get subscriptions
            subscriptions = stripe.Subscription.list(
                created={'gte': start_date},
                limit=100
            )
            active_subscriptions = len([s for s in subscriptions.data if s.status == 'active'])

            return {
                'available_balance': available,
                'pending_balance': pending,
                'total_revenue': total_revenue,
                'successful_charges': successful_charges,
                'new_customers': new_customers,
                'active_subscriptions': active_subscriptions,
                'charges': charges.data[:10]  # Last 10 charges
            }
        except Exception as e:
            return {"error": str(e)}

    def get_coinbase_revenue(self) -> Dict:
        """Get Coinbase Commerce revenue"""
        if not self.coinbase_enabled:
            return {"error": "Coinbase not configured"}

        try:
            headers = {
                'X-CC-Api-Key': self.coinbase_key,
                'X-CC-Version': '2018-03-22'
            }

            # Get charges
            response = requests.get(
                'https://api.commerce.coinbase.com/charges',
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                charges = data.get('data', [])

                completed = [c for c in charges if c.get('timeline', [{}])[-1].get('status') == 'COMPLETED']
                total_revenue = sum(
                    float(c.get('pricing', {}).get('local', {}).get('amount', 0))
                    for c in completed
                )

                return {
                    'total_charges': len(charges),
                    'completed_charges': len(completed),
                    'total_revenue': total_revenue,
                    'recent_charges': charges[:5]
                }
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def get_api_stats(self) -> Dict:
        """Get API usage statistics"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                return {
                    'status': 'online',
                    'health': health
                }
            else:
                return {'status': 'error', 'code': response.status_code}
        except Exception as e:
            return {'status': 'offline', 'error': str(e)}

    def display_dashboard(self):
        """Display beautiful ASCII dashboard"""
        stripe_data = self.get_stripe_revenue()
        coinbase_data = self.get_coinbase_revenue()
        api_data = self.get_api_stats()

        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Header
        print("\033[95m")
        print("╔═══════════════════════════════════════════════════════════════════════╗")
        print("║                                                                       ║")
        print("║              💎 NEXUS AGI REVENUE DASHBOARD 💎                        ║")
        print("║                                                                       ║")
        print("╚═══════════════════════════════════════════════════════════════════════╝")
        print("\033[0m")

        print(f"\n📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Stripe Revenue
        print("\033[96m" + "="*75 + "\033[0m")
        print("\033[96m💳 STRIPE REVENUE (Last 30 Days)\033[0m")
        print("\033[96m" + "="*75 + "\033[0m")

        if 'error' not in stripe_data:
            print(f"\n💰 Total Revenue:          \033[92m${stripe_data['total_revenue']:,.2f}\033[0m")
            print(f"💵 Available Balance:      \033[92m${stripe_data['available_balance']:,.2f}\033[0m")
            print(f"⏳ Pending Balance:        \033[93m${stripe_data['pending_balance']:,.2f}\033[0m")
            print(f"\n👥 New Customers:          \033[94m{stripe_data['new_customers']}\033[0m")
            print(f"📊 Active Subscriptions:   \033[94m{stripe_data['active_subscriptions']}\033[0m")
            print(f"✅ Successful Charges:     \033[94m{stripe_data['successful_charges']}\033[0m")

            if stripe_data.get('charges'):
                print("\n\033[95m📋 Recent Charges:\033[0m")
                for charge in stripe_data['charges'][:5]:
                    amount = charge.amount / 100
                    status = '✅' if charge.paid else '❌'
                    date = datetime.fromtimestamp(charge.created).strftime('%Y-%m-%d %H:%M')
                    print(f"  {status} ${amount:>8.2f} - {date} - {charge.description or 'N/A'}")
        else:
            print(f"\n\033[91m❌ Error: {stripe_data['error']}\033[0m")
            print("\033[93m⚠  Configure STRIPE_SECRET_KEY in .env to enable Stripe tracking\033[0m")

        # Coinbase Revenue
        print("\n" + "\033[96m" + "="*75 + "\033[0m")
        print("\033[96m₿ CRYPTOCURRENCY REVENUE (Coinbase Commerce)\033[0m")
        print("\033[96m" + "="*75 + "\033[0m")

        if 'error' not in coinbase_data:
            print(f"\n💰 Total Revenue:          \033[92m${coinbase_data['total_revenue']:,.2f}\033[0m")
            print(f"📊 Total Charges:          \033[94m{coinbase_data['total_charges']}\033[0m")
            print(f"✅ Completed Charges:      \033[94m{coinbase_data['completed_charges']}\033[0m")
        else:
            print(f"\n\033[91m❌ Error: {coinbase_data['error']}\033[0m")
            print("\033[93m⚠  Configure COINBASE_COMMERCE_API_KEY in .env to enable crypto tracking\033[0m")

        # API Status
        print("\n" + "\033[96m" + "="*75 + "\033[0m")
        print("\033[96m🔧 API STATUS\033[0m")
        print("\033[96m" + "="*75 + "\033[0m")

        if api_data['status'] == 'online':
            print(f"\n✅ Status: \033[92mONLINE\033[0m")
            print(f"🌐 URL: {self.api_url}")
        else:
            print(f"\n❌ Status: \033[91mOFFLINE\033[0m")
            print(f"⚠  Error: {api_data.get('error', 'Unknown')}")

        # Summary
        print("\n" + "\033[95m" + "="*75 + "\033[0m")
        print("\033[95m📊 SUMMARY\033[0m")
        print("\033[95m" + "="*75 + "\033[0m")

        total_revenue = 0
        if 'error' not in stripe_data:
            total_revenue += stripe_data['total_revenue']
        if 'error' not in coinbase_data:
            total_revenue += coinbase_data['total_revenue']

        print(f"\n💎 Total Revenue (30 days): \033[92m\033[1m${total_revenue:,.2f}\033[0m")

        # Projections
        daily_avg = total_revenue / 30
        monthly_projection = daily_avg * 30
        yearly_projection = monthly_projection * 12

        print(f"📈 Daily Average:          \033[94m${daily_avg:,.2f}\033[0m")
        print(f"📅 Monthly Projection:     \033[94m${monthly_projection:,.2f}\033[0m")
        print(f"🎯 Yearly Projection:      \033[94m${yearly_projection:,.2f}\033[0m")

        # Footer
        print("\n" + "\033[95m" + "="*75 + "\033[0m")
        print("\033[93m💡 Tip: Keep this dashboard running to monitor revenue in real-time!\033[0m")
        print("\033[93m🔄 Auto-refreshes every 60 seconds. Press Ctrl+C to exit.\033[0m")
        print("\033[95m" + "="*75 + "\033[0m\n")


def main():
    """Main function"""
    monitor = RevenueMonitor()

    print("\033[92m")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  🚀 Starting Nexus AGI Revenue Monitor...                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print("\033[0m\n")

    # Check configuration
    if not monitor.stripe_enabled and not monitor.coinbase_enabled:
        print("\033[91m❌ ERROR: No payment providers configured!\033[0m")
        print("\n\033[93mPlease configure at least one payment provider in .env:\033[0m")
        print("  • STRIPE_SECRET_KEY=sk_live_...")
        print("  • COINBASE_COMMERCE_API_KEY=...")
        print("\nSee ACTIVATE_PAYMENTS.md for setup instructions.")
        sys.exit(1)

    try:
        # Continuous monitoring
        while True:
            monitor.display_dashboard()
            time.sleep(60)  # Refresh every 60 seconds
    except KeyboardInterrupt:
        print("\n\n\033[92m✅ Revenue monitor stopped. Keep earning! 💰\033[0m\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
