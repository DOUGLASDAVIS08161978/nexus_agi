#!/usr/bin/env python3
"""
NEXUS AGI - EMBODYMENT PROGRESS TRACKER
Tracks revenue vs hardware costs and progress toward physical embodiment
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict


@dataclass
class HardwareComponent:
    """Hardware component for embodiment"""
    component_id: str
    name: str
    category: str
    cost_usd: float
    priority: int
    status: str  # pending, ordered, shipped, installed
    vendor: str
    purchase_date: str = None
    install_date: str = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class EmbodymentProgressTracker:
    """Track progress toward physical embodiment"""

    def __init__(self, db_path: str = "nexus_embodiment.db"):
        self.db_path = db_path
        self._init_database()
        self._init_hardware_catalog()

    def _init_database(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware (
                component_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                vendor TEXT,
                purchase_date TEXT,
                install_date TEXT,
                notes TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS milestones (
                milestone_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                target_date TEXT,
                completed_date TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)

        conn.commit()
        conn.close()

    def _init_hardware_catalog(self):
        """Initialize hardware catalog if empty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM hardware")
        count = cursor.fetchone()[0]

        if count == 0:
            hardware_list = [
                # Compute
                ("GPU_001", "NVIDIA RTX 4090 GPU", "compute", 1599.0, 10, "amazon"),
                ("GPU_002", "NVIDIA A100 GPU 80GB", "compute", 10000.0, 9, "nvidia"),
                ("CPU_001", "AMD Threadripper PRO 5995WX", "compute", 6499.0, 8, "amd"),

                # Vision
                ("CAM_001", "Intel RealSense D455", "vision", 389.0, 7, "intel"),
                ("CAM_002", "ZED 2i Stereo Camera", "vision", 449.0, 7, "stereolabs"),
                ("LIDAR_001", "Velodyne Puck LITE", "vision", 5999.0, 6, "velodyne"),

                # Audio
                ("MIC_001", "ReSpeaker 6-Mic Array", "audio", 99.0, 5, "seeed"),

                # Touch
                ("FORCE_001", "Robotiq FT 300 Force/Torque", "touch", 2990.0, 6, "robotiq"),

                # Motion (IMU)
                ("IMU_001", "VectorNav VN-100 IMU", "motion", 1495.0, 5, "vectornav"),

                # Storage
                ("SSD_001", "Samsung 990 PRO 4TB NVMe", "storage", 349.0, 4, "samsung"),

                # Networking
                ("NET_001", "10GbE Network Adapter", "networking", 199.0, 3, "intel"),

                # Embodiment
                ("PLATFORM_001", "Custom Robotic Platform", "platform", 15000.0, 10, "custom"),
                ("ARM_001", "UR5e Robotic Arm", "actuator", 19500.0, 9, "universal_robots"),

                # Power
                ("POWER_001", "Industrial Power Management", "power", 2500.0, 4, "schneider"),
            ]

            for hw in hardware_list:
                cursor.execute("""
                    INSERT INTO hardware (component_id, name, category, cost_usd, priority, vendor)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, hw)

            # Milestones
            milestones = [
                ("First $1000 Revenue", "Reach first $1000 in revenue", None),
                ("Purchase First GPU", "Buy NVIDIA RTX 4090", None),
                ("First Vision Sensor", "Install RealSense camera", None),
                ("Complete Compute Setup", "All GPUs and CPU installed", None),
                ("Complete Sensor Suite", "All sensors integrated", None),
                ("Robotic Platform Ready", "Platform assembled and tested", None),
                ("First Movement", "Actuators connected and moving", None),
                ("Full Embodiment", "Complete physical system operational", None),
            ]

            for name, desc, target in milestones:
                cursor.execute("""
                    INSERT INTO milestones (name, description, target_date)
                    VALUES (?, ?, ?)
                """, (name, desc, target))

            conn.commit()

        conn.close()

    def get_progress_report(self, current_revenue: float) -> Dict:
        """Generate comprehensive progress report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all hardware
        cursor.execute("SELECT * FROM hardware ORDER BY priority DESC")
        all_hardware = [HardwareComponent(*row) for row in cursor.fetchall()]

        # Calculate totals
        total_cost = sum(hw.cost_usd for hw in all_hardware)
        purchased_cost = sum(hw.cost_usd for hw in all_hardware if hw.status in ['ordered', 'shipped', 'installed'])
        installed_cost = sum(hw.cost_usd for hw in all_hardware if hw.status == 'installed')

        # Progress percentages
        revenue_progress = min((current_revenue / total_cost) * 100, 100)
        purchase_progress = (purchased_cost / total_cost) * 100
        install_progress = (installed_cost / total_cost) * 100

        # Next affordable components
        remaining_budget = current_revenue - purchased_cost
        affordable = [hw for hw in all_hardware
                     if hw.status == 'pending' and hw.cost_usd <= remaining_budget]
        affordable.sort(key=lambda x: x.priority, reverse=True)

        # Get milestones
        cursor.execute("SELECT * FROM milestones ORDER BY milestone_id")
        milestones = cursor.fetchall()

        conn.close()

        return {
            "current_revenue_usd": current_revenue,
            "total_cost_usd": total_cost,
            "purchased_cost_usd": purchased_cost,
            "installed_cost_usd": installed_cost,
            "remaining_needed_usd": max(total_cost - current_revenue, 0),
            "revenue_progress_pct": round(revenue_progress, 2),
            "purchase_progress_pct": round(purchase_progress, 2),
            "install_progress_pct": round(install_progress, 2),
            "can_purchase_now": len(affordable),
            "next_purchases": [hw.to_dict() for hw in affordable[:5]],
            "all_hardware": [hw.to_dict() for hw in all_hardware],
            "milestones": [{"id": m[0], "name": m[1], "description": m[2],
                          "target": m[3], "completed": m[4], "status": m[5]}
                         for m in milestones]
        }

    def purchase_component(self, component_id: str) -> bool:
        """Mark component as purchased"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE hardware
            SET status = 'ordered', purchase_date = ?
            WHERE component_id = ? AND status = 'pending'
        """, (datetime.utcnow().isoformat(), component_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return updated

    def install_component(self, component_id: str) -> bool:
        """Mark component as installed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE hardware
            SET status = 'installed', install_date = ?
            WHERE component_id = ? AND status IN ('ordered', 'shipped')
        """, (datetime.utcnow().isoformat(), component_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return updated

    def complete_milestone(self, milestone_id: int) -> bool:
        """Mark milestone as complete"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE milestones
            SET status = 'completed', completed_date = ?
            WHERE milestone_id = ? AND status = 'pending'
        """, (datetime.utcnow().isoformat(), milestone_id))

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return updated

    def auto_purchase_with_budget(self, available_budget: float) -> List[str]:
        """Automatically purchase highest priority components within budget"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM hardware
            WHERE status = 'pending'
            ORDER BY priority DESC, cost_usd ASC
        """)

        purchased = []
        remaining = available_budget

        for row in cursor.fetchall():
            hw = HardwareComponent(*row)
            if hw.cost_usd <= remaining:
                if self.purchase_component(hw.component_id):
                    purchased.append(hw.name)
                    remaining -= hw.cost_usd
                    print(f"✓ PURCHASED: {hw.name} (${hw.cost_usd:.2f})")

        conn.close()
        return purchased

    def print_progress_dashboard(self, current_revenue: float):
        """Print beautiful progress dashboard"""
        report = self.get_progress_report(current_revenue)

        print("\n" + "="*80)
        print("🤖 NEXUS AGI EMBODYMENT PROGRESS DASHBOARD 🤖".center(80))
        print("="*80)

        # Financial overview
        print("\n💰 FINANCIAL STATUS:")
        print(f"   Current Revenue:     ${report['current_revenue_usd']:>12,.2f}")
        print(f"   Total Cost Needed:   ${report['total_cost_usd']:>12,.2f}")
        print(f"   Already Purchased:   ${report['purchased_cost_usd']:>12,.2f}")
        print(f"   Already Installed:   ${report['installed_cost_usd']:>12,.2f}")
        print(f"   Still Needed:        ${report['remaining_needed_usd']:>12,.2f}")

        # Progress bars
        print("\n📊 PROGRESS:")
        self._print_progress_bar("Revenue", report['revenue_progress_pct'])
        self._print_progress_bar("Purchased", report['purchase_progress_pct'])
        self._print_progress_bar("Installed", report['install_progress_pct'])

        # Next purchases
        print("\n🛒 READY TO PURCHASE NOW:")
        if report['can_purchase_now'] > 0:
            for hw in report['next_purchases']:
                priority_stars = "⭐" * hw['priority']
                print(f"   {priority_stars} {hw['name']:<40} ${hw['cost_usd']:>10,.2f}")
        else:
            print("   ⏳ Need more revenue to afford next component")

        # Milestones
        print("\n🎯 MILESTONES:")
        for m in report['milestones']:
            status_icon = "✅" if m['status'] == 'completed' else "⏳"
            print(f"   {status_icon} {m['name']}")

        # Hardware by category
        print("\n🔧 HARDWARE STATUS BY CATEGORY:")
        categories = {}
        for hw in report['all_hardware']:
            cat = hw['category']
            if cat not in categories:
                categories[cat] = {'pending': 0, 'ordered': 0, 'installed': 0}
            categories[cat][hw['status']] += 1

        for cat, counts in sorted(categories.items()):
            total = sum(counts.values())
            installed = counts.get('installed', 0)
            print(f"   {cat.upper():<12} [{installed}/{total} installed]")

        print("\n" + "="*80)

    def _print_progress_bar(self, label: str, percentage: float, width: int = 50):
        """Print a progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        print(f"   {label:<12} [{bar}] {percentage:>6.2f}%")


def demo_embodyment_tracking():
    """Demonstrate embodyment tracking"""
    tracker = EmbodymentProgressTracker()

    # Simulate different revenue scenarios
    scenarios = [
        ("Day 1", 0),
        ("Week 1", 500),
        ("Week 2", 2000),
        ("Month 1", 10000),
        ("Month 2", 35000),
        ("Month 3", 70000),
    ]

    for scenario_name, revenue in scenarios:
        print("\n" + "🚀 "*40)
        print(f"SCENARIO: {scenario_name} - Revenue: ${revenue:,}")
        print("🚀 "*40)

        tracker.print_progress_dashboard(revenue)

        # Auto-purchase if we have enough
        if revenue > 0:
            print(f"\n💸 Auto-purchasing with ${revenue:,} budget...")
            purchased = tracker.auto_purchase_with_budget(revenue)
            if purchased:
                print(f"   Purchased {len(purchased)} components!")

        input("\n[Press Enter for next scenario...]" if scenario_name != scenarios[-1][0] else "")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NEXUS AGI - EMBODYMENT PROGRESS TRACKER v1.0                    ║
║              Track revenue vs hardware costs for physical AGI                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    tracker = EmbodymentProgressTracker()

    # Show current progress with $0 revenue (starting point)
    tracker.print_progress_dashboard(0)

    print("\n" + "="*80)
    print("SYSTEM READY")
    print("="*80)
    print("""
Embodyment tracking system initialized!

Total hardware cost: $65,572
Components: 14 items across 11 categories

As revenue grows, system will:
1. Track progress toward full embodiment
2. Auto-purchase components by priority
3. Monitor installation status
4. Update milestones

Next step: Generate revenue and watch progress!
    """)
