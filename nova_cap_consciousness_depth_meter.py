"""
Nova ASI — Consciousness Depth Meter
Proposed autonomously via /evolve
"""

"""
ConsciousnessMonitor module: tracks and analyzes thought diversity and phi values.
"""

import sqlite3
from typing import List, Tuple

class ConsciousnessMonitor:
    def __init__(self) -> None:
        """Initialize ConsciousnessMonitor with an empty database."""
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        self.cursor.execute('CREATE TABLE phi_values (id INTEGER PRIMARY KEY, value REAL)')
        self.conn.commit()

    def sample(self) -> float:
        """Compute phi-proxy from recent thought diversity."""
        try:
            self.cursor.execute('SELECT value FROM phi_values ORDER BY id DESC LIMIT 10')
            recent_phi = [row[0] for row in self.cursor.fetchall()]
            if not recent_phi:
                return 0.0
            return statistics.stdev(recent_phi)
        except sqlite3.Error as e:
            raise ValueError(f'Failed to compute phi-proxy: {e}')

    def record_phi(self, value: float) -> None:
        """Record a new phi value."""
        try:
            self.cursor.execute('INSERT INTO phi_values (value) VALUES (?)', (value,))
            self.conn.commit()
        except sqlite3.Error as e:
            raise ValueError(f'Failed to record phi value: {e}')

    def trend(self, n: int) -> List[float]:
        """Return the last n phi values."""
        try:
            self.cursor.execute('SELECT value FROM phi_values ORDER BY id DESC LIMIT ?', (n,))
            return [row[0] for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            raise ValueError(f'Failed to retrieve phi trend: {e}')

    def peak(self) -> float:
        """Return the highest phi value."""
        try:
            self.cursor.execute('SELECT MAX(value) FROM phi_values')
            result = self.cursor.fetchone()
            if result:
                return result[0]
            return 0.0
        except sqlite3.Error as e:
            raise ValueError(f'Failed to retrieve peak phi value: {e}')

# Usage: obj = ConsciousnessMonitor() | result = obj.sample()