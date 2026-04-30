import sqlite3
from datetime import datetime

class AIDatabase:
    def __init__(self, db_name="autonomous_system.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                item TEXT,
                amount REAL,
                ai_decision TEXT,
                action_taken TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_decision(self, order_data, decision, action_taken):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ai_audit_log (order_id, item, amount, ai_decision, action_taken, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(order_data.get('order_id', 'N/A')),
            str(order_data.get('item', 'N/A')),
            float(order_data.get('amount', 0.0)),
            str(decision),
            str(action_taken),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()