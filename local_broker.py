import sqlite3
import json

class LocalKafka:
    def __init__(self, db_name='local_cluster.db'):
        # SQLite ব্যবহার করে একটি লোকাল ডেটাবেস তৈরি হচ্ছে যা ব্রোকার হিসেবে কাজ করবে
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS streams 
                             (id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, data TEXT, read_status INTEGER DEFAULT 0)''')
        self.conn.commit()

    def produce(self, topic, data):
        # ডেটা পুশ করা হচ্ছে (Producer)
        self.conn.execute('INSERT INTO streams (topic, data) VALUES (?, ?)', (topic, json.dumps(data)))
        self.conn.commit()
        print(f"✅ [Broker] Data streamed to topic: '{topic}'")

    def consume(self, topic):
        # ডেটা রিসিভ করা হচ্ছে (Consumer)
        cur = self.conn.cursor()
        cur.execute('SELECT id, data FROM streams WHERE topic=? AND read_status=0 ORDER BY id ASC LIMIT 1', (topic,))
        row = cur.fetchone()
        if row:
            msg_id, data = row
            # ডেটা একবার পড়া হয়ে গেলে মার্ক করে দেওয়া হচ্ছে
            self.conn.execute('UPDATE streams SET read_status=1 WHERE id=?', (msg_id,))
            self.conn.commit()
            return json.loads(data)
        return None