     # যে পাসওয়ার্ডটি কপি করেছিলেন
from confluent_kafka import Producer
import json

# আপনার Redpanda সার্ভারের কনফিগারেশন
config = {
    'bootstrap.servers': 'd7hhgff095r0u8rsbnmg.any.ap-south-1.mpx.prd.cloud.redpanda.com:9092',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'SCRAM-SHA-256',
    'sasl.username': 'my_ai_agent',
    'sasl.password': '8eKRFX8qB601N5nmx00ZlGYWueVisk',  # আপনার সেই আসল পাসওয়ার্ডটি এখানে বসান
    'client.id': 'python-producer',
    'request.timeout.ms': 30000,
}

producer = Producer(config)

def delivery_report(err, msg):
    if err is not None:
        print(f'❌ Failed to send: {err}')
    else:
        print(f'✅ Success! Data sent to: {msg.topic()} at partition {msg.partition()}')

print("--- Connecting and Sending data to Redpanda ---")

# ডামি ডেটা (Arijit-এর পিসি থেকে)
data = {
    "order_id": 202601,
    "user": "Arijit",
    "item": "Data Engineering Course",
    "status": "Success"
}

try:
    # এবার আমরা Redpanda-র ডিফল্ট 'hello-world' টপিকে পাঠাচ্ছি
    producer.produce(
        'hello-world', 
        value=json.dumps(data).encode('utf-8'), 
        callback=delivery_report
    )
    
    # ডেটা ডেলিভারি হওয়া পর্যন্ত অপেক্ষা
    producer.flush()
    print("--- Done ---")
    
except Exception as e:
    print(f"Error occurred: {e}")