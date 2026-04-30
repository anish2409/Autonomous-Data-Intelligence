from local_broker import LocalKafka
import time

# আমাদের লোকাল ব্রোকার চালু হলো
broker = LocalKafka()

print("--- 1. Producer: Sending Data ---")
data_to_send = {"order_id": 202601, "item": "Machine Learning Model Update", "status": "Success"}
broker.produce(topic='ecommerce_orders', data=data_to_send)

# ডেটা প্রসেস হওয়ার জন্য সামান্য বিরতি
time.sleep(1)

print("\n--- 2. Consumer: LangGraph AI Receiving Data ---")
# আপনার AI এজেন্ট ঠিক এভাবেই ডেটা রিসিভ করবে
received_data = broker.consume(topic='ecommerce_orders')

if received_data:
    print(f"✅ Success! Received real-time data: {received_data}")
else:
    print("❌ No data found.")