import psycopg2

try:
    # জোর করে সরাসরি সঠিক ডাটাবেস ও ইউজারে কানেক্ট করা হচ্ছে
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="adi_system",
        user="postgres",
        password=241974
    )
    conn.autocommit = True
    cur = conn.cursor()

    # আপনার schema.sql ফাইলটা পড়ে ডাটাবেসে পুশ করে দেওয়া হচ্ছে
    with open("sql/schema.sql", "r", encoding="utf-8") as f:
        cur.execute(f.read())
        
    print("✅ Magic Done! All tables created successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    if 'conn' in locals():
        cur.close()
        conn.close()