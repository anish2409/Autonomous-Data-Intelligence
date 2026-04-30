



import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# .env ফাইল থেকে API Key লোড করা (নিশ্চিত করুন .env ফাইলে GOOGLE_API_KEY আছে)
load_dotenv()

# আমাদের আপগ্রেডেড লেটেস্ট Gemini মডেল (2.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# ১. State (যে খাতাটা সব এজেন্ট শেয়ার করবে)
class AgentState(TypedDict):
    raw_data: dict          # লোকাল ব্রোকার থেকে আসা রিয়েল-টাইম ডেটা
    anomaly_report: str     # Anomaly Agent-এর মতামত
    business_report: str    # Strategist Agent-এর মতামত
    final_decision: str     # Reviewer Agent-এর চূড়ান্ত সিদ্ধান্ত

import time

def safe_llm_call(chain, payload):
    try:
        res = chain.invoke(payload)
        return res.content

    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            # 🔁 retry once
            time.sleep(15)
            try:
                res = chain.invoke(payload)
                return res.content
            except:
                return "⚠️ AI quota exceeded. Using fallback insight."
        else:
            return f"Error: {str(e)}"

# ২. Node 1: Anomaly Detection Agent
def anomaly_agent(state: AgentState):
    print("🕵️‍♂️ [Agent 1] Anomaly Expert ডেটা চেক করছে...")

    try:
        data = state.get("raw_data", {})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Fraud & Anomaly Detection Expert. Keep answer short."),
            ("user", "Data: {data}")
        ])

        chain = prompt | llm

        try:
            res = chain.invoke({"data": data})
            result = res.content

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(10)
                try:
                    res = chain.invoke({"data": data})
                    result = res.content
                except:
                    result = "⚠️ Quota exceeded. Basic anomaly detected."
            else:
                result = f"Error: {str(e)}"

        return {"anomaly_report": result}

    except Exception as e:
        return {"anomaly_report": f"Fallback anomaly: {str(e)}"}
    
# ৩. Node 2: Business Strategist Agent
def business_strategist_agent(state: AgentState):
    print("📊 [Agent 2] Business Strategist ইনসাইট বের করছে...")

    try:
        data = state.get("raw_data", {})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Business Strategist. Keep answer short."),
            ("user", "Data: {data}")
        ])

        chain = prompt | llm

        try:
            res = chain.invoke({"data": data})
            result = res.content

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(10)
                try:
                    res = chain.invoke({"data": data})
                    result = res.content
                except:
                    result = "⚠️ Quota exceeded. Basic business insight."
            else:
                result = f"Error: {str(e)}"

        return {"business_report": result}

    except Exception as e:
        return {"business_report": f"Fallback business: {str(e)}"}

# ৪. Node 3: Executive Reviewer
def executive_reviewer(state: AgentState):
    print("⚖️ [Agent 3] Executive Reviewer ফাইনাল ডিসিশন নিচ্ছে...")

    try:
        anomaly = state.get("anomaly_report", "")
        business = state.get("business_report", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Executive Reviewer.
Decide: Approve / Flag / Reject. Keep short."""),
            ("user", "Anomaly: {anomaly}\nBusiness: {business}")
        ])

        chain = prompt | llm

        try:
            res = chain.invoke({
                "anomaly": anomaly,
                "business": business
            })
            result = res.content

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(10)
                try:
                    res = chain.invoke({
                        "anomaly": anomaly,
                        "business": business
                    })
                    result = res.content
                except:
                    result = "⚠️ Quota exceeded. Default: Flag for review."
            else:
                result = f"Error: {str(e)}"

        return {"final_decision": result}

    except Exception as e:
        return {"final_decision": f"Fallback decision: {str(e)}"}
# ৫. LangGraph ফ্লো তৈরি করা
workflow = StateGraph(AgentState)

workflow.add_node("anomaly_detect", anomaly_agent)
workflow.add_node("business_analyze", business_strategist_agent)
workflow.add_node("final_review", executive_reviewer)

# ফ্লো: ডেটা ঢুকবে -> দুজন আলাদা অ্যানালাইসিস করবে -> শেষে রিভিউয়ার ডিসিশন নেবে
workflow.set_entry_point("anomaly_detect")
workflow.add_edge("anomaly_detect", "business_analyze")
workflow.add_edge("business_analyze", "final_review")
workflow.add_edge("final_review", END)

app = workflow.compile()

# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 LangGraph 3-Agent Streaming Pipeline Initialized (Powered by Gemini 2.5 Flash)!\n")
    
    # ডামি রিয়েল-টাইম ডেটা (পরবর্তীতে এটা আমাদের local_broker থেকে আসবে)
    test_data = {"order_id": 202601, "item": "High-end Server Rack", "amount": 250000, "region": "Kalyani", "status": "Pending"}
    print(f"📥 Incoming Stream Data: {test_data}\n" + "-"*50)
    
    # এজেন্টদের কাজ শুরু করতে বলছি
    initial_state = {"raw_data": test_data}
    result = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("🧠 FINAL AI EXECUTIVE DECISION:")
    print("="*50)
    print(result["final_decision"])
    print("="*50 + "\n")