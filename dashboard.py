import streamlit as st
import requests
import os
import pandas as pd
import uuid
from datetime import datetime

# Configure page
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

st.title("UPI Fraud Detection Dashboard")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", API_URL)
    
    st.markdown("---")
    st.header("Test Data")
    if st.button("Load Normal Sample"):
        st.session_state['sender_id'] = "user_normal"
        st.session_state['receiver_id'] = "merchant_123"
        st.session_state['amount'] = 150.0
        st.session_state['city'] = "Mumbai"
        st.session_state['device'] = "Android"
        
    if st.button("Load Fraud Sample"):
        st.session_state['sender_id'] = "user_normal"
        st.session_state['receiver_id'] = "unknown_hacker_999"
        st.session_state['amount'] = 250000.0
        st.session_state['city'] = "Moscow"
        st.session_state['device'] = "UnknownDevice"

# Initialize session state for form
for key, default in [('sender_id', 'user_123'), ('receiver_id', 'user_456'), 
                     ('amount', 500.0), ('city', 'Delhi'), ('device', 'iOS')]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Score Transaction")
    with st.form("txn_form"):
        sender_id = st.text_input("Sender ID", st.session_state['sender_id'])
        receiver_id = st.text_input("Receiver ID", st.session_state['receiver_id'])
        amount = st.number_input("Amount (INR)", value=float(st.session_state['amount']))
        city = st.text_input("City", st.session_state['city'])
        device = st.text_input("Device", st.session_state['device'])
        
        submitted = st.form_submit_button("Analyze Transaction")
        
    if submitted:
        payload = {
            "transaction_id": str(uuid.uuid4()),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "amount": amount,
            "city": city,
            "device": device,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            resp = requests.post(f"{api_url}/score", json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            st.subheader("Result")
            risk = data['risk']
            score = data['risk_score']
            
            color = "green"
            if risk == "REVIEW": color = "orange"
            if risk == "HIGH": color = "red"
            
            st.markdown(f"### Risk: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            st.progress(float(score))
            st.write(f"**Score:** {score:.3f}")
            
            if data['reasons']:
                st.write("**Reasons:**")
                for r in data['reasons']:
                    st.write(f"- {r}")
            else:
                st.write("No anomalous patterns detected.")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API: {e}")

with col2:
    st.header("Recent Alerts")
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Refresh Alerts"):
            pass # Just triggers rerun
    with col_b:
        risk_filter = st.selectbox("Risk Filter", ["HIGH", "REVIEW"], index=0)
    
    try:
        alerts_resp = requests.get(f"{api_url}/alerts?limit=100&risk={risk_filter}", timeout=5)
        if alerts_resp.status_code == 200:
            alerts = alerts_resp.json()
            if alerts:
                df_alerts = pd.DataFrame(alerts)
                
                # Show dataframe
                st.dataframe(df_alerts[['timestamp', 'sender_id', 'amount', 'risk', 'risk_score', 'reasons']], use_container_width=True)
                
                # Charts
                df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
                df_alerts['hour'] = df_alerts['timestamp'].dt.hour
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.write("Alerts by Hour")
                    st.bar_chart(df_alerts['hour'].value_counts().sort_index())
                    
                with chart_col2:
                    st.write("Score Distribution")
                    st.bar_chart(df_alerts['risk_score'].value_counts(bins=10).sort_index())
            else:
                st.info(f"No {risk_filter} alerts found.")
    except requests.exceptions.RequestException:
        st.error("API is down or unreachable. Please start the backend server.")
