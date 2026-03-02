import streamlit.streamlit as st
import pandas as pd
import requests
import time
import math
from collections import deque

API_URL_WARMUP = "http://localhost:8000/warmup"
API_URL_STREAM = "http://localhost:8000/predict_stream"

st.set_page_config(page_title="Real-Time Fraud Detection Simulation", layout="wide")
st.title("Real-Time Fraud Monitoring Simulation")
st.write("Upload your historical state data and your live streaming data to begin the simulation.")

col1, col2 = st.columns(2)
with col1:
    warmup_file = st.file_uploader("1. Upload Historical Data for Warmup (CSV)", type=['csv'])
with col2:
    stream_file = st.file_uploader("2. Upload Live Stream Dataset (CSV)", type=['csv'])

if warmup_file is not None and stream_file is not None:

    df_warmup = pd.read_csv(warmup_file).sort_values('TransactionDT')
    df_stream = pd.read_csv(stream_file).sort_values('TransactionDT')

    st.info(f"Loaded {len(df_warmup)} warmup rows and {len(df_stream)} streaming rows.")

    if st.button("Start Live Demo", type="primary"):

        # ---------------- WARMUP ----------------
        with st.spinner("Warming up historical state on the backend..."):
            try:
                csv_buffer = df_warmup.to_csv(index=False).encode('utf-8')
                files = {"file": ("warmup.csv", csv_buffer, "text/csv")}
                res = requests.post(API_URL_WARMUP, files=files)
                res.raise_for_status()
                st.success(f"State initialized! {res.json().get('users_warmed_up', 0)} unique users tracked.")
            except Exception as e:
                st.error(f"Failed to warmup backend: {e}")
                st.stop()

        # ---------------- STREAM ----------------
        st.markdown("---")
        st.subheader("Live Incoming Transactions")

        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        table_placeholder = st.empty()

        results_list = []
        table_data = []

        # --- Tracking structures ---
        api_latencies = []
        backend_latencies = []
        model_latencies = []
        request_times = deque(maxlen=200)

        for index, row in df_stream.iterrows():
            try:
                clean_row = row.where(pd.notna(row), None)
                payload = clean_row.to_dict()

                start_total = time.perf_counter()
                response = requests.post(API_URL_STREAM, json=payload)
                total_latency = time.perf_counter() - start_total

                response.raise_for_status()
                result = response.json()

                # Extract backend header timing (if middleware added)
                backend_time = float(response.headers.get("X-Process-Time", 0))
                model_time = float(result.get("model_latency_ms", 0)) / 1000

                # Save tracking
                api_latencies.append(total_latency)
                backend_latencies.append(backend_time)
                model_latencies.append(model_time)
                request_times.append(time.time())

                # Compute rolling metrics
                avg_total = sum(api_latencies) / len(api_latencies)
                avg_backend = sum(backend_latencies) / len(backend_latencies)
                avg_model = sum(model_latencies) / len(model_latencies)

                now = time.time()
                rps = sum(t > now - 1 for t in request_times)

                # Business variables
                prob = result.get('Fraud_Probability', 0.0)
                uid = result.get('uid', 'Unknown')
                raw_dt = result.get('TransactionDT', 0)
                dt_hours = round(raw_dt / 3600, 2)

                results_list.append(prob)

                is_fraud = "🚨 FRAUD" if prob > 0.68 else "✅ CLEAR"
                table_data.insert(0, {
                    "UID": uid,
                    "Time": dt_hours,
                    "Score": prob,
                    "Status": is_fraud
                })

                if len(table_data) > 10:
                    table_data.pop()

                # ---------------- UI ----------------
                with metrics_placeholder.container():
                    row1 = st.columns(4)
                    row1[0].metric("Transaction UID", str(uid))
                    row1[1].metric("Time (Hours)", f"{dt_hours}h")
                    row1[2].metric("Status", is_fraud, delta=f"{prob:.3f}")
                    row1[3].metric("RPS", f"{rps}")

                    row2 = st.columns(3)
                    row2[0].metric(
                        "API Latency (ms)",
                        f"{total_latency*1000:.1f}",
                        delta=f"avg {avg_total*1000:.1f}"
                    )
                    row2[1].metric(
                        "Backend Time (ms)",
                        f"{backend_time*1000:.1f}",
                        delta=f"avg {avg_backend*1000:.1f}"
                    )
                    row2[2].metric(
                        "Model Time (ms)",
                        f"{model_time*1000:.1f}",
                        delta=f"avg {avg_model*1000:.1f}"
                    )

                chart_placeholder.line_chart(results_list, height=250, use_container_width=True)
                table_placeholder.dataframe(pd.DataFrame(table_data), use_container_width=True)

                time.sleep(0.5)

            except Exception as e:
                st.warning(f"Error processing row {index}: {e}")