import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import threading
from collections import deque
from datetime import datetime

API_URL_WARMUP = "https://cnn-gru-v2-829168846186.asia-southeast2.run.app/warmup"
API_URL_STREAM = "https://cnn-gru-v2-829168846186.asia-southeast2.run.app/predict_stream"

BASE_RATE_RPS    = 2.0
BURST_PROB       = 0.08
BURST_SIZE_RANGE = (5, 20)
BURST_COOLDOWN   = 3.0

st.set_page_config(page_title="Real-Time Fraud Detection Simulation", layout="wide")
st.title("Real-Time Fraud Monitoring Simulation")
st.write("Upload your historical state data and your live streaming data to begin the simulation.")

col1, col2 = st.columns(2)
with col1:
    warmup_file = st.file_uploader("1. Upload Historical Data for Warmup (CSV)", type=["csv"])
with col2:
    stream_file = st.file_uploader("2. Upload Live Stream Dataset (CSV)", type=["csv"])

if warmup_file is not None and stream_file is not None:

    df_warmup = pd.read_csv(warmup_file).sort_values("TransactionDT")
    df_stream = pd.read_csv(stream_file).sort_values("TransactionDT")

    st.info(f"Loaded {len(df_warmup)} warmup rows and {len(df_stream)} streaming rows.")

    if st.button("Start Live Demo", type="primary"):

        # ── Warmup ────────────────────────────────────────────────────────
        with st.spinner("Warming up historical state on the backend..."):
            try:
                csv_buffer = df_warmup.to_csv(index=False).encode("utf-8")
                res = requests.post(API_URL_WARMUP,
                                    files={"file": ("warmup.csv", csv_buffer, "text/csv")})
                res.raise_for_status()
                st.success(f"State initialized! {res.json().get('users_warmed_up', 0)} unique users tracked.")
            except Exception as e:
                st.error(f"Failed to warmup backend: {e}")
                st.stop()

        # ── Stream UI placeholders ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Live Incoming Transactions")

        metrics_placeholder = st.empty()
        chart_placeholder   = st.empty()
        table_placeholder   = st.empty()

        # ── Shared state ───────────────────────────────────────────────────
        api_latencies   = []
        model_latencies = []
        request_times   = deque(maxlen=500)
        score_history   = []
        table_data      = []
        last_burst_time = 0.0
        row_iter        = df_stream.iterrows()
        rows_remaining  = [len(df_stream)]

        # ── Helpers ────────────────────────────────────────────────────────
        def pull_next_row():
            try:
                _, row = next(row_iter)
                rows_remaining[0] -= 1
                return row.where(pd.notna(row), None).to_dict(), rows_remaining[0]
            except StopIteration:
                return None, 0

        def send_one(payload: dict) -> dict | None:
            try:
                start = time.perf_counter()
                resp  = requests.post(API_URL_STREAM, json=payload, timeout=5)
                wall  = time.perf_counter() - start
                resp.raise_for_status()
                result = resp.json()
                result["_wall_ms"] = wall * 1000
                api_latencies.append(wall * 1000)
                model_latencies.append(float(result.get("model_latency_ms", 0.0)))
                request_times.append(time.time())
                return result
            except Exception as exc:
                st.warning(f"Request error: {exc}")
                return None

        def update_ui(result: dict):
            prob   = result.get("Fraud_Probability", 0.0)
            uid    = result.get("uid", "—")
            local_time = datetime.now().strftime("%H:%M:%S")
            status = "🚨 FRAUD" if prob > 0.68 else "✅ CLEAR"

            score_history.append(prob)
            table_data.insert(0, {
                "UID":      uid,
                "Time":     local_time,
                "Score":    round(prob, 4),
                "Status":   status,
                "API ms":   round(result["_wall_ms"], 1),
                "Model ms": round(result.get("model_latency_ms", 0), 2),
            })
            if len(table_data) > 12:
                table_data.pop()

            now  = time.time()
            n    = len(api_latencies)
            rps  = sum(t > now - 1 for t in request_times)
            p50  = float(np.percentile(api_latencies,   50)) if n else 0
            p95  = float(np.percentile(api_latencies,   95)) if n else 0
            mp50 = float(np.percentile(model_latencies, 50)) if n else 0
            mp95 = float(np.percentile(model_latencies, 95)) if n else 0

            with metrics_placeholder.container():
                r1 = st.columns(4)
                r1[0].metric("UID",             str(uid))
                r1[1].metric("Time",            local_time)
                r1[2].metric("Status",          status, delta=f"{prob:.4f}")
                r1[3].metric("RPS (live)",      str(rps))

                r2 = st.columns(4)
                r2[0].metric("API p50 (ms)",    f"{p50:.1f}")
                r2[1].metric("API p95 (ms)",    f"{p95:.1f}")
                r2[2].metric("Model p50 (ms)",  f"{mp50:.2f}")
                r2[3].metric("Model p95 (ms)",  f"{mp95:.2f}")

                r3 = st.columns(3)
                r3[0].metric("Total sent",      str(n))
                r3[1].metric("High-risk (>0.45)",
                             str(sum(s > 0.68 for s in score_history)))
                r3[2].metric("Rows remaining",  str(rows_remaining[0]))

            chart_placeholder.line_chart(
                score_history[-300:], height=220, use_container_width=True
            )
            table_placeholder.dataframe(
                pd.DataFrame(table_data), use_container_width=True, hide_index=True
            )

        # ── Main streaming loop ────────────────────────────────────────────
        while True:
            payload, remaining = pull_next_row()
            if payload is None:
                st.success("Stream complete — all transactions processed.")
                break

            now        = time.time()
            in_cooldown = (now - last_burst_time) < BURST_COOLDOWN

            if not in_cooldown and np.random.random() < BURST_PROB:
                # Burst: collect rows then fire concurrently
                burst_n       = int(np.random.randint(*BURST_SIZE_RANGE))
                burst_payloads = [payload]
                for _ in range(burst_n - 1):
                    bp, remaining = pull_next_row()
                    if bp is None:
                        break
                    burst_payloads.append(bp)

                results_bucket: list[dict | None] = [None] * len(burst_payloads)

                def _fire(idx, pld):
                    results_bucket[idx] = send_one(pld)

                threads = [
                    threading.Thread(target=_fire, args=(i, p), daemon=True)
                    for i, p in enumerate(burst_payloads)
                ]
                for t in threads: t.start()
                for t in threads: t.join()

                for r in results_bucket:
                    if r:
                        update_ui(r)

                last_burst_time = time.time()

            else:
                result = send_one(payload)
                if result:
                    update_ui(result)

                gap = np.random.exponential(1.0 / BASE_RATE_RPS)
                time.sleep(gap)