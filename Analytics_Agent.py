import os
import json
import sqlite3
import random
import pandas as pd
import plotly.express as px
from openai import OpenAI
from fpdf import FPDF
from datetime import datetime, timedelta
import logging
import streamlit as st

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Connect to SQLite DB ---
conn = sqlite3.connect("banking.db")
cursor = conn.cursor()

# --- Function to Get Database Schema ---
def get_schema():
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schema[table_name] = [col[1] for col in columns]
    return schema

# --- GPT SQL Suggestion ---
def ask_gpt_for_sql(user_query):
    schema = get_schema()  # Get schema dynamically
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a banking SQL assistant. Your job is to translate natural questions to SQLite queries.\n"
                    f"The database includes the following tables and columns:\n"
                    f"{json.dumps(schema, indent=4)}\n"  # Pass the schema to OpenAI
                    "Use SQLite-compatible syntax: strftime for dates, CASE for quarters.\n"
                    "Respond with JSON: { \"intent\": ..., \"sql\": ..., \"response_type\": \"text\" or \"chart\" }"
                )
            },
            {"role": "user", "content": user_query}
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2
        )
        reply = response.choices[0].message.content
        result = json.loads(reply)
        return result["intent"], result["sql"], result.get("response_type", "text")
    except Exception as e:
        st.error(f"❌ OpenAI Error: {e}")
        return None, None, None

# --- Streamlit UI ---
st.set_page_config(page_title="Banking Assistant", layout="wide")
st.title("📊 Banking Analytics AI Assistant")
user_input = st.text_input("Ask a question:", "Show monthly sales trends by account type")

if st.button("Run Query"):
    intent, sql, response_type = ask_gpt_for_sql(user_input)

    if not intent or not sql:
        st.warning("⚠️ I couldn't understand your query.")
        st.stop()

    st.markdown("#### 🧠 GPT-Suggested SQL:")
    st.code(sql, language="sql")

    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        st.error(f"💥 SQL Execution Error: {e}")
        st.stop()

    if df.empty:
        st.warning("📭 No data returned.")
        st.stop()

    st.dataframe(df)

    if response_type == "chart" and df.shape[1] >= 2:
        x_col, y_col = df.columns[0], df.columns[1]
        st.plotly_chart(px.bar(df, x=x_col, y=y_col, color=y_col,
                               hover_name=x_col, text_auto='.2s',
                               title=intent.replace("_", " ").title()))
    elif response_type == "text" and df.shape == (1, 1):
        st.success(f"{df.columns[0]}: {df.iloc[0, 0]}")

    st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "results.csv", "text/csv")

    # End