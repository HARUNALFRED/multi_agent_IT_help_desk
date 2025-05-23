import os
import json
import random
import logging
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from autogen import AssistantAgent, UserProxyAgent

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Setup - Create a mock database with tables and sample data
DB_PATH = "banking_data.db"

def create_db():
    # Create SQLite Database and tables
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            date TEXT,
            amount REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_acquisition (
            id INTEGER PRIMARY KEY,
            date TEXT,
            customers_acquired INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interest_data (
            id INTEGER PRIMARY KEY,
            date TEXT,
            interest_rate REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS loans_data (
            id INTEGER PRIMARY KEY,
            date TEXT,
            loan_amount REAL
        )
    ''')

    # Insert sample data (200 rows)
    for _ in range(200):
        cursor.execute("INSERT INTO sales (date, amount) VALUES (?, ?)", 
                       (f"2025-06-{random.randint(1, 30)}", random.uniform(1000, 10000)))
        cursor.execute("INSERT INTO customer_acquisition (date, customers_acquired) VALUES (?, ?)", 
                       (f"2025-06-{random.randint(1, 30)}", random.randint(50, 500)))
        cursor.execute("INSERT INTO interest_data (date, interest_rate) VALUES (?, ?)", 
                       (f"2025-06-{random.randint(1, 30)}", random.uniform(1.5, 5.5)))
        cursor.execute("INSERT INTO loans_data (date, loan_amount) VALUES (?, ?)", 
                       (f"2025-06-{random.randint(1, 30)}", random.uniform(50000, 500000)))

    conn.commit()
    conn.close()

create_db()  # Call this to create and populate the database

# Streamlit UI Setup
st.set_page_config(page_title="AI Analytics Agent", layout="centered")
st.title("AI Analytics Agent for Banking Data")
st.write("Ask questions about the bank's data and generate insights such as charts.")

# Initialize session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# LLM Configuration for Agent
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm_config = {
    "config_list": [{
        "model": "gpt-4",
        "api_key": OPENAI_API_KEY,
    }],
    "seed": 42,
    "temperature": 0.5,
}

# Agent Definitions
analytics_agent = AssistantAgent(
    name="AnalyticsAgent",
    llm_config=llm_config,
    system_message="""You are the AI Analytics Agent. Your role is to fetch data from the database, 
                      generate visual insights like charts or trend lines, and provide textual responses to user queries."""
)

# Query Handling Function
def handle_query(query: str) -> str:
    """Process the query, retrieve data from the database, and generate the response."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Handle different queries
    if "monthly sales for June" in query.lower():
        month_sales_query = """
        SELECT date, amount FROM sales WHERE date LIKE '2025-06%' ORDER BY date
        """
        df = pd.read_sql(month_sales_query, conn)
        if df.empty:
            return "No sales data found for June 2025."
        
        # Generate a trend chart for monthly sales
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        monthly_sales = df.resample('M').sum()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(monthly_sales.index, monthly_sales['amount'], marker='o', color='blue', linestyle='-', linewidth=2)
        ax.set_title("Monthly Sales for June 2025")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales Amount")
        ax.grid(True)

        # Save the chart to a temporary file and display it in Streamlit
        chart_file = "monthly_sales_chart.png"
        fig.savefig(chart_file)
        st.image(chart_file)

        return "Here is the trend chart for the monthly sales for June 2025."

    # Add more queries as needed

    conn.close()
    return "Sorry, I didn't understand that. Please try again."

# User Input and Interaction
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_area("Ask a question related to banking data (e.g., monthly sales, customer acquisition):", height=100)
    submitted = st.form_submit_button("Submit")

    if submitted:
        if not user_input:
            st.error("Please enter a question before submitting.")
        else:
            with st.spinner("Processing your request..."):
                response = handle_query(user_input)
            
            st.markdown("### Response")
            st.write(response)
