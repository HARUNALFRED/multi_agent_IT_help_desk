
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import re
import openai
import random
from datetime import datetime, timedelta
import streamlit as st

# --- OpenAI API Key setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Connect to SQLite Database ---
conn = sqlite3.connect("banking.db")
cursor = conn.cursor()

# --- Drop and recreate the transactions table ---
cursor.execute("DROP TABLE IF EXISTS transactions")
cursor.execute("""
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    customer_id INTEGER,
    amount REAL,
    type TEXT
)
""")

# --- Insert Mock Data for Testing ---
base_date = datetime(2025, 1, 1)
for i in range(300):
    date = base_date + timedelta(days=random.randint(0, 180))
    cursor.execute("""
        INSERT INTO transactions (date, customer_id, amount, type) 
        VALUES (?, ?, ?, ?)
    """, (
        date.strftime('%Y-%m-%d'),
        random.randint(1000, 1100),
        round(random.uniform(100, 1000), 2),
        'credit'
    ))
conn.commit()

# --- Enhanced NLP Query Interpretation ---
def parse_query(query):
    query = query.lower()

    # Detect month, year, and quarter in the query
    month_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)", query)
    year_match = re.search(r"(202[0-9])", query)
    quarter_match = re.search(r"(q[1-4])", query)

    month = month_match.group(0).capitalize() if month_match else None
    year = year_match.group(0) if year_match else "2025"
    quarter = quarter_match.group(0) if quarter_match else None

    # Intent detection using OpenAI ChatCompletion
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that classifies user queries related to banking transactions."},
                {"role": "user", "content": f"Classify the intent of the following query: {query}"}
            ],
            max_tokens=50,
            temperature=0
        )
        intent = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None, None

    # Handling customer count queries
    if "customer" in query and ("count" in query or "total" in query):
        if month:
            return "customer_count", f"{year}-{datetime.strptime(month, '%B').month:02d}"
        return "customer_count", None

    # Handling monthly sales queries
    if "sales" in query and month:
        return "monthly_sales", month

    # Handling total sales queries for a year
    if "total sales" in query:
        return "total_sales", year

    # Handling sales trend queries
    if "trend" in query and "sales" in query or "plot" in query and "sales" in query:
        return "sales_trend", None

    # Handling sales by quarter
    if quarter:
        return "quarterly_sales", quarter

    return None, None

# --- SQL Query Execution Functions ---
def get_monthly_sales(month_name):
    try:
        month_number = datetime.strptime(month_name, "%B").month
    except ValueError:
        return pd.DataFrame()
    query = f"""
        SELECT strftime('%Y-%m', date) as month, SUM(amount) as total_sales
        FROM transactions
        WHERE strftime('%m', date) = '{month_number:02d}'
        GROUP BY month
    """
    return pd.read_sql_query(query, conn)

def get_customer_count(month_prefix):
    query = f"""
        SELECT COUNT(DISTINCT customer_id) as customer_count
        FROM transactions
        WHERE date LIKE '{month_prefix}%'
    """
    return pd.read_sql_query(query, conn)

def get_total_sales(year):
    query = f"""
        SELECT SUM(amount) as total_sales
        FROM transactions
        WHERE strftime('%Y', date) = '{year}'
    """
    return pd.read_sql_query(query, conn)

def get_quarterly_sales(quarter):
    quarter_months = {
        "q1": ['01', '02', '03'],
        "q2": ['04', '05', '06'],
        "q3": ['07', '08', '09'],
        "q4": ['10', '11', '12']
    }
    months = quarter_months.get(quarter.lower(), [])
    if not months:
        return pd.DataFrame()
    query = f"""
        SELECT strftime('%Y-%m', date) as quarter, SUM(amount) as total_sales
        FROM transactions
        WHERE strftime('%m', date) IN ({','.join([f'"{m}"' for m in months])})
        GROUP BY quarter
    """
    return pd.read_sql_query(query, conn)

def get_sales_trend():
    query = f"""
        SELECT strftime('%Y-%m', date) as month, SUM(amount) as total_sales
        FROM transactions
        GROUP BY month
        ORDER BY month
    """
    return pd.read_sql_query(query, conn)

# --- Streamlit App UI ---
st.title("Banking Analytics Assistant")
user_input = st.text_input("Ask a question:", "Generate bar chart /trendMonthly Sales for June")

if st.button("Run Query"):
    intent, arg = parse_query(user_input)

    if intent == "monthly_sales":
        df = get_monthly_sales(arg)
        if not df.empty:
            st.dataframe(df)
            fig, ax = plt.subplots()
            ax.bar(df["month"], df["total_sales"])
            ax.set_title(f"Monthly Sales for {arg}")
            st.pyplot(fig)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), f"sales_{arg}.csv", "text/csv")
        else:
            st.warning(f"No sales data found for {arg}")
    
    elif intent == "customer_count":
        df = get_customer_count(arg)
        if not df.empty:
            st.write(f"Total Customers in {arg}: {df['customer_count'].iloc[0]}")
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), f"customers_{arg}.csv", "text/csv")
        else:
            st.warning(f"No customer data found for {arg}")
    
    elif intent == "total_sales":
        df = get_total_sales(arg)
        if not df.empty:
            st.write(f"Total Sales in {arg}: {df['total_sales'].iloc[0]}")
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), f"total_sales_{arg}.csv", "text/csv")
        else:
            st.warning(f"No sales data found for {arg}")
    
    elif intent == "sales_trend":
        df = get_sales_trend()
        if not df.empty:
            st.dataframe(df)
            fig, ax = plt.subplots()
            ax.plot(df["month"], df["total_sales"], marker='o')
            ax.set_title("Sales Trend Over Time")
            ax.set_ylabel("Total Sales")
            st.pyplot(fig)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "sales_trend.csv", "text/csv")
        else:
            st.warning("No trend data found.")

    elif intent == "quarterly_sales":
        df = get_quarterly_sales(arg)
        if not df.empty:
            st.dataframe(df)
            fig, ax = plt.subplots()
            ax.bar(df["quarter"], df["total_sales"])
            ax.set_title(f"Sales in {arg}")
            st.pyplot(fig)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), f"sales_{arg}.csv", "text/csv")
        else:
            st.warning(f"No sales data found for {arg}")
    else:
        st.warning("Sorry, I could not understand your query.")
