import os
import json
import random
import logging
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document


# Load environment variables (e.g., OpenAI API Key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit's page layout and title
st.set_page_config(page_title="IT Support System (RAG)", layout="centered")

# Initialize session memory to store chat history across interactions in the session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize memory if it's not already present

# Knowledge Base Setup: Load FAQs from a JSON file and prepare documents for retrieval
kb_path = os.path.join(os.path.dirname(__file__), 'kb.json')  # Path to the knowledge base JSON file
with open(kb_path, encoding='utf-8') as f:
    kb_entries = json.load(f)  # Load the knowledge base entries from the JSON file

# Convert knowledge base entries into documents for Chroma vector database
docs: List[Document] = []
for entry in kb_entries:
    docs.append(Document(
        page_content=entry['answer'],  # The answer to the query is the document's content
        metadata={  # Store additional metadata for each document (question and ID)
            'id': entry.get('id'),
            'question': entry.get('question')
        }
    ))

# Initialize OpenAI embeddings using the API key
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Set up a Chroma vector store for semantic document search, using the embeddings
vectordb = Chroma.from_documents(
    documents=docs,  # List of documents to index
    embedding=embeddings,  # Embedding function to vectorize documents
    persist_directory='db/chroma'  # Directory to store the vector store (persistent storage)
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Setup retrieval with top-3 relevant docs

def retrieve_docs(query: str) -> str:
    """Retrieve relevant documentation from the knowledge base based on the user's query."""
    hits = retriever.get_relevant_documents(query)  # Retrieve top 3 relevant documents for the query
    if not hits:
        return "No relevant documentation found."  # If no results found, return this message
    
    # Format the retrieved documents into a readable response for the user
    results = []
    for doc in hits:
        question = doc.metadata.get('question', 'FAQ')  # Retrieve question from metadata
        results.append(f"**Q: {question}**\nA: {doc.page_content}")  # Format each result
    
    return "\n\n".join(results)  # Return all formatted results as a string

def escalate_ticket(query: str, analysis: str = "") -> str:
    """Create and log a support ticket for issues that need human intervention."""
    ticket_id = f"TICKET-{random.randint(1000, 9999)}"  # Generate a random ticket ID
    description = f"User Query: {query}\nAnalysis: {analysis}"  # Detailed description for the support team
    # Simulate logging to an external ticketing system (e.g., JIRA, ServiceNow)
    logger.info(f"Escalating issue with ticket {ticket_id}: {description}")
    return f"Escalated issue. Created ticket {ticket_id}. A support technician will contact you shortly."

# LLM Configuration: Setup GPT-4 for agent communication
llm_config = {
    "config_list": [{
        "model": "gpt-4",  # Using GPT-4 model for all agents
        "api_key": OPENAI_API_KEY,
    }],
    "seed": 42,  # Fixed seed for reproducibility in responses
    "temperature": 0.5,  # Temperature for GPT-4 model (controls randomness of responses)
}

# Agent Definitions:
# Master Agent - Coordinates the overall process and manages the IT query flow
master_agent = AssistantAgent(
    name="Master",
    llm_config=llm_config,
    system_message=""" 
    You are the Master Agent that orchestrates the IT support workflow:
    1. Determine if the query is IT-related. If not, provide a direct response explaining limitations.
    2. For IT-related queries, pass to the Planning Agent for execution plan development.
    3. Receive and review results from all agents.
    4. Provide a concise final response to the user.
    """
)

# Planning Agent - Validates and categorizes user queries, preparing them for resolution
planning_agent = AssistantAgent(
    name="Planning",
    llm_config=llm_config,
    system_message=""" 
    You are the Planning Agent responsible for:
    1. Validating if the user query is clear and complete.
    2. Refining the query if needed.
    3. Creating a structured execution plan.
    4. Categorizing the IT issue type (e.g., hardware, software, networking).
    """
)

# Analysis Agent - Extracts details from the query and determines severity of the issue
analysis_agent = AssistantAgent(
    name="Analysis",
    llm_config=llm_config,
    system_message=""" 
    You are the Analysis Agent responsible for:
    1. Identifying key entities in the query (e.g., device, error codes).
    2. Determining severity (Low, Medium, High, Critical).
    3. Structuring information for the Resolution phase.
    """
)

# Resolution Agent - Uses predefined knowledge to resolve issues or escalates if needed
resolution_agent = AssistantAgent(
    name="Resolution",
    llm_config=llm_config,
    system_message=""" 
    You are the Resolution Agent responsible for:
    1. Using tools to retrieve knowledge base articles or documentation.
    2. Providing clear step-by-step instructions for issue resolution.
    3. Deciding if the issue requires escalation to a technician.
    """
)

# Escalation Agent - Escalates complex issues to IT support by generating tickets
escalation_agent = AssistantAgent(
    name="Escalation",
    llm_config=llm_config,
    system_message=""" 
    You are the Escalation Agent responsible for:
    1. Creating support tickets for issues that cannot be resolved automatically.
    2. Sending ticket information to the IT team.
    3. Ensuring the user receives follow-up information and expectations.
    """
)

# Main function to handle IT query processing through the multi-agent workflow
def handle_it_query(query: str) -> str:
    """Process IT queries through the multi-agent workflow."""
    query = query.strip()  # Remove leading/trailing spaces
    if not query:
        return "Please enter an IT question or issue."  # Return message if no query is provided

    workflow_logs = {"query": query}
    
    try:
        # 1. Master Agent determines if the query is IT-related
        master_proxy = UserProxyAgent(name="MasterProxy", human_input_mode="NEVER", code_execution_config=False)
        master_prompt = f"User query: '{query}'. Determine if this is an IT-related issue."
        master_proxy.initiate_chat(master_agent, message=master_prompt, max_turns=1)
        initial_assessment = master_proxy.chat_messages[master_agent][-1]["content"]
        workflow_logs["initial_assessment"] = initial_assessment
        
        # If not IT-related, return the response directly
        if "NOT IT-RELATED" in initial_assessment.upper():
            return initial_assessment
        
        # 2. Planning Agent: Validate and categorize the issue
        plan_proxy = UserProxyAgent(name="PlanningProxy", human_input_mode="NEVER", code_execution_config=False)
        plan_proxy.initiate_chat(planning_agent, message=query, max_turns=1)
        planning_output = plan_proxy.chat_messages[planning_agent][-1]["content"]
        workflow_logs["planning"] = planning_output
        
        # 3. Analysis Agent: Extract key details and determine issue severity
        analysis_proxy = UserProxyAgent(name="AnalysisProxy", human_input_mode="NEVER", code_execution_config=False)
        analysis_proxy.initiate_chat(analysis_agent, message=planning_output, max_turns=1)
        analysis_output = analysis_proxy.chat_messages[analysis_agent][-1]["content"]
        workflow_logs["analysis"] = analysis_output
        
        # 4. Resolution Agent: Resolve the issue or escalate if needed
        res_proxy = UserProxyAgent(name="ResolutionProxy", human_input_mode="NEVER", code_execution_config=False)
        resolution_input = f"User Query: {query}\n\nPlanning: {planning_output}\n\nAnalysis: {analysis_output}"
        res_proxy.initiate_chat(resolution_agent, message=resolution_input, max_turns=1)
        resolution_output = res_proxy.chat_messages[resolution_agent][-1]["content"]
        workflow_logs["resolution"] = resolution_output
        
        # 5. Escalation if needed: Handle complex issues that require human intervention
        escalation_output = None
        if "ESCALATION NEEDED" in resolution_output.upper():
            esc_proxy = UserProxyAgent(name="EscalationProxy", human_input_mode="NEVER", code_execution_config=False)
            escalation_input = f"Original Query: {query}\n\nAnalysis: {analysis_output}\n\nResolution Attempt: {resolution_output}"
            esc_proxy.initiate_chat(escalation_agent, message=escalation_input, max_turns=1)
            escalation_output = esc_proxy.chat_messages[escalation_agent][-1]["content"]
            workflow_logs["escalation"] = escalation_output

        # 6. Final response from Master Agent
        final_master_proxy = UserProxyAgent(name="FinalMasterProxy", human_input_mode="NEVER", code_execution_config=False)
        final_prompt = (
            f"Complete workflow results for query: '{query}':\n\n"
            f"Planning: {planning_output}\n\n"
            f"Analysis: {analysis_output}\n\n"
            f"Resolution: {resolution_output}\n\n"
            f"Synthesize these results into a clear, helpful response for the user."
        )
        
        final_master_proxy.initiate_chat(master_agent, message=final_prompt, max_turns=1)
        final_response = final_master_proxy.chat_messages[master_agent][-1]["content"]
        workflow_logs["final_response"] = final_response

        # Save results to session memory
        st.session_state.chat_history.append({
            "user": query, 
            "assistant": final_response,
            "workflow_logs": workflow_logs
        })
        
        return final_response

    except Exception as e:
        logger.error(f"Error in workflow: {e}", exc_info=True)
        return f"An error occurred during processing: {str(e)}\n\nPlease try rephrasing your question."


# Streamlit User Interface Setup
st.title("AI Help Desk")
st.write("Ask any IT support question, and our multi-agent system will assist you.")

# Form for users to submit queries
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_area("Describe your IT issue:", height=100)
    show_logs = st.checkbox("Show workflow details", value=False)
    submitted = st.form_submit_button("Submit")

    # Handle the query submission
    if submitted:
        if not user_input:
            st.error("Please type a message before submitting.")
        else:
            with st.spinner("Processing your request through our agent workflow..."):
                response = handle_it_query(user_input)
            
            # Display the final response to the user
            st.markdown("### Response")
            st.write(response)

from sklearn.externals import joblib  # For loading saved ML models
import numpy as np

# Example of a simple ML model (replace with your trained model)
model = joblib.load('path_to_trained_model.pkl')  # Load a pre-trained ML model for issue classification

def predict_issue(query: str) -> str:
    """Predict the issue type using an ML model."""
    # Preprocess the query (this would depend on the model you're using)
    processed_query = preprocess_query(query)
    
    # Make a prediction based on the model
    prediction = model.predict([processed_query])
    issue_type = prediction[0]
    
    return issue_type

# Before passing the query to the Planning Agent, predict the issue type
predicted_issue = predict_issue(user_input)


def collect_feedback(user_feedback: str, query: str, solution: str):
    """Collect user feedback to improve the system."""
    feedback_data = {
        "query": query,
        "solution": solution,
        "feedback": user_feedback
    }
    
    # Save the feedback in the database (SQLite)
    with sqlite3.connect('feedback.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (query TEXT, solution TEXT, feedback TEXT)''')
        cursor.execute('INSERT INTO feedback (query, solution, feedback) VALUES (?, ?, ?)', 
                       (query, solution, user_feedback))
        conn.commit()

    # Optionally, retrain the model with new feedback data
    retrain_model()

def retrain_model():
    """Retrain the model based on the collected feedback."""
    # Here, you would collect the feedback data and retrain the model
    print("Retraining model with new feedback...")


import requests

def create_service_now_ticket(query: str, analysis: str):
    """Create a support ticket in ServiceNow for complex issues."""
    # Define the ServiceNow endpoint and credentials
    url = "https://instance.service-now.com/api/now/table/incident"
    headers = {"Content-Type": "application/json", "Authorization": "Basic <Your_ServiceNow_API_Auth>"}
    
    # Prepare the data for the ticket
    ticket_data = {
        "short_description": query,
        "description": analysis,
        "category": "IT Support",
        "urgency": "1",  # Set based on issue severity
        "impact": "1",   # Set based on issue impact
    }
    
    # Send a POST request to ServiceNow to create a ticket
    response = requests.post(url, json=ticket_data, headers=headers)
    
    if response.status_code == 201:
        ticket_number = response.json()['result']['number']
        return f"Ticket created successfully. Ticket number: {ticket_number}"
    else:
        return "Failed to create a ticket in ServiceNow."

def handle_it_query(query: str) -> str:
    """Process IT queries through the multi-agent workflow, with ML and external integration."""
    query = query.strip()  # Clean the query input
    if not query:
        return "Please enter an IT question or issue."

    workflow_logs = {"query": query}

    try:
        # Predict the issue type using Machine Learning (ML)
        predicted_issue = predict_issue(query)  # Predict the IT issue category (e.g., hardware, network)

        # First, determine if it's an IT issue through the Master Agent
        master_proxy = UserProxyAgent(name="MasterProxy", human_input_mode="NEVER", code_execution_config=False)
        master_prompt = f"User query: '{query}'. Is this issue related to IT?"
        master_proxy.initiate_chat(master_agent, message=master_prompt, max_turns=1)
        initial_assessment = master_proxy.chat_messages[master_agent][-1]["content"]
        workflow_logs["initial_assessment"] = initial_assessment

        # If not IT-related, return response directly
        if "NOT IT-RELATED" in initial_assessment.upper():
            return initial_assessment

        # Planning, Analysis, and Resolution Steps
        plan_proxy = UserProxyAgent(name="PlanningProxy", human_input_mode="NEVER", code_execution_config=False)
        plan_proxy.initiate_chat(planning_agent, message=query, max_turns=1)
        planning_output = plan_proxy.chat_messages[planning_agent][-1]["content"]
        workflow_logs["planning"] = planning_output

        analysis_proxy = UserProxyAgent(name="AnalysisProxy", human_input_mode="NEVER", code_execution_config=False)
        analysis_proxy.initiate_chat(analysis_agent, message=planning_output, max_turns=1)
        analysis_output = analysis_proxy.chat_messages[analysis_agent][-1]["content"]
        workflow_logs["analysis"] = analysis_output

        res_proxy = UserProxyAgent(name="ResolutionProxy", human_input_mode="NEVER", code_execution_config=False)
        resolution_input = f"User Query: {query}\n\nPlanning: {planning_output}\n\nAnalysis: {analysis_output}"
        res_proxy.initiate_chat(resolution_agent, message=resolution_input, max_turns=1)
        resolution_output = res_proxy.chat_messages[resolution_agent][-1]["content"]
        workflow_logs["resolution"] = resolution_output

        # Escalation if necessary
        escalation_output = None
        if "ESCALATION NEEDED" in resolution_output.upper():
            # Create a ticket in ServiceNow
            escalation_output = create_service_now_ticket(query, analysis_output)
            workflow_logs["escalation"] = escalation_output

        # Master summarization and final response
        final_master_proxy = UserProxyAgent(name="FinalMasterProxy", human_input_mode="NEVER", code_execution_config=False)
        final_prompt = f"Complete workflow results: {workflow_logs}"
        final_master_proxy.initiate_chat(master_agent, message=final_prompt, max_turns=1)
        final_response = final_master_proxy.chat_messages[master_agent][-1]["content"]

        # Save to session memory for future reference
        st.session_state.chat_history.append({
            "user": query,
            "assistant": final_response,
            "workflow_logs": workflow_logs
        })

        return final_response

    except Exception as e:
        logger.error(f"Error in workflow: {e}", exc_info=True)
        return f"An error occurred: {str(e)}. Please try again later."

# Streamlit UI setup (no changes required here)
