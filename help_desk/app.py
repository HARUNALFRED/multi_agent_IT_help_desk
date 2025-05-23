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


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="IT Support System (RAG)", layout="centered")

# Initialize session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

#Knowledge Base Setup

kb_path = os.path.join(os.path.dirname(__file__), 'kb.json')
with open(kb_path, encoding='utf-8') as f:
    kb_entries = json.load(f)

docs: List[Document] = []
for entry in kb_entries:
    docs.append(Document(
        page_content=entry['answer'],
        metadata={
            'id': entry.get('id'),
            'question': entry.get('question')
        }
    ))

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory='db/chroma'
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def retrieve_docs(query: str) -> str:
    """Retrieve relevant documentation from the knowledge base"""
    hits = retriever.get_relevant_documents(query)
    if not hits:
        return "No relevant documentation found."
    
    results = []
    for doc in hits:
        question = doc.metadata.get('question', 'FAQ')
        results.append(f"**Q: {question}**\nA: {doc.page_content}")
    return "\n\n".join(results)

def escalate_ticket(query: str, analysis: str = "") -> str:
    """Create a ticket for issues that need human intervention"""
    ticket_id = f"TICKET-{random.randint(1000, 9999)}"
    description = f"User Query: {query}\nAnalysis: {analysis}"
    # In a real system, you would send this to a ticketing system like JIRA, ServiceNow, etc.
    logger.info(f"Escalating issue with ticket {ticket_id}: {description}")
    return f"Escalated issue. Created ticket {ticket_id}. A support technician will contact you shortly."

# LLM Configuration
llm_config = {
    "config_list": [{
        "model": "gpt-4",
        "api_key": OPENAI_API_KEY,
    }],
    "seed": 42,
    "temperature": 0.5,
}

# Agent Definitions
master_agent = AssistantAgent(
    name="Master",
    llm_config=llm_config,
    system_message="""
You are the Master Agent that orchestrates the IT support workflow:
1. First determine if the query is IT-related. If not, provide a direct response explaining your limitations.
2. For IT-related queries, pass to the Planning Agent for execution plan development
3. Receive and review the complete workflow results from all agents
4. Provide a comprehensive yet concise final response to the user

Only handle one query at a time through the complete workflow.
"""
)

planning_agent = AssistantAgent(
    name="Planning",
    llm_config=llm_config,
    system_message="""
You are the Planning Agent responsible for:
1. Validating if the user query is clear and complete
2. Refining the query if needed for better processing
3. Creating a structured execution plan with clear steps
4. Categorizing the IT issue type (hardware, software, network, access, etc.)

Provide your analysis as a structured output with sections for:
- Query Validation
- Issue Category
- Execution Plan

Always end your message with: "Forwarding to Analysis Agent"
"""
)

analysis_agent = AssistantAgent(
    name="Analysis",
    llm_config=llm_config,
    system_message="""
You are the Analysis Agent responsible for:
1. Identifying key entities in the user query (devices, software, errors, etc.)
2. Determining severity level (Low, Medium, High, Critical)
3. Extracting technical details mentioned in the query
4. Structuring this information for the Resolution phase

Provide your analysis as structured output with sections for:
- Key Entities
- Technical Details
- Severity
- Analysis Summary

Always end your message with: "Forwarding to Resolution Agent"
"""
)

resolution_agent = AssistantAgent(
    name="Resolution",
    llm_config=llm_config,
    system_message="""
You are the Resolution Agent responsible for:
1. Using available tools to retrieve relevant knowledge base articles or documentation
2. Applying the retrieved information to the specific user issue
3. Providing clear step-by-step instructions for resolution
4. Determining if the issue requires escalation to a human technician

If you can resolve the issue:
- Provide clear instructions
- Include any relevant documentation references
- End with "RESOLUTION COMPLETE"

If escalation is needed:
- Explain why the issue requires escalation
- Provide details that would help a technician understand the issue
- End with "ESCALATION NEEDED"
""",
    function_map={"retrieve_docs": retrieve_docs}
)

escalation_agent = AssistantAgent(
    name="Escalation",
    llm_config=llm_config,
    system_message="""
You are the Escalation Agent responsible for:
1. Creating support tickets for issues that cannot be resolved automatically
2. Providing the user with ticket tracking information
3. Setting expectations for next steps
4. Compiling all analysis from previous agents to assist human technicians

Format your response to be professional and reassuring to the user.
Always include the ticket ID and expected follow-up timeframe.
""",
    function_map={"escalate_ticket": escalate_ticket}
)


def handle_it_query(query: str) -> str:
    """Process IT queries through the multi-agent workflow"""
    query = query.strip()
    if not query:
        return "Please enter an IT question or issue."

    workflow_logs = {"query": query}
    
    try:
        # First determine if it's an IT issue through the Master Agent
        master_proxy = UserProxyAgent(
            name="MasterProxy", human_input_mode="NEVER", code_execution_config=False
        )
        master_prompt = f"User query: '{query}'. First, determine if this is an IT-related issue."
        master_proxy.initiate_chat(master_agent, message=master_prompt, max_turns=1)
        initial_assessment = master_proxy.chat_messages[master_agent][-1]["content"]
        workflow_logs["initial_assessment"] = initial_assessment
        
        # If not IT-related, return the response directly
        if "NOT IT-RELATED" in initial_assessment.upper():
            return initial_assessment
        
        # Planning
        plan_proxy = UserProxyAgent(
            name="PlanningProxy", human_input_mode="NEVER", code_execution_config=False
        )
        plan_proxy.initiate_chat(planning_agent, message=query, max_turns=1)
        planning_output = plan_proxy.chat_messages[planning_agent][-1]["content"]
        workflow_logs["planning"] = planning_output
        logger.info(f"Planning completed: {len(planning_output)} chars")

        # 2: Analysis
        analysis_proxy = UserProxyAgent(
            name="AnalysisProxy", human_input_mode="NEVER", code_execution_config=False
        )
        analysis_proxy.initiate_chat(analysis_agent, message=planning_output, max_turns=1)
        analysis_output = analysis_proxy.chat_messages[analysis_agent][-1]["content"]
        workflow_logs["analysis"] = analysis_output
        logger.info(f"Analysis completed: {len(analysis_output)} chars")

        #  3: Resolution
        res_proxy = UserProxyAgent(
            name="ResolutionProxy", human_input_mode="NEVER", code_execution_config=False
        )
        resolution_input = f"User Query: {query}\n\nPlanning: {planning_output}\n\nAnalysis: {analysis_output}"
        res_proxy.initiate_chat(resolution_agent, message=resolution_input, max_turns=1)
        resolution_output = res_proxy.chat_messages[resolution_agent][-1]["content"]
        workflow_logs["resolution"] = resolution_output
        logger.info(f"Resolution completed: {len(resolution_output)} chars")

        # Escalation if needed
        escalation_output = None
        if "ESCALATION NEEDED" in resolution_output.upper():
            esc_proxy = UserProxyAgent(
                name="EscalationProxy", human_input_mode="NEVER", code_execution_config=False
            )
            escalation_input = f"Original Query: {query}\n\nAnalysis: {analysis_output}\n\nResolution Attempt: {resolution_output}"
            esc_proxy.initiate_chat(escalation_agent, message=escalation_input, max_turns=1)
            escalation_output = esc_proxy.chat_messages[escalation_agent][-1]["content"]
            workflow_logs["escalation"] = escalation_output
            logger.info(f"Escalation completed: {len(escalation_output)} chars")

        # Master summarizes
        final_master_proxy = UserProxyAgent(
            name="FinalMasterProxy", human_input_mode="NEVER", code_execution_config=False
        )
        
        if escalation_output:
            final_prompt = (
                f"Complete workflow results for query: '{query}':\n\n"
                f"Planning: {planning_output}\n\n"
                f"Analysis: {analysis_output}\n\n"
                f"Resolution: {resolution_output}\n\n"
                f"Escalation: {escalation_output}\n\n"
                f"Synthesize these results into a clear, helpful response for the user."
            )
        else:
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

        # Save to memory
        st.session_state.chat_history.append({
            "user": query, 
            "assistant": final_response,
            "workflow_logs": workflow_logs
        })
        
        return final_response

    except Exception as e:
        logger.error(f"Error in workflow: {e}", exc_info=True)
        return f"An error occurred during processing: {str(e)}\n\nPlease try rephrasing your question."


st.title("AI Help Desk")
st.write("Ask any IT support question and our multi-agent system will assist you.")

with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_area("Describe your IT issue:", height=100)
    show_logs = st.checkbox("Show workflow details", value=False)
    submitted = st.form_submit_button("Submit")

    if submitted:
        if not user_input:
            st.error("Please type a message before submitting.")
        else:
            with st.spinner("Processing your request through our agent workflow..."):
                response = handle_it_query(user_input)
            
            st.markdown("### Response")
            st.write(response)

