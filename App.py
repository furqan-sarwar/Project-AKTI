import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pypdf
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import streamlit as st
from io import BytesIO

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Streamlit UI for uploading a PDF and interacting with the AI agent
st.title("AI AGENT FOR CHAT WITH PDF")

# Step 1: Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Step 2: Save the uploaded PDF file temporarily
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and split the uploaded PDF document
    document_loader = PyPDFLoader("uploaded_pdf.pdf")
    document = document_loader.load_and_split()

    # Step 3: Create embeddings and store chunks in FAISS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    text = text_splitter.split_documents(document)

    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=GOOGLE_API_KEY)
    vector = FAISS.from_documents(text, embeddings)
    retriever = vector.as_retriever()

    # Step 4: Tool for retrieving information from the PDF
    from langchain.tools.retriever import create_retriever_tool
    retriever_tool = create_retriever_tool(retriever=retriever, name="PDF Search Tool",
                                           description="Use this tool to search information from the uploaded PDF.")

    search = TavilySearchResults()

    # Step 5: Create tools and LLM
    tools = [search, retriever_tool]
    chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=GOOGLE_API_KEY)

    # Load the prompt for the agent
    prompt = hub.pull("hwchase17/react")

    # Step 6: Create the agent and executor
    agent = create_react_agent(llm=chat, tools=tools, prompt=prompt)
    executer = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Step 7: Ask a question about the uploaded PDF
    user_question = st.text_input("Ask a question about the uploaded PDF")

    if user_question:
        # Step 8: Get the agent's response
        with st.spinner("Searching for the answer..."):
            if user_question:
                response = executer.invoke({"input": user_question})

                answer = response.get('output', 'No response received.')
                
                st.markdown(f"**User:**\n\n{user_question}")
                st.markdown(f"**AI:**\n\n{answer}")
                
     else:
         st.error("Please provide accurate input.")

            
# else:
    # st.warning("Please upload a PDF file to get started.")
