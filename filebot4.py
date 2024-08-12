import streamlit as st
import pandas as pd
import ollama
import PyPDF2
import re
import folium
from streamlit_folium import folium_static

# Define your custom Modelfile prompt
modelfile = '''
FROM llama3
SYSTEM "modelfile"
'''

# Create the model using your custom Modelfile
ollama.create(model='data_processor', modelfile=modelfile)

# Function to read data from different file types
def read_data(file):
    if file.type == "text/csv":
        return pd.read_csv(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)
    elif file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "text/plain":
        return read_txt(file)
    else:
        return None

# Function to read PDF files
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return pd.DataFrame({"text": [text]})

# Function to read TXT files
def read_txt(file):
    return pd.DataFrame({"text": [file.read().decode("utf-8")]})

# Function to combine multiple DataFrames
def combine_dataframes(dataframes):
    return pd.concat(dataframes, ignore_index=True)

# Function to get the data types summary
def summarize_data_types(data):
    data_types = {column: str(data[column].dtype) for column in data.columns}
    return data_types

# Function to extract URLs from text
def extract_urls(text):
    url_pattern = re.compile(r'(https?://\S+)')
    urls = url_pattern.findall(text)
    return urls

# Function to detect location queries and provide map representation
def is_location_query(query):
    location_keywords = ["map", "location", "where", "place", "city", "area"]
    return any(keyword in query.lower() for keyword in location_keywords)

def get_map(location):
    map_ = folium.Map(location=[28.6139, 77.2090], zoom_start=12)  # Default to New Delhi
    folium.Marker(location=[28.6139, 77.2090], popup=location).add_to(map_)
    return map_

# Function to get response from the Ollama model based on data
def getOllamaResponse(data, input_text):
    # Prepare the prompt with data and user question
    data_str = data.to_dict(orient='records')
    column_names = list(data.columns)
    num_records = len(data)
    data_types = summarize_data_types(data)

    prompt = f"""
    You are an intelligent chatbot designed to analyze data and provide insights. 
    The data provided contains {num_records} records with the following columns and their data types:
    {', '.join([f"{col} (type: {data_types[col]})" for col in column_names])}.
    
    Here are the details of the data:
    {data_str}

    Please respond to the following question with only relevant insights, analysis, or information directly related to the question: "{input_text}"
    Ensure that your response is clear, concise, and focused on the specific aspects of the data related to the question.
    """
    
    # Generate the response from the Ollama model
    response = ollama.chat(model='data_processor', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    response_text = response['message']['content']
    urls = extract_urls(response_text)
    return response_text, urls

# Function to get a general response when no data is provided
def get_general_response(input_text):
    prompt = f"""
    You are an intelligent chatbot designed to answer general questions and provide relevant information. 
    Please respond to the following query: "{input_text}"
    """
    
    response = ollama.chat(model='data_processor', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    response_text = response['message']['content']
    urls = extract_urls(response_text)
    return response_text, urls

# Function to train or fine-tune the model
def train_model(combined_data):
    # Placeholder function for model training
    # Implement training logic based on your model and requirements
    pass

# Streamlit configuration
st.set_page_config(page_title="FileBOT - Intelligent Data Chatbot", page_icon='🤖', layout='wide')

# Header
st.title("📊 FileBOT - Intelligent Data Chatbot")

# Sidebar for file uploads
st.sidebar.header("Upload Your Files")
uploaded_files = st.sidebar.file_uploader("Upload CSV, XLSX, PDF, or TXT files", type=["csv", "xlsx", "pdf", "txt"], accept_multiple_files=True)

# Data processing and display
data = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = read_data(uploaded_file)
        if df is not None:
            data.append(df)
        else:
            st.sidebar.error(f"Could not read file: {uploaded_file.name}")

    if data:
        combined_data = combine_dataframes(data)
        st.write("### Data Preview")
        st.dataframe(combined_data.head())

        # Train or fine-tune the model with the combined data
        if st.button("Train Model"):
            train_model(combined_data)
            st.success("Model trained successfully!")

# Input field for user question
st.write("### Ask a Question About Your Data or General Queries")
input_text = st.text_input("Enter your question here:")

# Handling submission
if st.button("Get Response"):
    if input_text:
        if data:
            # Generate response and extract URLs based on uploaded data
            generated_response, urls = getOllamaResponse(combined_data, input_text)
        else:
            # Generate a general response if no data is uploaded
            generated_response, urls = get_general_response(input_text)
        
        # Display chatbot response
        st.markdown("### Chatbot Response:")
        st.write(generated_response)
        
        # Display extracted URLs if any
        if urls:
            st.markdown("### Related Links:")
            for url in urls:
                st.markdown(f"- [Link]({url})")
        
        # Check for location queries and display map if necessary
        if is_location_query(input_text):
            map_ = get_map(input_text)
            st.markdown("### Map Representation:")
            folium_static(map_)
    else:
        st.warning("Please enter a question to get a response.")
