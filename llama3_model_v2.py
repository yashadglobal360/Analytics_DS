import streamlit as st
from langchain_community.llms import Ollama
import pandas as pd
from pandasai import SmartDataframe

# Initialize the LLM
llm = Ollama(model="llama3")

# Streamlit app title
st.title("Data Analysis with PandasAI")

# File uploader
uploader_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploader_file is not None:
    # Read and display the data
    data = pd.read_csv(uploader_file)
    st.write(data.head(3))
    
    # Initialize SmartDataframe
    df = SmartDataframe(data, config={"llm": llm})
    
    # Text area for prompt
    prompt = st.text_area("Enter your prompt:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    # Get response from PandasAI
                    response = df.chat(prompt)
                    
                    # Print the response for debugging
                    st.write("Response from PandasAI:")
                    st.write(response)
                    
                    # Check if the response indicates a chart file
                    if isinstance(response, dict) and response.get('type') == 'plot':
                        chart_path = response.get('value')
                        if chart_path and os.path.isfile(chart_path):
                            st.image(chart_path, caption='Generated Chart')
                        else:
                            st.warning("Chart file not found.")
                    elif response and '```' in response:
                        st.code(response)
                    else:
                        st.warning("No valid code or chart path found in the response.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a prompt!")
