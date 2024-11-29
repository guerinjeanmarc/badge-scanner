"""
Badge Scanner - Conference Badge Information Extractor
---------------------------------------------------
This application uses GPT-4o-mini to extract information from conference badge photos.
It provides a user-friendly interface for uploading photos, configuring extraction fields,
and downloading results in CSV format.
"""

from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image
import base64
import io
from typing import List, Dict
import tempfile

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Badge Scanner",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'extraction_fields' not in st.session_state:
    st.session_state.extraction_fields = [
        {'name': 'first_name', 'description': 'First name of the person'},
        {'name': 'last_name', 'description': 'Last name of the person'},
        {'name': 'company', 'description': 'Company name on the badge'}
    ]

if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

def encode_image(image: Image.Image) -> str:
    """
    Encode and optimize an image for API consumption.
    
    Args:
        image (PIL.Image): Input image to be processed
        
    Returns:
        str: Base64 encoded image string
    """
    img_copy = image.copy()
    
    # Resize image if it's too large (keeping aspect ratio)
    max_size = (800, 800)
    img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if in RGBA format
    if img_copy.mode == 'RGBA':
        img_copy = img_copy.convert('RGB')
    
    buffered = io.BytesIO()
    img_copy.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_badge(image_path: str, fields: List[Dict]) -> Dict:
    """Analyze a badge image and extract specified fields"""
    if not st.session_state.OPENAI_API_KEY:
        st.error("Please enter your OpenAI API key in the sidebar first!")
        st.stop()

    with Image.open(image_path) as img:
        base64_image = encode_image(img)
    
    chat = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=300, temperature=0)

    # Create the JSON schema based on dynamic fields
    properties = {
        field['name']: {
            "type": "string",
            "description": field['description']
        }
        for field in fields
    }
    
    json_schema = {
        "name": "analyze_badge",
        "description": "Extract information from a conference badge photo",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": [field['name'] for field in fields]
        }
    }

    structured_llm = chat.with_structured_output(json_schema)
    
    human_message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "Extract the following information from this conference badge photo. "
                        "If you're not sure about a field, return 'N/A'."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    )
    
    try:
        response = structured_llm.invoke([human_message])
        return response
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    """Main application function that sets up the Streamlit interface."""

    st.title("Conference Badge Information Extractor")
    
    # Sidebar for API key and field configuration
    with st.sidebar:

        st.header("OpenAI API Key")
        api_key = st.text_input(
            "Enter your OpenAI API key",
            type="password",  # This masks the API key
            help="Get your API key from https://platform.openai.com/api-keys",
            value=st.session_state.OPENAI_API_KEY if st.session_state.OPENAI_API_KEY else ""
        )
        
        if api_key:
            st.session_state.OPENAI_API_KEY = api_key
            os.environ['OPENAI_API_KEY'] = api_key

        st.header("Configure Extraction Fields")
        
        # Add new field button
        if st.button("Add New Field"):
            st.session_state.extraction_fields.append(
                {'name': '', 'description': ''}
            )
        
        # Display and edit fields
        updated_fields = []
        for i, field in enumerate(st.session_state.extraction_fields):
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                name = st.text_input(f"Field Name {i}", field['name'])
            with col2:
                desc = st.text_input(f"Description {i}", field['description'])
            with col3:
                if st.button("Remove", key=f"remove_{i}"):
                    continue
            updated_fields.append({'name': name, 'description': desc})
        
        st.session_state.extraction_fields = updated_fields

    # Main content
    st.header("Upload Badge Photos")
    uploaded_files = st.file_uploader(
        "Choose badge photos", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Extract Information"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, file in enumerate(uploaded_files):
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name

                status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {file.name}")
                
                # Process the image
                result = analyze_badge(tmp_path, st.session_state.extraction_fields)
                if result:
                    result['filename'] = file.name
                    results.append(result)
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))

            if results:
                # Convert results to DataFrame
                df = pd.DataFrame(results)
                
                # Display results
                st.header("Extracted Information")
                st.dataframe(df)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="badge_information.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
