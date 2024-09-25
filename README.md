### AI-Generated Legal Report Generator

This project is a Streamlit application that generates AI-powered legal reports. The application integrates various services such as OpenAI, Stripe, Google APIs, and MongoDB to provide a seamless experience for users to input case details, upload supporting documents and photos, and receive a comprehensive legal report.

#### Features

1. **Multi-step Form**: Collects client information, incident details, estimated compensation, supporting documents, and client questions.
2. **AI-Powered Analysis**: Uses OpenAI's GPT-4 to analyze images and summarize documents.
3. **Research Integration**: Searches Google Scholar and Google for relevant case laws and medical literature.
4. **Report Generation**: Compiles all collected data into a structured legal report.
5. **Payment Integration**: Uses Stripe for payment processing.
6. **MongoDB Storage**: Stores reports and uploaded files in MongoDB using GridFS.

#### Prerequisites

- Python 3.7+
- Streamlit
- OpenAI API Key
- Stripe API Key
- Google API Keys (Scholar and Search)
- MongoDB instance

#### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/legal-report-generator.git
    cd legal-report-generator
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up your `secrets.toml` file in the `.streamlit` directory with your API keys and MongoDB URI:
    ```toml
    OPENAI_API_KEY = "your_openai_api_key"
    STRIPE_SECRET_KEY = "your_stripe_secret_key"
    STRIPE_PUBLISHABLE_KEY = "your_stripe_publishable_key"
    GOOGLE_SCHOLAR_API_KEY = "your_google_scholar_api_key"
    GOOGLE_SEARCH_API_KEY = "your_google_search_api_key"
    GOOGLE_SEARCH_ENGINE_ID = "your_google_search_engine_id"
    MONGODB_URI = "your_mongodb_connection_uri"
    BASE_URL = "http://localhost:8501"
    ```

4. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

#### Usage

1. **Step 1: Client Information**
    - Enter the client's name, incident date, contact information, and confidentiality notice.
    - Click "Next" to proceed.

2. **Step 2: Incident Overview**
    - Provide a detailed description of the incident.
    - Click "Next" to proceed.

3. **Step 3: Estimated Compensation**
    - Enter the estimated economic, non-economic, and punitive damages.
    - Click "Next" to proceed.

4. **Step 4: Upload Supporting Documents and Photos**
    - Upload relevant documents (PDFs, DOCX) and photos (JPG, PNG).
    - Click "Next" to proceed.

5. **Step 5: Client Questions**
    - Enter any questions the client has and provide answers if available.
    - Click "Add Question" to save the question.
    - Click "Proceed to Payment" to continue.

6. **Step 6: Review and Payment**
    - Review all the entered information.
    - Click "Proceed to Payment" to create a Stripe checkout session.
    - Complete the payment to generate the report.

7. **Report Generation**
    - After successful payment, the app generates the legal report using AI and stores it in MongoDB.
    - The report is displayed on the screen.

#### Code Explanation

The main code is located in `app.py`. Here is a breakdown of its functionality:

1. **Imports and Configuration**:
    - Imports necessary libraries and initializes API keys and MongoDB connection.
    - Defines success and cancel URLs for Stripe payments.

    
```1:31:project_root/app.py
# app.py

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from io import BytesIO
from weasyprint import HTML
from datetime import datetime
import stripe
import openai
import os
import base64
from PIL import Image
from serpapi import GoogleSearch
import pymongo
import gridfs
from bson.objectid import ObjectId
from jinja2 import Template

# ---------------------------- Configuration ----------------------------

# Initialize OpenAI and Stripe with secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]
STRIPE_PUBLISHABLE_KEY = st.secrets["STRIPE_PUBLISHABLE_KEY"]

# Google APIs
GOOGLE_SCHOLAR_API_KEY = st.secrets.get("GOOGLE_SCHOLAR_API_KEY", "")
GOOGLE_SEARCH_API_KEY = st.secrets.get("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_ENGINE_ID = st.secrets.get("GOOGLE_SEARCH_ENGINE_ID", "")
```


2. **MongoDB Setup**:
    - Functions to get MongoDB client, database, and GridFS instance.

    
```33:51:project_root/app.py
# MongoDB Setup
def get_mongo_client():
    try:
        client = pymongo.MongoClient(st.secrets["MONGODB_URI"])
        return client
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

def get_db():
    client = get_mongo_client()
    if client:
        db = client.get_database("legal_reports_db")  # Change as needed
        return db
    return None
def get_gridfs(db):
    fs = gridfs.GridFS(db)
    return fs
```


3. **JSON Schema and HTML Template Loading**:
    - Functions to load JSON schema and HTML template for report generation.

    
```58:82:project_root/app.py
# ---------------------------- JSON Schema ----------------------------

def load_json_schema():
    try:
        with open("schemas/json_schema.json", "r") as f:
            schema = json.load(f)
        return schema
    except Exception as e:
        st.error(f"Error loading JSON schema: {e}")
        return None

json_schema = load_json_schema()

# ---------------------------- HTML Template ----------------------------

def load_html_template():
    try:
        with open("templates/report_template.html", "r") as f:
            template_str = f.read()
        return Template(template_str)
    except Exception as e:
        st.error(f"Error loading HTML template: {e}")
        return None

html_template = load_html_template()
```


4. **Helper Functions**:
    - Functions to upload and download files from MongoDB, create Stripe checkout session, process images, summarize documents, and search Google Scholar and Google.

    
```84:272:project_root/app.py
# ---------------------------- Helper Functions ----------------------------

def upload_file_to_mongodb(fs, file, metadata):
    """
    Uploads a file to MongoDB using GridFS with associated metadata.
    Returns the file_id as a string.
    """
    try:
        file_id = fs.put(file, filename=file.name, metadata=metadata)
        return str(file_id)
    except Exception as e:
        st.error(f"Error uploading file to MongoDB: {e}")
        return None
def upload_files_to_mongodb(fs, files, file_type, report_id):
    """
    Uploads a list of files to MongoDB using GridFS and associates them with a report.
    Returns a list of file_ids.
    """
    file_ids = []
    for file in files:
        metadata = {
            "report_id": report_id,
            "file_type": file_type,
            "uploaded_at": datetime.utcnow()
        }
        file_id = upload_file_to_mongodb(fs, file, metadata)
        if file_id:
            file_ids.append(file_id)
    return file_ids

def download_file_from_mongodb(fs, file_id):
    """
    Downloads a file from MongoDB using its file_id.
    Returns the file content in bytes.
    """
    try:
        file = fs.get(ObjectId(file_id))
        return file.read()
    except Exception as e:
        st.error(f"Error downloading file from MongoDB: {e}")
        return None

def create_checkout_session():
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'AI-Generated Legal Report',
                    },
                    'unit_amount': 199,  # $1.99 in cents
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
        )
        return checkout_session.url
    except Exception as e:
        st.error(f"Error creating Stripe Checkout Session: {e}")
        return None
def process_image(image, case_details):
    """
    Sends the image to OpenAI GPT-4 Vision API and returns the description.
    """
    try:
        # Convert image to bytes
        img = Image.open(image)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        # Create the prompt
        prompt = f"What's happening in this image in regards to this case: {case_details}"
        
        # Placeholder for OpenAI GPT-4 Vision API call
        # Replace with the actual API call as per OpenAI's documentation
        # For example purposes, we'll simulate the response
        response = openai.Image.create(
            image=img_bytes,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        description = response['data'][0]['text']
        return description.strip()
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return ""
def summarize_document(document, case_details):
    """
    Summarizes the uploaded document and extracts relevant information using OpenAI.
    """
    try:
        # Read the document content
        if document.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(document)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        elif document.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            import docx
            doc = docx.Document(document)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = document.read().decode("utf-8")
        
        # Create the prompt
        prompt = f"""
        Summarize the following document and extract relevant information related to the case details provided.
        
        **Case Details:**
        {case_details}
        
        **Document Content:**
        {text}
        
        Provide the summary and extracted information in a structured format.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            n=1,
            stop=None
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"Error summarizing document: {e}")
        return ""
def search_google_scholar(query):
    """
    Searches Google Scholar using SerpAPI and returns relevant case laws.
    """
    try:
        params = {
            "api_key": GOOGLE_SCHOLAR_API_KEY,
            "engine": "google_scholar",
            "q": query,
            "hl": "en",
            "num": 5
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        cases = []
        for result in results.get('organic_results', []):
            cases.append({
                "case_name": result.get('title'),
                "outcome": result.get('snippet'),
                "link": result.get('link')
            })
        return cases
    except Exception as e:
        st.error(f"Error searching Google Scholar: {e}")
        return []

def search_google(query):
    """
    Searches Google using SerpAPI and returns relevant medical literature.
    """
    try:
        params = {
            "api_key": GOOGLE_SEARCH_API_KEY,
            "engine": "google",
            "q": query,
            "hl": "en",
            "num": 5,
            "cx": GOOGLE_SEARCH_ENGINE_ID
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
```


5. **Main Application Logic**:
    - Initializes session state and handles multi-step form for collecting user inputs.
    - Generates AI workflows outputs, compiles the report, and saves it to MongoDB.
    - Handles payment success and cancellation.

    
```543:827:project_root/app.py
def main():
    st.set_page_config(page_title="AI-Generated Legal Report", layout="wide")
    st.title("AI-Generated Legal Report Generator")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'user_inputs' not in st.session_state:
        st.session_state.user_inputs = {}
    if 'document_files' not in st.session_state:
        st.session_state.document_files = []
    if 'photo_files' not in st.session_state:
        st.session_state.photo_files = []
    if 'report' not in st.session_state:
        st.session_state.report = None

    # Connect to MongoDB
    db = get_db()
    fs = get_gridfs(db) if db else None

    # Check if payment was successful
    query_params = st.experimental_get_query_params()
    if 'success' in query_params and query_params['success'][0] == 'true':
        st.success("Payment successful! Generating your legal report...")
        if st.session_state.report is None:
            with st.spinner("Generating report..."):
                # Generate AI Workflows Outputs
                case_details = f"Client Name: {st.session_state.user_inputs.get('client_name')}, Incident Date: {st.session_state.user_inputs.get('incident_date')}"
                
                # Image Analysis
                image_descriptions = analyze_uploaded_images(st.session_state.photo_files, case_details) if st.session_state.photo_files else []
                # Document Summarization
                document_summaries = summarize_uploaded_documents(st.session_state.document_files, case_details) if st.session_state.document_files else []
                
                # Research Agent
                case_laws, medical_literature = perform_research(case_details)
                
                # Generate Report
                report = generate_report(
                    st.session_state.user_inputs,
                    document_summaries,
                    image_descriptions,
                    case_laws,
                    medical_literature
                )
                st.session_state.report = report
                
                # Save Report to MongoDB
                if fs and report:
                    # Convert report JSON to bytes
                    report_json_bytes = json.dumps(report, indent=2).encode('utf-8')
                    report_pdf = generate_pdf(report)
                    
                    # Upload JSON Report
                    json_metadata = {
                        "report_type": "json",
                        "client_name": st.session_state.user_inputs.get('client_name'),
                        "incident_date": st.session_state.user_inputs.get('incident_date'),
                        "generated_at": datetime.utcnow()
                    }
                    json_report_id = upload_file_to_mongodb(fs, BytesIO(report_json_bytes), json_metadata)
                    
                    # Upload PDF Report
                    pdf_metadata = {
                        "report_type": "pdf",
                        "client_name": st.session_state.user_inputs.get('client_name'),
                        "incident_date": st.session_state.user_inputs.get('incident_date'),
                        "generated_at": datetime.utcnow()
                    }
                    pdf_report_id = upload_file_to_mongodb(fs, BytesIO(report_pdf), pdf_metadata)
                    
                    # Upload Original Documents
                    if st.session_state.document_files:
                        document_ids = upload_files_to_mongodb(fs, st.session_state.document_files, "document", json_report_id)
                        st.session_state.user_inputs['document_ids'] = document_ids
                        
                    # Upload Original Photos
                    if st.session_state.photo_files:
                        photo_ids = upload_files_to_mongodb(fs, st.session_state.photo_files, "image", json_report_id)
                        st.session_state.user_inputs['photo_ids'] = photo_ids
                        
                    # Save Report Metadata in a Separate Collection
                    reports_collection = db.get_collection("reports")
                    report_entry = {
                        "client_name": st.session_state.user_inputs.get('client_name'),
                        "incident_date": st.session_state.user_inputs.get('incident_date'),
                        "contact_info": st.session_state.user_inputs.get('contact_info', 'N/A'),
                        "incident_overview": st.session_state.user_inputs.get('incident_overview'),
                        "estimated_compensation": st.session_state.user_inputs.get('economic_damages', 0),
                        "non_economic_damages": st.session_state.user_inputs.get('non_economic_damages', 0),
                        "punitive_damages": st.session_state.user_inputs.get('punitive_damages', 0),
                        "document_ids": st.session_state.get('document_ids', []),
                        "photo_ids": st.session_state.get('photo_ids', []),
                        "report_json_id": json_report_id,
                        "report_pdf_id": pdf_report_id,
                        "generated_at": datetime.utcnow()
                    }
                    reports_collection.insert_one(report_entry)
                    
        if st.session_state.report:
            render_report(st.session_state.report)
        return
    elif 'canceled' in query_params and query_params['canceled'][0] == 'true':
        st.warning("Payment canceled. You can retry to generate the report.")

    # Multi-step form
    if st.session_state.step == 1:
        st.header("Step 1: Client Information")
        with st.form(key='step1_form'):
            client_name = st.text_input("Client Name", value=st.session_state.user_inputs.get('client_name', ''))
            incident_date = st.date_input("Incident Date", value=pd.to_datetime(st.session_state.user_inputs.get('incident_date', '2024-01-01')))
            contact_info = st.text_input("Contact Information (Optional)", value=st.session_state.user_inputs.get('contact_info', ''))
            confidentiality_notice = st.text_area("Confidentiality Notice", value=st.session_state.user_inputs.get('confidentiality_notice', ''))
            submit_step1 = st.form_submit_button("Next")

        if submit_step1:
            if not client_name or not incident_date:
                st.error("Please provide both Client Name and Incident Date.")
            else:
                st.session_state.user_inputs['client_name'] = client_name
                st.session_state.user_inputs['incident_date'] = incident_date.strftime('%Y-%m-%d')
                st.session_state.user_inputs['contact_info'] = contact_info
                st.session_state.user_inputs['confidentiality_notice'] = confidentiality_notice
                st.session_state.step = 2
                st.experimental_rerun()
    elif st.session_state.step == 2:
        st.header("Step 2: Incident Overview")
        with st.form(key='step2_form'):
            incident_overview = st.text_area("Describe what happened in detail:", value=st.session_state.user_inputs.get('incident_overview', ''))
            submit_step2 = st.form_submit_button("Next")

        if submit_step2:
            if not incident_overview:
                st.error("Please provide a detailed Incident Overview.")
            else:
                st.session_state.user_inputs['incident_overview'] = incident_overview
                st.session_state.step = 3
                st.experimental_rerun()

    elif st.session_state.step == 3:
        st.header("Step 3: Estimated Compensation")
        with st.form(key='step3_form'):
            economic_damages = st.number_input(
                "Economic Damages (e.g., Medical expenses, Lost wages, Property damage):",
                min_value=0.0,
                step=100.0,
                value=st.session_state.user_inputs.get('economic_damages', 0.0)
            )
            non_economic_damages = st.number_input(
                "Non-Economic Damages (e.g., Pain and suffering, Emotional distress):",
                min_value=0.0,
                step=100.0,
                value=st.session_state.user_inputs.get('non_economic_damages', 0.0)
            )
            punitive_damages = st.number_input(
                "Punitive Damages (if applicable):",
                min_value=0.0,
                step=100.0,
                value=st.session_state.user_inputs.get('punitive_damages', 0.0)
            )
            submit_step3 = st.form_submit_button("Next")

        if submit_step3:
            st.session_state.user_inputs['economic_damages'] = economic_damages
            st.session_state.user_inputs['non_economic_damages'] = non_economic_damages
            st.session_state.user_inputs['punitive_damages'] = punitive_damages
            st.session_state.step = 4
            st.experimental_rerun()
    elif st.session_state.step == 4:
        st.header("Step 4: Upload Supporting Documents and Photos")
        with st.form(key='step4_form'):
            documents = st.file_uploader("Upload Documents (e.g., PDFs, DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
            photos = st.file_uploader("Upload Photos (e.g., JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            submit_step4 = st.form_submit_button("Next")

        if submit_step4:
            # Store file objects
            if documents:
                st.session_state.document_files.extend(documents)
            if photos:
                st.session_state.photo_files.extend(photos)
            st.session_state.step = 5
            st.experimental_rerun()

    elif st.session_state.step == 5:
        st.header("Step 5: Client Questions")
        with st.form(key='step5_form'):
            client_question = st.text_input("Your Question:")
            client_answer = st.text_input("Answer (if any):")
            submit_step5 = st.form_submit_button("Add Question")

        if submit_step5:
            if client_question:
                if 'client_questions' not in st.session_state.user_inputs:
                    st.session_state.user_inputs['client_questions'] = []
                st.session_state.user_inputs['client_questions'].append({
                    "question": client_question,
                    "answer": client_answer
                })
                st.experimental_rerun()
            else:
                st.error("Please enter a question.")

        # Display existing questions
        if 'client_questions' in st.session_state.user_inputs:
            st.subheader("Your Questions:")
            for idx, qa in enumerate(st.session_state.user_inputs['client_questions'], 1):
                st.markdown(f"**Q{idx}:** {qa['question']}")
                if qa['answer']:
                    st.markdown(f"**A{idx}:** {qa['answer']}")
                st.markdown("---")

        # Navigation Buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back"):
                st.session_state.step = 4
                st.experimental_rerun()
        with col2:
            if st.button("Proceed to Payment"):
                st.session_state.step = 6
                st.experimental_rerun()
    elif st.session_state.step == 6:
        st.header("Step 6: Review and Payment")
        st.write("### Review Your Information")
        st.subheader("Client Information")
        st.write(f"**Name:** {st.session_state.user_inputs.get('client_name')}")
        st.write(f"**Incident Date:** {st.session_state.user_inputs.get('incident_date')}")
        st.write(f"**Contact Information:** {st.session_state.user_inputs.get('contact_info', 'N/A')}")

        st.subheader("Incident Overview")
        st.write(st.session_state.user_inputs.get('incident_overview'))

        st.subheader("Estimated Compensation")
        st.write(f"- **Economic Damages:** ${st.session_state.user_inputs.get('economic_damages', 0):,.2f}")
        st.write(f"- **Non-Economic Damages:** ${st.session_state.user_inputs.get('non_economic_damages', 0):,.2f}")
        st.write(f"- **Punitive Damages:** ${st.session_state.user_inputs.get('punitive_damages', 0):,.2f}")

        st.subheader("Supporting Documents and Photos")
        st.write("**Documents:**")
        if st.session_state.document_files:
            for doc in st.session_state.document_files:
                st.write(f"- {doc.name}")
        else:
            st.write("None")

        st.write("**Photos:**")
        if st.session_state.photo_files:
            for photo in st.session_state.photo_files:
                st.write(f"- {photo.name}")
        else:
            st.write("None")
        st.subheader("Client Questions")
        if 'client_questions' in st.session_state.user_inputs and st.session_state.user_inputs['client_questions']:
            for idx, qa in enumerate(st.session_state.user_inputs['client_questions'], 1):
                st.write(f"**Q{idx}:** {qa['question']}")
                if qa['answer']:
                    st.write(f"**A{idx}:** {qa['answer']}")
                st.markdown("---")
        else:
            st.write("No questions provided.")

        st.markdown("---")
        st.write("### Payment")
        st.write("To generate your legal report, please proceed with the payment of $1.99.")

        if st.button("Proceed to Payment"):
            # Create Stripe Checkout Session
            checkout_url = create_checkout_session()
            if checkout_url:
                st.markdown(f"[Click here to pay $1.99]({checkout_url})", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("After payment, you will be redirected here to view your report.")

    else:
        st.write("Unknown step.")
if __name__ == "__main__":
    main()
```


#### License

This project is licensed under the MIT License. See the LICENSE file for details.

#### Acknowledgements

- OpenAI for the GPT-4 API.
- Stripe for payment processing.
- Google APIs for research integration.
- MongoDB for data storage.

---

This README provides a comprehensive overview of the project, its features, installation steps, usage instructions, and a detailed explanation of the code.
