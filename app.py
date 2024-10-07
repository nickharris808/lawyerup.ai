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

# Define the Stripe Checkout Success and Cancel URLs
BASE_URL = st.secrets.get("BASE_URL", "http://localhost:8501")
SUCCESS_URL = f"{BASE_URL}/?success=true"
CANCEL_URL = f"{BASE_URL}/?canceled=true"

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
    Sends the image to OpenAI gpt-4o-mini-2024-07-18 Vision API and returns the description.
    """
    try:
        # Convert image to bytes
        img = Image.open(image)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        # Create the prompt
        prompt = f"What's happening in this image in regards to this case: {case_details}"
        
        # Placeholder for OpenAI gpt-4o-mini-2024-07-18 Vision API call
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
            model="gpt-4o-mini-2024-07-18",
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
        
        literature = []
        for result in results.get('organic_results', []):
            literature.append({
                "title": result.get('title'),
                "snippet": result.get('snippet'),
                "link": result.get('link')
            })
        return literature
    except Exception as e:
        st.error(f"Error searching Google: {e}")
        return []

def perform_research(case_details):
    """
    Searches for relevant case laws and medical literature using Google APIs.
    """
    # Define search queries based on case details
    case_law_query = f"legal cases similar to {case_details}"
    medical_literature_query = f"medical studies related to {case_details}"
    
    # Search Google Scholar for case laws
    case_laws = search_google_scholar(case_law_query) if GOOGLE_SCHOLAR_API_KEY else []
    
    # Search Google for medical literature
    medical_literature = search_google(medical_literature_query) if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID else []
    
    return case_laws, medical_literature

def generate_report(user_inputs, document_summaries, image_descriptions, case_laws, medical_literature):
    """
    Generates the legal report using OpenAI based on collected data and AI workflows.
    """
    try:
        # Compile all case details
        case_details = f"Client Name: {user_inputs.get('client_name')}, Incident Date: {user_inputs.get('incident_date')}"
    
        # Create the prompt for report generation
        prompt = f"""
        Using the provided data, generate a comprehensive legal report in JSON format that includes the following sections:
    
        **Client Information:**
        Name: {user_inputs.get('client_name')}
        Incident Date: {user_inputs.get('incident_date')}
        Contact Information: {user_inputs.get('contact_info', 'N/A')}
    
        **Incident Overview:**
        {user_inputs.get('incident_overview')}
    
        **Estimated Compensation:**
        - Economic Damages: ${user_inputs.get('economic_damages', 0)}
        - Non-Economic Damages: ${user_inputs.get('non_economic_damages', 0)}
        - Punitive Damages: ${user_inputs.get('punitive_damages', 0)}
    
        **Supporting Documents Summaries:**
        {json.dumps(document_summaries, indent=2) if document_summaries else 'None'}
    
        **Image Descriptions:**
        {json.dumps(image_descriptions, indent=2) if image_descriptions else 'None'}
    
        **Client Questions:**
        {json.dumps(user_inputs.get('client_questions', []), indent=2)}
    
        **Case Laws:**
        {json.dumps(case_laws, indent=2) if case_laws else 'None'}
    
        **Medical Literature:**
        {json.dumps(medical_literature, indent=2) if medical_literature else 'None'}
    
        Using this information, generate the following sections:
        - Key Findings
        - Applicable Laws and Statutes
        - Precedent Cases
        - Strength of Case
        - Legal Arguments
        - Action Plan Steps
        - Case Valuation
    
        Ensure the output is in valid JSON format adhering to the specified schema below.
        Provide only the JSON output without any additional text.
    
        ### JSON Schema:
        {json.dumps(json_schema, indent=2)}
        """
    
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=5000,  # Increased to accommodate detailed JSON
            n=1,
            stop=None
        )
    
        report_content = response.choices[0].message.content.strip()
    
        # Attempt to parse JSON
        report = json.loads(report_content)
        return report
    
    except json.JSONDecodeError as e:
        st.error("Failed to parse the generated report as JSON. Please try again.")
        st.write("### Raw Output:")
        st.text(report_content)
        return None
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None

def render_report(report):
    """
    Displays the generated report in a structured and readable format.
    """
    # Display the report in a readable format
    st.header("Your AI-Generated Legal Report")

    st.subheader("Client Information")
    st.write(f"**Name:** {report.get('client_name')}")
    st.write(f"**Incident Date:** {report.get('incident_date')}")
    st.write(f"**Contact Information:** {report.get('contact_info', 'N/A')}")

    st.subheader("Incident Overview")
    st.write(report.get('incident_overview'))

    st.subheader("Key Findings")
    for point in report.get('key_findings', []):
        st.markdown(f"- {point}")

    st.subheader("Estimated Compensation")
    compensation = report.get('estimated_compensation', {})
    st.write(f"- **Economic Damages:** ${compensation.get('economic_damages', 0):,.2f}")
    st.write(f"- **Non-Economic Damages:** ${compensation.get('non_economic_damages', 0):,.2f}")
    st.write(f"- **Punitive Damages:** ${compensation.get('punitive_damages', 0):,.2f}")

    # Compensation Breakdown Chart
    df_comp = pd.DataFrame({
        'Damage Type': ['Economic Damages', 'Non-Economic Damages', 'Punitive Damages'],
        'Amount': [
            compensation.get('economic_damages', 0),
            compensation.get('non_economic_damages', 0),
            compensation.get('punitive_damages', 0)
        ]
    })
    fig = px.pie(df_comp, names='Damage Type', values='Amount', title='Compensation Breakdown')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Applicable Laws and Statutes")
    st.write(report.get('applicable_laws'))

    st.subheader("Precedent Cases")
    for case in report.get('precedent_cases', []):
        st.markdown(f"**{case['case_name']}**: {case['outcome']}")

    st.subheader("Strength of Case")
    st.write(report.get('strength_of_case'))

    st.subheader("Legal Arguments")
    legal_args = report.get('legal_arguments', {})
    st.write(f"**Liability Assessment:** {legal_args.get('liability_assessment')}")
    st.write(f"**Negligence Elements:** {legal_args.get('negligence_elements')}")
    st.write(f"**Counterarguments:** {legal_args.get('counterarguments')}")

    st.subheader("Action Plan Steps")
    for step in report.get('action_plan_steps', []):
        st.markdown(f"- {step}")

    st.subheader("Case Valuation")
    case_val = report.get('case_valuation', {})
    st.write(f"**Valuation Methodology:** {case_val.get('valuation_methodology')}")
    st.write("**Factors Influencing Case Value:**")
    for factor in case_val.get('factors_influencing', []):
        st.markdown(f"- {factor}")

    # Download Report as JSON
    st.subheader("Download Your Report")
    report_json = json.dumps(report, indent=2)
    b64 = base64.b64encode(report_json.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="legal_report.json">Download Report as JSON</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Download Report as PDF
    if st.button("Download Report as PDF"):
        pdf = generate_pdf(report)
        st.download_button(
            label="Download Report as PDF",
            data=pdf,
            file_name='legal_report.pdf',
            mime='application/pdf'
        )

def generate_pdf(report):
    """
    Converts the generated report into a well-formatted PDF using WeasyPrint.
    """
    if not html_template:
        st.error("HTML template not loaded. Cannot generate PDF.")
        return None

    try:
        # Render HTML with Jinja2 template
        html_content = html_template.render(
            client_name=report.get('client_name'),
            incident_date=report.get('incident_date'),
            contact_info=report.get('contact_info', 'N/A'),
            incident_overview=report.get('incident_overview'),
            key_findings=report.get('key_findings', []),
            economic_damages=report.get('estimated_compensation', {}).get('economic_damages', 0),
            non_economic_damages=report.get('estimated_compensation', {}).get('non_economic_damages', 0),
            punitive_damages=report.get('estimated_compensation', {}).get('punitive_damages', 0),
            applicable_laws=report.get('applicable_laws'),
            precedent_cases=report.get('precedent_cases', []),
            strength_of_case=report.get('strength_of_case'),
            legal_arguments=report.get('legal_arguments', {}),
            action_plan_steps=report.get('action_plan_steps', []),
            case_valuation=report.get('case_valuation', {})
        )
        
        pdf = HTML(string=html_content).write_pdf()
        return pdf
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# ---------------------------- AI Workflows ----------------------------

def analyze_uploaded_images(images, case_details):
    """
    Processes each uploaded image using OpenAI's gpt-4o-mini-2024-07-18 Vision API.
    Returns a list of descriptions.
    """
    descriptions = []
    for image in images:
        description = process_image(image, case_details)
        if description:
            descriptions.append(description)
    return descriptions

def summarize_uploaded_documents(documents, case_details):
    """
    Processes each uploaded document using OpenAI's gpt-4o-mini-2024-07-18 API.
    Returns a list of summaries.
    """
    summaries = []
    for document in documents:
        summary = summarize_document(document, case_details)
        if summary:
            summaries.append(summary)
    return summaries

def perform_research(case_details):
    """
    Searches for relevant case laws and medical literature using Google APIs.
    Returns case laws and medical literature lists.
    """
    # Define search queries based on case details
    case_law_query = f"legal cases similar to {case_details}"
    medical_literature_query = f"medical studies related to {case_details}"
    
    # Search Google Scholar for case laws
    case_laws = search_google_scholar(case_law_query) if GOOGLE_SCHOLAR_API_KEY else []
    
    # Search Google for medical literature
    medical_literature = search_google(medical_literature_query) if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID else []
    
    return case_laws, medical_literature

# ---------------------------- Main Application ----------------------------

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
