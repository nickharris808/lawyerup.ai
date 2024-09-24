# AI-Generated Legal Report Generator

This project is a Streamlit application that generates AI-powered legal reports. The application integrates with various APIs, including OpenAI, Stripe, Google Scholar, and MongoDB, to provide a comprehensive and automated solution for generating legal reports.

## Prerequisites

Before deploying the application, ensure you have the following:

1. **Python 3.7+** installed on your machine.
2. **Streamlit** installed (`pip install streamlit`).
3. **API keys** for OpenAI, Stripe, Google Scholar, and Google Search.
4. **MongoDB** connection URI.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/legal-report-generator.git
    cd legal-report-generator
    ```

2. **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

1. **Create a `.streamlit/secrets.toml` file:**

    This file will store your API keys and other secrets. Here is an example configuration:

    ```toml:.streamlit/secrets.toml
    OPENAI_API_KEY = "your_openai_api_key"
    STRIPE_SECRET_KEY = "your_stripe_secret_key"
    STRIPE_PUBLISHABLE_KEY = "your_stripe_publishable_key"
    GOOGLE_SCHOLAR_API_KEY = "your_google_scholar_api_key"
    GOOGLE_SEARCH_API_KEY = "your_google_search_api_key"
    GOOGLE_SEARCH_ENGINE_ID = "your_google_search_engine_id"
    MONGODB_URI = "your_mongodb_connection_uri"  # e.g., mongodb+srv://<username>:<password>@cluster0.mongodb.net/mydatabase?retryWrites=true&w=majority
    BASE_URL = "http://localhost:8501"  # Change to your deployed app's URL if necessary
    ```

2. **Ensure the following files are in place:**
    - `project_root/templates/report_template.html`
    - `project_root/schemas/json_schema.json`

## Running the Application

1. **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

2. **Access the application:**
    Open your web browser and navigate to `http://localhost:8501`.

## Deployment

To deploy the application, you can use any cloud service that supports Python applications, such as Heroku, AWS, or Google Cloud. Below are the general steps for deploying on Heroku:

1. **Create a `Procfile` in the root directory:**
    ```sh
    echo "web: streamlit run app.py" > Procfile
    ```

2. **Create a `requirements.txt` file if not already present:**
    ```sh
    pip freeze > requirements.txt
    ```

3. **Login to Heroku and create a new app:**
    ```sh
    heroku login
    heroku create your-app-name
    ```

4. **Deploy the application:**
    ```sh
    git add .
    git commit -m "Initial commit"
    git push heroku master
    ```

5. **Set the environment variables on Heroku:**
    ```sh
    heroku config:set OPENAI_API_KEY=your_openai_api_key
    heroku config:set STRIPE_SECRET_KEY=your_stripe_secret_key
    heroku config:set STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key
    heroku config:set GOOGLE_SCHOLAR_API_KEY=your_google_scholar_api_key
    heroku config:set GOOGLE_SEARCH_API_KEY=your_google_search_api_key
    heroku config:set GOOGLE_SEARCH_ENGINE_ID=your_google_search_engine_id
    heroku config:set MONGODB_URI=your_mongodb_connection_uri
    heroku config:set BASE_URL=https://your-app-name.herokuapp.com
    ```

6. **Open the deployed application:**
    ```sh
    heroku open
    ```

## Usage

1. **Fill in the client information and incident details.**
2. **Upload supporting documents and photos.**
3. **Review the information and proceed to payment.**
4. **After successful payment, the AI-generated legal report will be displayed and available for download.**

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

This README provides a comprehensive guide to deploying and running the AI-Generated Legal Report Generator application. If you have any questions or need further assistance, please refer to the documentation or contact the project maintainers.
