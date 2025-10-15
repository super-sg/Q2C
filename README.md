EduRAG: Smart Education using RAG (Streamlit Edition)EduRAG is an AI-powered educational tool designed to provide fact-grounded answers to student queries directly from NCERT textbooks. This project leverages a Retrieval-Augmented Generation (RAG) pipeline to ensure that the information provided is accurate, relevant, and free from the noise of unverified internet sources.This version features a simple and interactive frontend built with Streamlit.How to Run This ProjectFollow these steps to get your EduRAG application running locally.Prerequisites:Python 3.8 or higherpip (Python package installer)Step 1: Set Up Your Project FolderCreate a new folder for your project (e.g., EduRAG_Streamlit).Inside this folder, create the four files I provided: streamlit_app.py, ingest.py, requirements.txt, and this README.md.Create a folder named data. This is where you will place your NCERT textbook PDFs.Your folder structure should look like this:EduRAG_Streamlit/
│
├── data/
│   └── (Your NCERT PDFs go here)
│
├── streamlit_app.py
├── ingest.py
├── requirements.txt
└── README.md
Step 2: Create a Virtual EnvironmentOpen your terminal or command prompt, navigate into your project folder, and run the following commands to create and activate a Python virtual environment.# Navigate to your project directory
cd EduRAG_Streamlit

# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
Step 3: Install DependenciesWith your virtual environment active, install all the necessary libraries from the requirements.txt file.pip install -r requirements.txt
Step 4: Add Your API KeyIn the main project folder (EduRAG_Streamlit), create a new file named .env.Open this file and add your Google Gemini API key like this:GEMINI_API_KEY="YOUR_API_KEY_HERE"
Step 5: Process Your TextbooksAdd your NCERT textbook PDF files into the data folder.Run the ingest.py script from your terminal. This will read the PDFs, process them, and create a faiss_index folder. You only need to do this once.python ingest.py
This step might take a few minutes depending on the size of your textbooks.Step 6: Run the Streamlit AppYou're all set! Start the application by running the following command in your terminal:streamlit run streamlit_app.py
Your web browser should automatically open with the EduRAG chat application running. You can now start asking questions!