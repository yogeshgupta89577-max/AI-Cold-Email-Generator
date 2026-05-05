import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text_func):
    st.set_page_config(page_title="AI Cold Email Generator", page_icon="📧", layout="wide")

    # ---------- UI ENHANCEMENTS ----------
    st.markdown("""
        <style>
        /* Main App Background */
        .stApp {
            background-color: #f4f4f9; /* Soft Grey Background */
        }

        /* Center and Animate Gradient Heading */
        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .header-container {
            text-align: center;
            padding: 2rem 0;
        }

        .main-header { 
            font-size: 4rem !important; /* Increased size */
            font-weight: 800; 
            background: linear-gradient(-45deg, #4F46E5, #9333EA, #2563EB, #06B6D4);
            background-size: 400% 400%;
            animation: gradient-animation 5s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            margin-bottom: 0px;
        }

        .sub-header { 
            font-size: 1.25rem; 
            color: #111827; 
            text-align: center;
            margin-top: -10px;
            margin-bottom: 0rem;
            font-weight: 600;
        }

        /* Input Styling */
        .stTextInput > div > div > input {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #d1d5db;
        }

        /* Button Styling */
        .stButton>button {
            border-radius: 10px;
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            color: white;
            font-weight: bold;
            padding: 0.6rem 2rem;
            border: none;
            transition: transform 0.2s;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
            color: white;
            border: none;
        }

        /* Container for results */
        .result-box {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------- HEADER SECTION ----------
    st.markdown('<div class="header-container"><h1 class="main-header">📧 AI Cold Email Generator</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Craft professional emails in seconds using Llama 3.3</p>', unsafe_allow_html=True)

    # ---------- MAIN CONTENT ----------
    # Centering the input columns
    _, col2, _ = st.columns([1, 4, 1])

    with col2:
        url_input = st.text_input("🔗 Job URL or Job Description:", placeholder="Paste link or text here...")
        submit_button = st.button("Generate Cold Email ✨")

        if submit_button:
            if not url_input:
                st.error("Please provide a URL or job text.")
                return

            with st.spinner("Processing your professional email..."):
                try:
                    # STEP 1: Load Data
                    if url_input.startswith("http"):
                        loader = WebBaseLoader([url_input])
                        data = loader.load()
                        if not data:
                            st.error("Failed to load content from URL.")
                            return
                        raw_data = data[0].page_content
                    else:
                        raw_data = url_input

                    # STEP 2: Clean
                    cleaned_data = clean_text_func(raw_data)

                    # STEP 3: Load Portfolio
                    portfolio.load_portfolio()

                    # STEP 4: Extract Jobs
                    jobs = llm.extract_jobs(cleaned_data)

                    if not jobs:
                        st.warning("No job details found. Try another input.")
                        return

                    job = jobs[0] # Focus on the first job

                    # STEP 5: Generate Email
                    skills = job.get('skills', [])
                    links = portfolio.query_links(skills) if skills else []
                    email = llm.write_mail(job, links)

                    # STEP 6: Display Result
                    st.markdown("---")
                    st.subheader(f"🎯 Targeted Role: {job.get('role', 'N/A')}")
                    
                    st.text_area(
                        label="Your Generated Email:",
                        value=email,
                        height=400
                    )

                    st.download_button(
                        "📥 Download Email",
                        data=email,
                        file_name="cold_email.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)