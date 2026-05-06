import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):

    st.set_page_config(
        page_title="AI Cold Email Generator",
        page_icon="📧",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>

    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 55px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 20px;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #d1d5db;
        margin-bottom: 40px;
    }

    /* Input Box */
    .stTextInput > div > div > input {
        background-color: #1f2937;
        color: white;
        border-radius: 12px;
        border: 2px solid #4f46e5;
        padding: 12px;
        font-size: 16px;
    }
    ::placeholder {
    color: white !important;
    opacity: 0.7 !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(to right, #4f46e5, #9333ea);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px;
        border: none;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(to right, #4338ca, #7e22ce);
    }

    /* Output Box */
    .stCodeBlock {
        border-radius: 15px;
        border: 1px solid #4f46e5;
    }

    /* Card */
    .custom-card {
        background-color: rgba(255,255,255,0.08);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }

    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(
        """
        <div class="main-title">
            📧 AI Cold Email Generator
        </div>

        <div class="sub-title">
            Generate Professional AI-Powered Job Application Emails Instantly
        </div>
        """,
        unsafe_allow_html=True
    )

   
    url_input = st.text_input(
    "",
    placeholder="🔗 Paste Job URL"
)

    submit_button = st.button("🚀 Generate Email")

    if submit_button:

        if not url_input.strip():
            st.warning("Please enter a valid URL")
            return

        with st.spinner("Generating professional email..."):

            try:

                loader = WebBaseLoader([url_input])

                data = clean_text(
                    loader.load().pop().page_content
)

                # Limit text size
                data = data[:4000]

                portfolio.load_portfolio()

                jobs = llm.extract_jobs(data)

                for job in jobs:

                    skills = job.get('skills', [])

                    # Safe skill conversion
                    if isinstance(skills, str):
                        skills = [skills]

                    elif isinstance(skills, int):
                        skills = [str(skills)]

                    elif isinstance(skills, list):
                        skills = [str(skill) for skill in skills]

                    else:
                        skills = []

                    # Prevent empty query issue
                    if skills:
                        links = portfolio.query_links(skills)
                    else:
                        links = []

                    email = llm.write_mail(job, links)

                    st.success("✅ Email Generated Successfully")

                    st.code(email, language='markdown')

            except Exception as e:

                st.error(f"❌ Error: {e}")

 


if __name__ == "__main__":

    chain = Chain()

    portfolio = Portfolio()

    create_streamlit_app(
        chain,
        portfolio,
        clean_text
    )