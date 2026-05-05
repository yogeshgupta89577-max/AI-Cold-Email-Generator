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
        .stApp { background-color: #f4f4f9; }
        .header-container { text-align: center; padding: 2rem 0; }
        .main-header { 
            font-size: 3.5rem !important; 
            font-weight: 800; 
            background: linear-gradient(-45deg, #4F46E5, #9333EA, #2563EB, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stButton>button {
            border-radius: 10px;
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            color: white;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header-container"><h1 class="main-header">📧 AI Cold Email Generator</h1></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;">Craft professional emails in seconds using Llama 3.3</p>', unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 4, 1])

    with col2:
        url_input = st.text_input("🔗 Job URL or Job Description:", placeholder="Paste link or text here...")
        submit_button = st.button("Generate Cold Email ✨")

        if submit_button:
            if not url_input:
                st.error("Please provide a URL or job text.")
                return

            with st.spinner("Processing..."):
                try:
                    # STEP 1: Load & Scraping (Fix for empty data)
                    if url_input.startswith("http"):
                        # User-Agent header add kiya hai taaki sites block na karein
                        loader = WebBaseLoader([url_input])
                        loader.requests_kwargs = {'headers': {'User-Agent': 'Mozilla/5.0'}}
                        data = loader.load()
                        
                        if not data or not data[0].page_content:
                            st.error("Could not read website content. Please paste the job text directly.")
                            return
                        raw_data = data[0].page_content
                    else:
                        raw_data = url_input

                    # STEP 2: Clean Text
                    cleaned_data = clean_text_func(raw_data)

                    # STEP 3: Load Portfolio
                    portfolio.load_portfolio()

                    # STEP 4: Extract Jobs (Ensuring it's a list)
                    jobs = llm.extract_jobs(cleaned_data)

                    # Error Check: Agar 'int' ya non-list return hua ho
                    if not isinstance(jobs, list) or len(jobs) == 0:
                        st.warning("No job details could be parsed. Check the URL or paste text.")
                        return

                    job = jobs[0] 

                    # STEP 5: Generate Email (Fix for skills data type)
                    skills = job.get('skills', [])
                    
                    # Agar skills string format mein hain (e.g. "Python, Java"), list mein convert karein
                    if isinstance(skills, str):
                        skills = [s.strip() for s in skills.split(',')]
                    
                    # Safety check for ChromaDB query
                    links = []
                    if skills and isinstance(skills, list):
                        links = portfolio.query_links(skills)

                    email = llm.write_mail(job, links)

                    # STEP 6: Display Result
                    st.markdown("---")
                    st.subheader(f"🎯 Targeted Role: {job.get('role', 'N/A')}")
                    
                    st.text_area(label="Your Generated Email:", value=email, height=400)

                    st.download_button(
                        "📥 Download Email",
                        data=email,
                        file_name="cold_email.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Error detail: {e}")
                    st.info("Tip: Try pasting the Job Description text directly instead of the URL.")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio, clean_text)