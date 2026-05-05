import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        # API Key validation check
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from a careers page. 
            Extract job postings and return them in JSON format with the following keys:
            'role', 'experience', 'skills', and 'description'.
            Return ONLY a valid JSON list of objects. 
            ### JSON ONLY (NO PREAMBLE):
            """
        )
        
        chain = prompt_extract | self.llm
        res = chain.invoke(input={"page_data": cleaned_text})
        
        try:
            parser = JsonOutputParser()
            parsed_res = parser.parse(res.content)
            
            # --- CRITICAL FIX FOR 'int' error ---
            # Ensure result is always a list of dictionaries[cite: 1, 2]
            if isinstance(parsed_res, dict):
                return [parsed_res]
            elif isinstance(parsed_res, list):
                return parsed_res
            else:
                return [] # Agar result koi integer ya unexpected type ho
                
        except OutputParserException:
            # Context window issues handle karne ke liye empty list return karein
            return []

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Yogesh Gupta, an aspiring Data Scientist and Software Developer.
            Write a highly personalized cold email to the hiring manager.
            - Focus on how your skills (Python, ML, etc.) solve the job requirements.
            - Mention these specific portfolio projects/links: {link_list}
            - Maintain a professional yet enthusiastic tone.
            - Keep it under 200 words.

            ### EMAIL ONLY (NO SUBJECT LINE):
            """
        )
        # Ensure links is a string for the prompt
        link_str = ", ".join(links) if isinstance(links, list) else str(links)
        
        chain = prompt_email | self.llm
        res = chain.invoke({"job_description": str(job), "link_list": link_str})
        return res.content