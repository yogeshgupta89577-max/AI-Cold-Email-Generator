import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
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
            Return ONLY a valid JSON list.
            ### JSON ONLY (NO PREAMBLE):
            """
        )
        
        chain = prompt_extract | self.llm
        res = chain.invoke(input={"page_data": cleaned_text})
        
        try:
            # Automatic parsing using LangChain's internal logic
            parser = JsonOutputParser()
            return parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big or unable to parse jobs.")

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Your name, an aspiring Data Scientist and Software Developer.
            Write a highly personalized cold email to the hiring manager.
            - Focus on how your skills (Python, ML, etc.) solve the job requirements.
            - Mention these specific portfolio projects/links: {link_list}
            - Maintain a professional yet enthusiastic tone.
            - Keep it under 200 words.

            ### EMAIL ONLY:
            """
        )
        chain = prompt_email | self.llm
        res = chain.invoke({"job_description": str(job), "link_list": links})
        return res.content