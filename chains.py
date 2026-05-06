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
            model_name="llama-3.3-70b-versatile",
            max_tokens=1024
        )

    def extract_jobs(self, cleaned_text):

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:

            The scraped text is from the careers page of a website.

            Extract the job postings and return ONLY valid JSON.

            The JSON should contain:
            - role
            - experience
            - skills
            - description

            Do not include markdown, explanation, or backticks.

            ### VALID JSON:
            """
        )

        chain_extract = prompt_extract | self.llm

        res = chain_extract.invoke({
            "page_data": cleaned_text
        })

        try:

            content = res.content.strip()

            if not content:
                raise OutputParserException(
                    "Empty response from LLM"
                )

            json_parser = JsonOutputParser()

            res = json_parser.parse(content)

        except OutputParserException:

            raise OutputParserException(
                "Context too big. Unable to parse jobs."
            )

        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):

        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:

        You are an enthusiastic job applicant.

        Write a professional job application email for the above job.

        The email should:
        - Express interest in the role
        - Highlight relevant skills
        - Mention enthusiasm to learn and contribute
        - Be concise and professional

        Also include relevant portfolio links if available:
        {link_list}

        Do not include any explanation or preamble.

        ### EMAIL:
        """
    )

        chain_email = prompt_email | self.llm

        res = chain_email.invoke({
            "job_description": str(job),
            "link_list": str(links)
        })

        return res.content


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))