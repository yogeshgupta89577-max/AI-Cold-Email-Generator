import pandas as pd
import chromadb
import uuid


class Portfolio:

    def __init__(self, file_path="my_portfolio.csv"):

        self.file_path = file_path

        self.data = pd.read_csv(file_path)

        self.chroma_client = chromadb.PersistentClient(path="vectorstore")

        self.collection = self.chroma_client.get_or_create_collection(
            name="portfolio"
        )

    def load_portfolio(self):

        # Load only once
        if self.collection.count() == 0:

            for _, row in self.data.iterrows():

                techstack = row.get("Techstack", "")
                links = row.get("Links", "")

                # Handle NaN / int / float safely
                techstack = str(techstack) if pd.notna(techstack) else ""
                links = str(links) if pd.notna(links) else ""

                # Skip empty rows
                if not techstack.strip():
                    continue

                self.collection.add(
                    documents=[techstack],
                    metadatas=[{"links": links}],
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):

        # Ensure proper format
        if isinstance(skills, str):
            skills = [skills]

        elif isinstance(skills, int):
            skills = [str(skills)]

        elif isinstance(skills, list):
            skills = [str(skill) for skill in skills]

        else:
            skills = []

        # Prevent empty query
        if not skills:
            return []

        results = self.collection.query(
            query_texts=skills,
            n_results=2
        )

        return results.get("metadatas", [])