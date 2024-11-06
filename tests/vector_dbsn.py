import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import chromadb
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel,Field
from typing import List,Annotated
import numpy as np
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from dotenv import load_dotenv
from collections import defaultdict
from utils.utils import  sanitize_string
## load data
from langchain.chains.summarize import load_summarize_chain
import json
from langchain import PromptTemplate


load_dotenv()

PREPROCESS=True  # to calculate embeddings //needs only to be done once else set false
MODEL_NAME="gpt-4o"
DATA_PATH="./data/transcripts.json"
# Open the JSON file and load its content
with open(DATA_PATH, 'r') as file:
    data = json.load(file)


data_list=[ sanitize_string(d['Unterhaltung']) for d in data]


#///////////////////VECTOR DATABASE aber OPTIONAL FÜR DIESEN USECASE FYI /////////////////////////////////////////////
    # Split documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # make sure each document is iin a split
    chunk_sources = []
    chunks = []
    ct=0
    for doc in data_list:

        for chunk in text_splitter.split_text(doc):
            chunks.append(sanitize_string(chunk))
            chunk_sources.append(f"chunk_{ct}")

    df=pd.DataFrame(data={'chunk':chunks,'chunk_src':chunk_sources})





class VectorStoreRetriever:
    def __init__(self, vectors, oai_client, df: pd.DataFrame):
        self._arr = vectors
        self._client = oai_client
        self.df = df

    @classmethod
    def from_docs(cls, df):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Use PersistentClient instead of EphemeralClient
        persistent_client = chromadb.PersistentClient(path="./dbs")

        openai_lc_client = Chroma.from_texts(
            df.chunk.tolist(),
            embeddings,
            client=persistent_client,
            collection_name="openai_collection"
        )

        vecs = openai_lc_client._collection.get(include=['embeddings', 'documents'])
        return cls(vecs, openai_lc_client, df)

    @classmethod
    def load(cls):
        persistent_client = chromadb.PersistentClient(path="./dbs")
        collection = persistent_client.get_collection("openai_collection")

        vecs = collection.get(include=['embeddings', 'documents'])

        # Reconstruct df from saved data
        df = pd.DataFrame({
            'chunk': vecs['documents'],
            'chunk_src': [vecs['documents'][i] for i in range(len(vecs['documents']))]  # You might want to adjust this
        })

        openai_lc_client = Chroma(
            client=persistent_client,
            collection_name="openai_collection",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
        )

        return cls(vecs, openai_lc_client, df)

    def query(self, query: str, k: int = 5) -> pd.DataFrame:
        store = defaultdict(list)
        results_with_scores = self._client.similarity_search_with_score(query, k=k)

        for doc, score in results_with_scores:
            doc_index = np.argwhere(self.df.chunk.values.reshape(-1) == doc.page_content)[0]
            store['files'].append(self.df.iloc[doc_index].chunk_src.squeeze())
            store['scores'].append(score)
            store['content'].append(doc.page_content)
            store['embeddings'].append(
                self._arr['embeddings'][np.argwhere(np.array(self._arr['documents']) == doc.page_content)[0][0]]
            )

        return pd.DataFrame(store)




if PREPROCESS:
    retriever = VectorStoreRetriever.from_docs(df) # wenn lokal noch keine vector db erstellt wurde


retriever = VectorStoreRetriever.load() # wenn vector db bereits esxisterit das heißt 2ter run und so



print(retriever.query("was ist das häufigst thema ")) # vector datenbank wird aber hier eigentlich nicht benötigt returned t5  textchunks bezogen auf die frage


#///////////////////RELEVANT/////////////////////////////////////////////
if MODEL_NAME=="gpt-4o":
    llm = ChatOpenAI(temperature=0,   model=MODEL_NAME,max_tokens=None,streaming=True)
elif MODEL_NAME=='claude-3-5-sonnet':
    llm= ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2)


# die beiden PROMPT templates müssen angepasstwerden um die zusammenfassung so zu steuern dass man die top 10 was auch immer bekommt
# "was ist das top topic
map_prompt = """
Schreibe eine prägnante Zusammenfassung des angegebenen Textes.
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])


combine_prompt = """
Schreibe eine prägnante Zusammenfassung des angegebenen Textes.
Nutze Aufzählungspunkte, um die wichtigsten Informationen hervorzuheben.
```{text}```
BULLET POINT SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=500)

docs = text_splitter.create_documents(['\n'.join(data_list)])
summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
#                                      verbose=True
                                    )
output = summary_chain.run(docs)
print(output)