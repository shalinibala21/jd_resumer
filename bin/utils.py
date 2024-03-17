import openai
import pandas as pd
#from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.llms import OpenAI
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
import os
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import time

#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(session_id):
    docs=[]
    df=pd.read_csv('var/CVs_Info_Extracted.csv')
    print(df.shape)
    for index, row in df.iterrows():

        print('adding  for {index}')

        
        chunks=row['All_Info_JSON']
        filename=row['CV_Filename']


        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename,"id":"","session_id":session_id},
        ))

    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    logging.info(f'printing embeddings')
    logging.info(embeddings)

    return embeddings





# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary

# Function to push data to Vector Store - Pinecone
#push_to_pinecone(pine_cone_api_key,pine_cone_index,embeddings,final_docs_list)

def push_to_pinecone(pinecone_apikey, pine_cone_index, embeddings, docs):

    

    from pinecone import Pinecone,PodSpec
    os.environ['PINECONE_API_KEY'] = pinecone_apikey
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    pc = Pinecone(api_key=api_key)

    #from pinecone import ServerlessSpec, PodSpec
    #spec = PodSpec(environment=environment)




    pc.delete_index(pine_cone_index)
    pc.create_index(
    name="testindex",
    dimension=384,
    metric="cosine",
    spec=PodSpec(
        environment="gcp-starter"
    )
    )

    index = pc.Index(pine_cone_index)









    logging.info(index.describe_index_stats())






    from langchain.vectorstores import Pinecone as PineconeVectorStore
    PineconeVectorStore.from_documents(docs, embeddings, index_name=pine_cone_index)

    logging.info(index.describe_index_stats())
    logging.info(type(index.describe_index_stats()))
    time.sleep(30)

    





# Function to get relevant documents from Vector Store - Pinecone based on user input
#similar_docs(job_description,document_count,pine_cone_api_key,pine_cone_index,embeddings,session_id)
def similar_docs(query, k, pinecone_apikey, pinecone_index_name, embeddings, session_id):
    
    import time

    from pinecone import Pinecone
    os.environ['PINECONE_API_KEY'] = pinecone_apikey
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    pc = Pinecone(api_key=api_key)

    #insert index name
 
    index = pc.Index(pinecone_index_name)
    logging.info(index.describe_index_stats())

    text_field = "text"

    from langchain.vectorstores import Pinecone
    vectorstore = Pinecone(index, embeddings, text_field)




    similar_docs=vectorstore.similarity_search_with_relevance_scores(query, int(k))
    #similar_docs = vectorstore.similarity_search_with_score(query, int(k),{"session_id":session_id})
    logging.info('*****SIMILAR DOCS*****')
    logging.info(similar_docs)

    results=[]
    for item in range(len(similar_docs)):

        tuple_result=similar_docs[item]
        match_score=tuple_result[1]

        doc=vars(similar_docs[item][0])
        meta_data=doc['metadata']
        file=meta_data['name']


        doc_dict={'score':match_score,'name':file}

        results.append(doc_dict)

    print(results[:k])
    return results