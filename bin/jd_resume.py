#get reesumes from cloud
#todo add handling for docs vs pdf
#decide the path resume/data

from OCR_Reader import CVsReader
from ChatGPT_Pipeline import CVsInfoExtractor
import sys

from google.cloud import storage
from pathlib import Path
import logging
from urllib.parse import urlparse
import uuid
from utils import *


logging.basicConfig(level=logging.INFO)



def get_docs_from_gcp(inputPath):


    bucket_name =  urlparse(inputPath).path.split('/')[-1].strip()
    logging.info(f'{bucket_name=}')


    path_to_private_key = 'var/fifth.json'
    client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)
    bucket = storage.Bucket(client, bucket_name)
    str_folder_name_on_gcs = 'RESUME/'
    local_resume_path='./var/data'

    # Create the directory locally
    Path(f'{local_resume_path}/RESUME').mkdir(parents=True, exist_ok=True)

    blobs = bucket.list_blobs(prefix=str_folder_name_on_gcs)
    for blob in blobs:
        if not blob.name.endswith('/'):
            # This blob is not a directory!
            print(f'Downloading file [{blob.name}]')
            logging.info(f'{local_resume_path}/{blob.name}')
            blob.download_to_filename(f'{local_resume_path}/{blob.name}')


def normalise_cvs(cvs_directory_path_arg):

    
    cvs_reader = CVsReader(cvs_directory_path = cvs_directory_path_arg)
    cvs_content_df = cvs_reader.read_cv()
    cvs_info_extractor = CVsInfoExtractor(cvs_df = cvs_content_df, openai_api_key = 'openai_api_key_arg', desired_positions = 'desired_positions')
    extract_cv_info_dfs = cvs_info_extractor.extract_cv_info()

    




def jd_resume(job_description,category,document_count,inputPath):
    

    local_resume_path='./var/data/resume'
    pine_cone_api_key='c2041025-efd8-4c5b-9ebd-783ac3305c8a'
    pine_cone_index='testindex'
    Path(local_resume_path).mkdir(parents=True, exist_ok=True)
    

    #inputPath= "https://console.cloud.google.com/storage/browser/hackathontestdata2024 " 
    get_docs_from_gcp(inputPath)
    normalise_cvs(local_resume_path)

    session_id=uuid.uuid4().hex
    embeddings=create_embeddings_load_data()
    final_docs_list=create_docs(session_id)


    logging.info(final_docs_list)
    
    push_to_pinecone(pine_cone_api_key,pine_cone_index,embeddings,final_docs_list)
    relavant_docs=similar_docs(job_description,document_count,pine_cone_api_key,pine_cone_index,embeddings,session_id)

    dictToReturn={}
    if len(relavant_docs)>0:
        dictToReturn['status']='success'
    else:
        dictToReturn['status']='failure'

    dictToReturn['count']=len(relavant_docs) if relavant_docs is not None else 0
    dictToReturn['metadata']={}
    dictToReturn['metadata']['confidenceScore']=1
    dictToReturn['results']=relavant_docs

    return dictToReturn 




if __name__=='__main__':
    jd_resume('Loan','',3,'')
