# import modules
import os
import pandas as pd
import openai
#from openai import InvalidRequestError
import os
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion,Choice,ChatCompletionMessage,CompletionUsage
import time
import json
from json import JSONDecodeError
from tqdm import tqdm
# add a progress bar to pandas operations
tqdm.pandas(desc='CVs')


import os
from openai import AzureOpenAI

# define the path to the output CSV file
output_csv_file_path = './var/CVs_Info_Extracted.csv'

# define the path to the output Excel file
output_excel_file_path = './var/CVs_Info_Extracted.xlsx'


# define a class to extract CV information
class CVsInfoExtractor:
    # define a constructor that initializes the class with a DataFrame of CVs
    def __init__(self, cvs_df, openai_api_key, desired_positions):
        self.cvs_df = cvs_df
        if os.path.exists(output_csv_file_path):
            os.remove(output_csv_file_path)


        # open a file in read mode and read the contents of the file into a variable
        with open('./var/Engineered_Prompt/Prompt.txt', 'r') as file:
            self.prompt = file.read()
        
        # Join the desired positions into a comma-separated string
        suitable_positions_str = "(" + ", ".join(desired_positions) + ")"

        # Replace the placeholder in the prompt with the formatted suitable positions string
        self.prompt = self.prompt.replace('(suitable position for the candidate)', suitable_positions_str)
        
        
        # set the OpenAI API key
        openai.api_key = openai_api_key

        self.client = AzureOpenAI(api_version="2023-12-01-preview"
                     ,azure_endpoint='https://openai-hackathonresumematch.openai.azure.com/'
                     ,api_key='ceb16d76eefe4388bfd446c453514a74', )


    # define internal function to call GPT for CV info extraction
    def _call_gpt_for_cv_info_extraction(self, prompt, cv_content, model, temperature = 0):

        # create a dict of parameters for the ChatCompletion API
        completion_params = {
            'model': model,
            'messages': [{"role": "system", "content": prompt},
                        {"role": "user", "content": cv_content}],
            'temperature': temperature}

        # send a request to the ChatCompletion API and store the response
        #response = openai.ChatCompletion.create(**completion_params)

        response = self.client.chat.completions.create(
                    model="match3", # model = "deployment_name".
                    messages= [{"role": "system", "content": prompt},{"role": "user", "content": cv_content}],
                    temperature=temperature
                    )

        #response=ChatCompletion(id='chatcmpl-93KixBuUxIIyP4qQ8DM7tRSyg7y1o', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{\n"Education Bachelor University":"University of Tennessee",\n"Education Bachelor GPA":"",\n"Education Bachelor Major":"Human Services",\n"Education Bachelor Graduation Date":"Month_Name, YYYY",\n"Education Masters University":"University of North Carolina",\n"Education Masters GPA":"",\n"Education Masters Major":"Radio, Television & Motion Pictures",\n"Education Masters Graduation Date":"Month_Name, YYYY",\n"Education PhD University":"",\n"Education PhD GPA":"",\n"Education PhD Major":"",\n"Education PhD Graduation Date":"",\n"Years of Experience":"27",\n"Experience Companies":["Company Name"],\n"Top 5 Responsibilities/Projects Titles":["Organize major campus events","Assisting academic departments with minor events","Create videos for various departments","Engaged in strategic planning","Managed advertising budget"],\n"Top 5 Courses/Certifications Titles":[],\n"Top 3 Technical Skills":[],\n"Top 3 Soft Skills":["Multi-Tasking","Media Relations","Results-oriented"],\n"Current Employment Status":"Full-time",\n"Nationality":"",\n"Current Residence":"City, State",\n"Suitable Position":"d",\n"Candidate Rating (Out of 10)":"",\n"summary":"Experienced Media Activities Specialist with a strong background in event planning, media relations, and strategic initiatives. Skilled in organizing major campus events, creating videos for academic purposes, and managing advertising budgets. Strong multitasking abilities and results-oriented mindset."\n}', role='assistant', function_call=None, tool_calls=None), content_filter_results={'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}})], created=1710581675, model='gpt-35-turbo', object='chat.completion', system_fingerprint='fp_8abb16fa4e', usage=CompletionUsage(completion_tokens=275, prompt_tokens=2124, total_tokens=2399), prompt_filter_results=[{'prompt_index': 0, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}])
        
        print(response)
        print(type(response)    )
        print('choices' in response )
         # if the response contains choices and at least one choice, extract the message content
        # if 'choices' in response and len(response.choices) > 0:
        #cleaned_response = response['choices'][0]['message']['content']
        cleaned_response = (((response.choices[0]).message).content)

        print(type(cleaned_response)    )
        print(cleaned_response)
        json_response=cleaned_response

        # try:
        #     # try to convert the message content to a JSON object
        #     json_response = json.loads(cleaned_response)
        # except JSONDecodeError:
        #     # if the conversion fails, set the JSON response to None
        #     json_response = None  
        # else:
        #     # if the response does not contain choices or no choice, set the JSON response to None
        #     json_response = None    
        


            
        # return the JSON response
        return json_response
    
    
    # Defines internal function to normalize a JSON response from GPT
    def _normalize_gpt_json_response(self, CV_Filename, json_response):
        
        # Creates a DataFrame with one column "CV_Filename", the values of this column is from the "CV_Filename"
        CV_Filename_df = pd.DataFrame([CV_Filename], columns = ['CV_Filename'])

        # Creates a DataFrame with one column "All_Info_JSON", the values of this column is the JSON response
        df_CV_Info_Json = pd.DataFrame([[json_response]], columns = ['All_Info_JSON'])


        print("json ********** ")
        # Normalize the JSON response, flattening it into a table
        decoded_json = json.loads(json_response)

        df_CV_Info_Json_normalized = pd.json_normalize(decoded_json)

        # Concatenates the three DataFrame along the columns
        df = pd.concat([CV_Filename_df, df_CV_Info_Json_normalized, df_CV_Info_Json], axis=1)
        
        # Returns the final DataFrame
        return df


    # Defines internal function to write the DataFrame into a CSV file
    def _write_response_to_file(self, df):

        # Checks if the output CSV file already exists
        if os.path.isfile(output_csv_file_path):
            # If the file exists, append the DataFrame into the CSV file without writing headers
            df.to_csv(output_csv_file_path, mode='a', index=False, header=True)
        else:
            # If the file doesn't exist, write the DataFrame into a new CSV file
            df.to_csv(output_csv_file_path, mode='w', index=False)


    # Define the internal function _gpt_pipeline
    def _gpt_pipeline(self, row, model = 'gpt-3.5-turbo'):

        # Retrieve the CV Filename and Content from the given row
        CV_Filename = row['CV_Filename']
        CV_Content = row['CV_Content']

        # Sleep for 5 seconds to delay the next operation
        time.sleep(2)
        
        
        # Print status message indicating GPT is being called for CV info extraction
        print('Calling GPT For CV Info Extraction...')

        # Call the GPT model for CV information extraction
        json_response = self._call_gpt_for_cv_info_extraction(prompt=self.prompt, cv_content=CV_Content, model=model)
      

        # Print status message indicating normalization of GPT response
        print('Normalizing GPT Response...')

        # Normalize the GPT JSON response
        df = self._normalize_gpt_json_response(CV_Filename, json_response)

        # Print status message indicating that the results are being appended to the CSV file
        print('Appending Results To The CSV File...')

        # Write the normalized response to a file
        self._write_response_to_file(df)
        
        # Print a line for clarity in the output
        print('----------------------------------------------')

        # Return the GPT JSON response
        return json_response




    # Define the internal function _write_final_results_to_excel
    def _write_final_results_to_excel(self):
        # Load the CSV file into a pandas DataFrame
        df_to_excel = pd.read_csv(output_csv_file_path)

        # Write the DataFrame to an Excel file
        df_to_excel.to_excel(output_excel_file_path)

        # Return the DataFrame
        return df_to_excel


    # Define the main function extract_cv_info
    def extract_cv_info(self):
        # Print a status message indicating the start of the ResumeGPT Pipeline
        print('---- Excecuting ResumeGPT Pipeline ----')
        print('----------------------------------------------')

        # Apply the _gpt_pipeline function to each row in cvs_df DataFrame

        print(self.cvs_df)
        self.cvs_df['CV_Info_Json'] = self.cvs_df.progress_apply(self._gpt_pipeline, axis=1)
        print(self.cvs_df)
        

        # Print a status message indicating the completion of the extraction
        print('Extraction Completed!')

        # Print a status message indicating that results are being saved to Excel
        print('Saving Results to Excel...')

        # Write the final results to an Excel file
        final_df = self._write_final_results_to_excel()

        # Return the final DataFrame
        return final_df
