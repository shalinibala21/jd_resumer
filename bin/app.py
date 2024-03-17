#streamlit run app.py

from utils import *
import uuid
from schema import upload_request_schema
from jsonschema import validate
from jd_resume import jd_resume

from flask import Flask, request, jsonify
app = Flask(__name__)



@app.route('/upload', methods=["POST"])
def upload():

    request_data = request.json
    try:
         validate(instance=request_data, schema=upload_request_schema)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    print(request_data)

    dictToReturn=jd_resume(request_data['context'],request_data['category'],request_data['noOfMatches'],request_data['inputPath'])
    return jsonify(dictToReturn)

