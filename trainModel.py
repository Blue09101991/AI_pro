from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

# app = Flask(__name__)

# CORS(app)



os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 1240
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your Symptoms."),
                     outputs="text",
                     title="Diagnose")

index = construct_index("docs")
iface.launch(share=True)


# with open('index.json') as f:
#     index = json.load(f)

# # response = chatbot("")
# @app.route('/predict', methods=['POST'])
# def hello():
#     # print(request.args.get('name'))
#     req = request.get_json()
#     input_text = req["name"]
#     response = chatbot(input_text)
#     data = {
#         'diagnosis': response
#     }
#     return jsonify(data)

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)