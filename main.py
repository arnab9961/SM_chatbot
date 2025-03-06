import os
import json
import sys
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add the services directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'services'))

from chatbot_service import ChatbotService

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the JSON data first to check if it exists
try:
    data_path = os.path.join(os.getcwd(), "dataset.json")
    with open(data_path, "r") as f:
        json_data = json.load(f)
    print(f"Successfully loaded dataset from {data_path}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    json_data = None

# Load the GPT-2 model
try:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("Successfully loaded GPT-2 model")
except Exception as e:
    print(f"Error loading GPT-2 model: {str(e)}")
    model = None
    tokenizer = None

# Instantiate the ChatbotService with the model and JSON data
if json_data and model:
    chatbot_service = ChatbotService(model, json_data)
    print("Chatbot service initialized successfully")
else:
    print("Failed to initialize chatbot service")
    chatbot_service = None

class ChatInput(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the SM Technology GPT-2 Chatbot!"}

@app.post("/chat")
async def chat(chat_input: ChatInput):
    if not chatbot_service:
        return {"response": "Chatbot service is not available at the moment. Please try again later."}
    
    response = chatbot_service.get_response(chat_input.message)
    return {"response": response}

# Test FAQs directly from the dataset
@app.get("/faq")
async def get_faq():
    if json_data:
        return {"faq": json_data["faq"]}
    return {"error": "FAQ data not available"}

# Test prompts
test_prompts = {
    "company_info": [
        "What is SM Technology?",
        "Who owns SM Technology?",
        "Tell me about bdCalling IT",
        "Who is the CEO of bdCalling IT?",
        "Who is the General Manager of SM Technology?",
        "What's the relationship between SM Technology and bdCalling IT?",
        "Who is the GM of Sales at SM Technology?"
    ],
    "services": [
        "What services does SM Technology offer?",
        "Tell me about your mobile app development services",
        "Do you build AI solutions?",
        "What technologies do you use for website development?",
        "Can you create a WordPress website for me?",
        "Do you offer data entry services?",
        "What kind of CMS development do you provide?"
    ],
    "pricing": [
        "How much does it cost to develop a mobile app?",
        "What's the pricing for website development?",
        "How much would an AI solution cost?",
        "What are your rates for data entry services?",
        "What's the starting price for a CMS development project?"
    ],
    "tech_stack": [
        "What technologies does SM Technology work with?",
        "Do you use React for development?",
        "Can you develop applications with Flutter?",
        "Do you work with Laravel?",
        "Is Next.js part of your tech stack?"
    ],
    "company_structure": [
        "What are the sister concerns of bdCalling IT?",
        "Who is the chairperson of bdCalling IT?",
        "How many sister companies does bdCalling IT have?",
        "Tell me about Spart Tech Agency",
        "Is Back Bancher related to SM Technology?"
    ],
    "complex_queries": [
        "Who is the GM of SM Technology and what services do you provide?",
        "Tell me about your AI solutions and their pricing",
        "What technologies do you use for mobile app development and how much does it cost?",
        "Is SM Technology part of bdCalling IT, and who runs these companies?",
        "Can you develop a website using WordPress and what would be the approximate cost?"
    ],
    "edge_cases": [
        "What's your physical office address?",
        "How many employees work at SM Technology?",
        "When was SM Technology founded?",
        "Can you help me with hardware repair?",
        "Do you offer graphic design services?"
    ]
}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)