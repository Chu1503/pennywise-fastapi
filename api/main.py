import spacy
from spacy.tokens import Span
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Load your existing trained model with a relative path
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output/model-best')
nlp1 = spacy.load(model_path)

# Define the FastAPI app
app = FastAPI()

# Define the request model
class RequestModel(BaseModel):
    user_id: str
    text: str

# Define greetings and responses
greetings = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! What can I do for you?",
    "how are you": "I'm just a program, but thanks for asking! How can I help you?",
    "good morning": "Good morning! How can I assist you today?",
    "good evening": "Good evening! How can I help you?",
}

@app.get("/")
async def read_root():
    return "Pennywise-NER is running!"

@app.post("/predict")
async def process_text(request: RequestModel):
    text = request.text.lower()  # Convert input to lowercase

    # Check for greetings
    if text in greetings:
        return {"response": greetings[text]}

    # Process the input text with the model
    doc = nlp1(text)

    # Step 1: Create a dictionary to hold results categorized by label
    results = {}

    # Store expenses separately
    expenses = []

    # Extract entities and corresponding amounts
    for token in doc:
        if token.like_num:
            expenses.append(token.text)  # Store found expenses in the list

    for ent in doc.ents:
        if ent.label_ != "EXPENSE":  # Handle non-expense entities
            if ent.label_ not in results:
                results[ent.label_] = {"description": [], "amounts": []}
            results[ent.label_]["description"].append(ent.text)
            # Map the latest expense to this entity if available
            if expenses:
                results[ent.label_]["amounts"].append(expenses.pop(0))  # Pop the first expense amount

    # Step 2: Create the final response structure
    final_response = []
    for category, info in results.items():
        response_entry = {
            "user_id": request.user_id,
            "category": category,
            "description": ", ".join(info["description"]),
            "amount": ", ".join(info["amounts"])  # Aggregate expense amounts for this category
        }
        final_response.append(response_entry)

    # Return the result; if no entities are found, raise an error
    return final_response if final_response else HTTPException(status_code=400, detail="Sorry, I did not get that.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
