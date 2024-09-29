import spacy
import re
from fastapi import FastAPI
from pydantic import BaseModel

# Load your existing trained model with a relative path
model_path = 'model-best'
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
    # Convert input to lowercase and sanitize the text
    text = request.text.lower()
    sanitized_text = re.sub(r'[^a-z0-9., ]', '', text)  # Allow only lowercase letters, numbers, '.', and ','

    # Check for greetings
    if sanitized_text in greetings:
        return {"response": greetings[sanitized_text]}

    # Process the sanitized input text with the model
    doc = nlp1(sanitized_text)

    # Step 1: Create a list to hold each expense dictionary
    expenses = []
    expense_amounts = []  # List to hold expenses

    # Extract entities and corresponding amounts
    for ent in doc.ents:
        if ent.label_ != "EXPENSE":  # Handle non-expense entities
            category = ent.label_
            expenses.append({
                "category": category,
                "description": ent.text,
                "amount": 0.0  # Initialize the amount as 0 for now
            })

    # Extract numerical values (expenses)
    for token in doc:
        if token.like_num:
            expense_amounts.append(float(token.text))

    # Logic to combine expenses with categories
    index = 0
    for i, data in enumerate(expenses):
        if index < len(expense_amounts):  # Ensure there's a corresponding expense
            expenses[i]["amount"] = expense_amounts[index]  # Assign the expense amount
            index += 1  # Move to the next expense

    # Prepare the final response with a list of expenses
    final_response = {
        "user_id": request.user_id,
        "expenses": expenses  # Attach the list of expenses
    }

    # Check for valid output conditions
    if not expenses:
        return {
            "user_id": request.user_id,
            "response": "Sorry, I did not get that."
        }

    # Return the final output
    return final_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)