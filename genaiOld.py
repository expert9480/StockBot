# from google import genai #pip install -q -U google-genai
import google.generativeai as genai
import os
from dotenv import load_dotenv
import prompts

"""
Notes:
Maybe trade on other markets during market closure. (Like HK)
Incorporate Jim Kramer info. (Do opposite of what he says)
try also using bitcoin rainbow chart
if price is a certain % close to buy price, consider selling to prevent losses
keep a bit of s&p 500 or nasdaq for stabilitiy
every few hours email a report to me
If P/E is 36 or higher, reconsider buying stock. (High P/E means overvalued)
"""

# Load environment variables
load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv('geminiKey'))

# System message to define the persona
# chatbot_instructions = prompts.chatbotInstructions
chatbot_instructions = prompts.chatbotInstructionsTesting

# Initialize the model with system instructions
model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview", #change to thinking model
    system_instruction=chatbot_instructions
)

def start_chat():
    # Start a chat session to maintain context (optional but recommended)
    chat_session = model.start_chat(history=[])
    
    print("Chat Active (Type 'exit' to stop chat)")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Chat closing!")
            break
        
        try:
            # Send message to the model
            response = chat_session.send_message(user_input)
            print(f"\nGemini: {response.text}\n")
        except Exception as e:
            print(f"An error occurred: {e}")

def messageGemini(articleList,stockInfo,jimfo):
    chatSession = model.start_chat(history=[])
    response = chatSession.send_message(" Your next task is to analyze the following articles and current stock information: " + articleList + " " + stockInfo + " " + jimfo)
    return response.text

if __name__ == "__main__":
    start_chat()