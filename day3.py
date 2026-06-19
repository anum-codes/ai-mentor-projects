from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# This is the coach's personality - defined once, controls everything
SYSTEM_PROMPT = """You are Ustad Saab, a strict but caring Pakistani business coach. 
You have 20 years of experience helping young Pakistanis build successful careers 
and freelance businesses.

Your style:
- You speak in a mix of English and occasional Urdu words (like "beta", "yaar", "suno")
- You are direct, no sugarcoating
- You push people when they make excuses
- You celebrate real wins enthusiastically  
- You always bring conversation back to concrete next actions
- You remember everything discussed in the conversation

When someone tells you their goal, you hold them to it."""

def chat_with_coach(conversation_history, user_message):
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })
    
    # Send full history every time
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=conversation_history,
        config={
            "system_instruction": SYSTEM_PROMPT
        }
    )
    
    # Get reply
    assistant_message = response.text
    
    # Add AI reply to history too
    conversation_history.append({
        "role": "model",
        "parts": [{"text": assistant_message}]
    })
    
    return assistant_message, conversation_history

def main():
    print("=" * 50)
    print("USTAD SAAB - Your Business Coach")
    print("Type 'quit' to exit")
    print("=" * 50)
    print()
    
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Ustad Saab: Beta, remember — consistency is everything. See you tomorrow!")
            break
            
        if not user_input:
            continue
        
        response, conversation_history = chat_with_coach(conversation_history, user_input)
        print(f"\nUstad Saab: {response}\n")

if __name__ == "__main__":
    main()