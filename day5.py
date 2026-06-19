from google import genai
from dotenv import load_dotenv
import os
import requests
import sys
from bs4 import BeautifulSoup
from datetime import datetime

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:2000]
    except Exception as e:
        return f"Could not read website: {str(e)}"

def generate_proposal(business_url, business_text, service_type="AI automation"):
    prompt = f"""You are an expert AI automation consultant writing a cold outreach proposal.

Website: {business_url}
Business Content: {business_text}
Service you are offering: {service_type}

Write a professional, personalized proposal that includes:
1. A personalized opening showing you understood their business (2-3 sentences)
2. 3 specific opportunities for THEIR business (be specific, not generic)
3. Expected benefits
4. A simple call to action

Tone: Professional but conversational. Not salesy.
Length: 250-300 words maximum.
Format: Plain text, no markdown symbols."""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text

def save_proposal(url, proposal):
    # Create proposals folder if it doesn't exist
    os.makedirs("proposals", exist_ok=True)
    
    # Clean filename
    clean_url = url.replace("https://","").replace("http://","").replace("/","_")[:40]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proposals/{clean_url}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Proposal for: {url}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(proposal)
    
    return filename

def main():
    # Check if URL was provided
    if len(sys.argv) < 2:
        print("Usage: python day5.py <website_url> [service_type]")
        print("Example: python day5.py https://example.com 'AI chatbot'")
        sys.exit(1)
    
    url = sys.argv[1]
    service_type = sys.argv[2] if len(sys.argv) > 2 else "AI automation"
    
    print(f"\n🔍 Analyzing: {url}")
    print(f"📋 Service: {service_type}")
    print("-" * 50)
    
    # Scrape
    print("Reading website...")
    business_text = scrape_website(url)
    
    if "Could not read" in business_text:
        print(f"❌ {business_text}")
        sys.exit(1)
    
    print("✅ Website read successfully")
    
    # Generate
    print("🤖 Generating proposal...")
    proposal = generate_proposal(url, business_text, service_type)
    
    # Display
    print("\n" + "=" * 60)
    print("PROPOSAL")
    print("=" * 60)
    print(proposal)
    print("=" * 60)
    
    # Save
    filename = save_proposal(url, proposal)
    print(f"\n💾 Saved to: {filename}")

if __name__ == "__main__":
    main()