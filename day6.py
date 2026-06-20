from google import genai
from dotenv import load_dotenv
import os
import sys
import fitz  # pymupdf

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def read_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        num_pages = len(doc)
        
        for page_num in range(num_pages):
            page = doc[page_num]
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        doc.close()
        print(f"✅ Read {num_pages} pages from {pdf_path}")
        return full_text
    
    except Exception as e:
        return f"Could not read PDF: {str(e)}"

def ask_about_pdf(pdf_text, question, conversation_history):
    """Ask a question about the PDF content"""
    
    # First message includes the PDF content
    if len(conversation_history) == 0:
        first_message = f"""I have a document for you to analyze. Here is the full content:

{pdf_text[:8000]}

--- END OF DOCUMENT ---

Now answer this question about the document:
{question}"""
        conversation_history.append({
            "role": "user",
            "parts": [{"text": first_message}]
        })
    else:
        # Follow-up questions don't need to resend the PDF
        conversation_history.append({
            "role": "user",
            "parts": [{"text": question}]
        })
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=conversation_history,
        config={
            "system_instruction": """You are an expert document analyst. 
You have been given a document to analyze.
Answer questions accurately based ONLY on the document content.
If the answer is not in the document, say so clearly.
Be concise and specific."""
        }
    )
    
    answer = response.text
    conversation_history.append({
        "role": "model",
        "parts": [{"text": answer}]
    })
    
    return answer, conversation_history

def main():
    if len(sys.argv) < 2:
        print("Usage: python day6.py <path_to_pdf>")
        print("Example: python day6.py document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n📄 Loading PDF: {pdf_path}")
    pdf_text = read_pdf(pdf_path)
    
    if "Could not read" in pdf_text:
        print(f"❌ {pdf_text}")
        sys.exit(1)
    
    print("\n🤖 PDF loaded! You can now ask questions about it.")
    print("Type 'quit' to exit\n")
    print("=" * 50)
    
    conversation_history = []
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🤖 Thinking...")
        answer, conversation_history = ask_about_pdf(
            pdf_text, question, conversation_history
        )
        print(f"\n📝 Answer:\n{answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()