"""
Quick sanity check that all 4 judge backends respond.
Tests with a single trivial prompt.
"""
import os
from dotenv import load_dotenv

load_dotenv()

PROMPT = "Reply with exactly the word 'pong' and nothing else."

print("Testing all 4 judges with a ping/pong test...\n")

# 1. Llama via Ollama
print("1. Llama (Ollama, local)")
try:
    import ollama
    resp = ollama.chat(model='llama3', messages=[{"role": "user", "content": PROMPT}])
    print(f"   Response: {resp['message']['content'].strip()}")
except Exception as e:
    print(f"   FAILED: {e}")

# 2. Anthropic Claude
print("\n2. Claude (Haiku)")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=20,
        messages=[{"role": "user", "content": PROMPT}]
    )
    print(f"   Response: {resp.content[0].text.strip()}")
except Exception as e:
    print(f"   FAILED: {e}")

# 3. Google Gemma
print("\n3. Gemma (Google AI Studio)")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemma-3-27b-it')
    resp = model.generate_content(PROMPT)
    print(f"   Response: {resp.text.strip()}")
except Exception as e:
    print(f"   FAILED: {e}")

print("\nDone.")