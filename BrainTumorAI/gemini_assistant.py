import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if api_key:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    # OpenRouter model identifier for Gemini
    MODEL_NAME = "google/gemini-2.0-flash-001"
else:
    st.error("OpenRouter API key not found. Please check your .env file.")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image_for_blind(image_path, diagnostic_info=None):
    """
    Generates a detailed audio-friendly description of the MRI image using OpenRouter (Gemini).
    """
    try:
        base64_image = encode_image(image_path)
        prompt = f"""
        You are a medical AI assistant for blind users. 
        Analyze this MRI image and the provided diagnostic information: {diagnostic_info}.
        Explain in a very clear, empathetic, and descriptive way what the scan shows.
        Describe the visual aspects of the tumor (if any), its location, and what it means.
        Since the user is blind, use spatial descriptions (e.g., 'on the top right side of the brain').
        Keep the language accessible but professional.
        End with a reassuring statement.
        """
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image via OpenRouter: {str(e)}"

def get_gemini_response(chat_history, user_input):
    """
    Handles chatbot interactions via OpenRouter.
    """
    try:
        messages = [{"role": "system", "content": "You are a helpful medical AI assistant specialized in Neuro-Radiology."}]
        
        for msg in chat_history:
            role = "user" if msg['role'] == 'user' else "assistant"
            messages.append({"role": role, "content": msg['parts'][0] if isinstance(msg['parts'], list) else msg['content']})
            
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response via OpenRouter: {str(e)}"

def tts_component(text, auto_play=True):
    """
    Injects JavaScript for Web Speech API TTS.
    """
    if not text:
        return ""
    
    # Escape quotes and newlines for JS
    safe_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ").replace("\r", " ")
    
    html_code = f"""
    <script>
    function speak() {{
        window.speechSynthesis.cancel();
        const msg = new SpeechSynthesisUtterance("{safe_text}");
        msg.rate = 1.0;
        msg.pitch = 1.0;
        window.speechSynthesis.speak(msg);
    }}
    if ({str(auto_play).lower()}) {{
        setTimeout(speak, 500);
    }}
    </script>
    <div style="display: flex; gap: 10px; align-items: center;">
        <button onclick="speak()" style="
            background-color: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        ">🔊 Read Description Aloud</button>
        <button onclick="window.speechSynthesis.cancel()" style="
            background-color: #ef4444;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        ">🛑 Stop</button>
    </div>
    """
    return html_code
