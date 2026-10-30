from __future__ import annotations

import os
import base64
from typing import TYPE_CHECKING, Any
from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# Load environment variables safely
load_dotenv()

def _get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, "OpenRouter API key not configured. Add OPENROUTER_API_KEY to your .env file."
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        return client, None
    except Exception as e:
        return None, str(e)

MODEL_NAME = "google/gemini-2.0-flash-001"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image_for_blind(image_path: str, diagnostic_info: str | None = None) -> str:
    """
    Generates an image description through the optional OpenRouter service.
    """
    client, _ = _get_client()
    if not client:
        return (
            "An image description service is not configured. "
            f"The available model output is: {diagnostic_info or 'no classifier output supplied'}. "
            "This is not a radiological interpretation."
        )

    try:
        base64_image = encode_image(image_path)
        prompt = f"""
        Provide a brief, accessible description of visible image content for accessibility.
        The optional model output is: {diagnostic_info}.
        Do not diagnose, infer a tumor subtype, claim a lesion is present or absent, assign anatomy or laterality,
        provide anatomical or MNI coordinates, or suggest treatment. Grad-CAM and all displayed coordinates are
        illustrative only. State when image interpretation is uncertain.
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
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("The image-description service returned no text.")
        return content
    except Exception as e:
        return f"Image-description request failed: {e}"

def get_gemini_response(chat_history: list[dict[str, Any]], user_input: str) -> str:
    """
    Handles chatbot interactions via OpenRouter / Gemini.
    """
    client, _ = _get_client()
    if not client:
        user_lower = user_input.lower()
        if "treatment" in user_lower or "protocol" in user_lower or "surgery" in user_lower:
            return "This prototype cannot provide treatment or surgical advice. Consult a qualified clinician."
        elif "confidence" in user_lower or "accuracy" in user_lower:
            return "The displayed softmax scores are uncalibrated model outputs. Grad-CAM is qualitative and has not been validated for tumor localization."
        elif "mni" in user_lower or "coordinates" in user_lower or "location" in user_lower:
            return "This prototype does not calculate valid anatomical or MNI coordinates. Its displayed coordinate values are illustrative simulations."
        else:
            return (
                f"Regarding '{user_input}': this prototype provides only a binary image-classifier output and "
                "illustrative visualizations. It does not provide medical advice."
            )

    try:
        messages: list[ChatCompletionMessageParam] = [{
            "role": "system",
            "content": (
                "You are an accessibility-focused assistant for a research software prototype. Do not diagnose, "
                "infer tumor subtypes, make claims about lesion presence, provide treatment advice, or claim "
                "anatomical/MNI localization. Explain that scores are uncalibrated and coordinates/geometry "
                "are illustrative simulations. For medical questions, direct the user to a qualified clinician."
            ),
        }]
        
        for msg in chat_history:
            content = msg['parts'][0] if isinstance(msg.get('parts'), list) else msg.get('content', '')
            message_content = content if isinstance(content, str) else str(content)
            if msg.get("role") == "user":
                messages.append({"role": "user", "content": message_content})
            else:
                messages.append({"role": "assistant", "content": message_content})
            
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("The assistant service returned no text.")
        return content
    except Exception as e:
        return f"Assistant request failed: {e}"

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
    <div style="display: flex; gap: 10px; align-items: center; margin-top: 10px;">
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
