from __future__ import annotations

import os
import mimetypes
from typing import TYPE_CHECKING, Any
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "Google Gemini API key not configured. Add GEMINI_API_KEY to the environment."
    return genai.Client(api_key=api_key), None


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def describe_image_for_blind(image_path: str, diagnostic_info: str | None = None) -> str:
    """
    Generates a non-clinical image description through the optional Gemini API.
    """
    client, error = _get_client()
    if not client:
        return (
            "An image description service is not configured. "
            f"The available model output is: {diagnostic_info or 'no classifier output supplied'}. "
            f"This is not a radiological interpretation. {error}"
        )

    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            return "Image description failed: only JPEG, PNG, and WebP images are supported."
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        prompt = f"""
        Provide a brief, accessible description of visible image content for accessibility.
        The optional model output is: {diagnostic_info}.
        Do not diagnose, infer a tumor subtype, claim a lesion is present or absent, assign anatomy or laterality,
        provide anatomical or MNI coordinates, or suggest treatment. Grad-CAM and all displayed coordinates are
        illustrative only. State when image interpretation is uncertain.
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            ),
        )
        content = response.text
        if not content:
            return "Image-description request failed: Gemini returned no text."
        return content
    except (OSError, errors.APIError) as error:
        return f"Image-description request failed: {error}"


def get_gemini_response(chat_history: list[dict[str, Any]], user_input: str) -> str:
    """
    Handles text-only chatbot interactions through the Gemini API.
    """
    client, error = _get_client()
    if not client:
        user_lower = user_input.lower()
        context_marker = "current page context (text summary only; no image pixels):"
        if context_marker in user_lower:
            context_and_question = user_input.split(":", 1)[1]
            context = context_and_question.partition("\n\nQuestion:")[0].strip()
            return (
                "Gemini is not configured, so no AI review was generated. The app-provided page summary is: "
                f"{context} Scores are uncalibrated; Grad-CAM and all coordinates or planning values are "
                f"illustrative, not clinical findings. {error} This is not a diagnosis."
            )
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
        contents: list[types.Content] = []
        for msg in chat_history:
            parts = msg.get("parts")
            message = parts[0] if isinstance(parts, list) and parts else msg.get("content", "")
            message_text = message if isinstance(message, str) else str(message)
            contents.append(
                types.Content(
                    role="user" if msg.get("role") == "user" else "model",
                    parts=[types.Part.from_text(text=message_text)],
                )
            )

        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a non-clinical explainer for a research software prototype. Explain only the supplied "
                    "text context and general interface concepts. In this text chat you do not receive image pixels; "
                    "never claim to have visually inspected an uploaded image. Do not diagnose, infer tumor subtypes, "
                    "claim that a tumor is present or absent, provide treatment or surgical advice, or claim "
                    "anatomical/MNI localization. Explicitly describe classifier scores as uncalibrated and Grad-CAM "
                    "as qualitative, not a segmentation mask. Coordinates, volumes, safety scores, trajectories, "
                    "queue data, and telemetry are illustrative or simulated. For medical questions, direct the user "
                    "to a qualified clinician."
                )
            ),
        )
        content = response.text
        if not content:
            return "Assistant request failed: Gemini returned no text."
        return content
    except errors.APIError as error:
        return f"Assistant request failed: {error}"

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
