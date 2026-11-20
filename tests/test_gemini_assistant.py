from types import SimpleNamespace

from BrainTumorAI import gemini_assistant


def test_client_uses_gemini_api_key_from_environment(monkeypatch) -> None:
    expected_client = object()
    captured = {}

    def create_client(**kwargs):
        captured.update(kwargs)
        return expected_client

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(gemini_assistant.genai, "Client", create_client)

    client, error = gemini_assistant._get_client()

    assert client is expected_client
    assert error is None
    assert captured == {"api_key": "test-gemini-key"}


def test_unconfigured_page_reviewer_explains_text_context_without_diagnosing(monkeypatch) -> None:
    monkeypatch.setattr(
        gemini_assistant,
        "_get_client",
        lambda: (None, "Gemini API key not configured. Add GEMINI_API_KEY to the environment."),
    )
    context = "Page: Robotics Simulation. Model label: no_tumor; score: 77.0% (uncalibrated)."
    prompt = (
        "Current page context (text summary only; no image pixels):\n"
        f"{context}\n\nQuestion: What does this result mean?"
    )

    response = gemini_assistant.get_gemini_response([], prompt)

    assert context in response
    assert "not a diagnosis" in response.lower()
    assert "GEMINI_API_KEY" in response


def test_page_reviewer_sends_only_text_context_and_nonclinical_instructions(monkeypatch) -> None:
    requests = []

    class FakeModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(text="Plain-language explanation.")

    client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(gemini_assistant, "_get_client", lambda: (client, None))
    history = [{"role": "user", "parts": ["Earlier question"]}]
    prompt = (
        "Current page context (text summary only; no image pixels):\n"
        "Model score is uncalibrated.\n\nQuestion: Explain the score."
    )

    response = gemini_assistant.get_gemini_response(history, prompt)

    assert response == "Plain-language explanation."
    request = requests[0]
    assert request["model"] == gemini_assistant.MODEL_NAME
    assert "never claim to have visually inspected an uploaded image" in request["config"].system_instruction
    assert "Do not diagnose" in request["config"].system_instruction
    assert request["contents"][0].role == "user"
    assert request["contents"][0].parts[0].text == "Earlier question"
    assert request["contents"][-1].parts[0].text == prompt
    assert all(not part.inline_data for item in request["contents"] for part in item.parts)


def test_image_description_sends_image_to_gemini_with_safety_prompt(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"test image bytes")
    requests = []

    class FakeModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(text="A grayscale image.")

    client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(gemini_assistant, "_get_client", lambda: (client, None))

    response = gemini_assistant.describe_image_for_blind(str(image_path), "uncalibrated score")

    assert response == "A grayscale image."
    request = requests[0]
    assert request["model"] == gemini_assistant.MODEL_NAME
    parts = request["contents"].parts
    assert "Do not diagnose" in parts[0].text
    assert parts[1].inline_data.data == b"test image bytes"
    assert parts[1].inline_data.mime_type == "image/png"
