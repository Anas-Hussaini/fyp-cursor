import os
import json
import re
from typing import Dict, Any


def _regex_extract(user_request: str) -> Dict[str, Any]:
    """Lightweight local fallback extraction using regex (GHz/MHz, gain, noise)."""
    extracted: Dict[str, Any] = {
        "freq_low": None,
        "freq_high": None,
        "gain_db": None,
        "noise_figure_db": None,
    }

    # Range like "1 to 6 GHz" or "1-6 GHz/MHz"
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(GHz|MHz)", user_request, re.I)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        unit = range_match.group(3).lower()
        if unit == "mhz":
            low /= 1000.0
            high /= 1000.0
        extracted["freq_low"] = low
        extracted["freq_high"] = high
    else:
        single_match = re.search(r"(\d+(?:\.\d+)?)\s*(GHz|MHz)", user_request, re.I)
        if single_match:
            value = float(single_match.group(1))
            unit = single_match.group(2).lower()
            if unit == "mhz":
                value /= 1000.0
            extracted["freq_low"] = value
            extracted["freq_high"] = value

    gain_match = re.search(r"(\d+\.?\d*)\s*dB.*gain", user_request, re.I)
    if gain_match:
        extracted["gain_db"] = float(gain_match.group(1))

    noise_match = re.search(r"(\d+\.?\d*)\s*dB.*noise", user_request, re.I)
    if noise_match:
        extracted["noise_figure_db"] = float(noise_match.group(1))

    return extracted


def extract_requirements_via_openai(user_request: str) -> Dict[str, Any]:
    """Try extracting requirements using OpenAI with structured JSON output.

    Returns a dict with keys: freq_low, freq_high, gain_db, noise_figure_db (all optional floats in GHz/dB).
    Falls back to regex if the SDK/key/model is unavailable or if parsing fails.
    """
    # Fallback if no API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
         return _regex_extract(user_request)

    # Lazy import to avoid hard dependency
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return _regex_extract(user_request)

    try:
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Ask for strict JSON. Many models support response_format json_object.
        system = (
            "You extract RF requirements from a short user prompt."
            " Output ONLY JSON with keys: freq_low, freq_high, gain_db, noise_figure_db."
            " Use GHz units for frequency (floats). If a single frequency is given, set both"
            " freq_low and freq_high to that value. Missing values must be null."
        )

        user = f"Prompt: {user_request}"

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)

        # Normalize keys and types
        result = {
            "freq_low": data.get("freq_low"),
            "freq_high": data.get("freq_high"),
            "gain_db": data.get("gain_db"),
            "noise_figure_db": data.get("noise_figure_db"),
        }

        # Ensure numbers if present
        for k in list(result.keys()):
            v = result[k]
            if v is not None:
                try:
                    result[k] = float(v)
                except Exception:
                    result[k] = None

        # Fallback sanity: if no frequencies, try regex
        if result.get("freq_low") is None or result.get("freq_high") is None:
            regex_result = _regex_extract(user_request)
            for k, v in regex_result.items():
                if result.get(k) is None:
                    result[k] = v

        return result
    except Exception:
        # Final fallback
        return _regex_extract(user_request)


