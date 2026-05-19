"""
LLM provider catalog — single source of truth for all registered models.
Value format: PREFIX_model-name  (O_ = OpenAI, GG_ = Google GenAI)
"""
llm_provider_map: dict[str, str] = {
    "GEMINI_31_FLASH_LITE": "GG_gemini-3.1-flash-lite",
    "GPT4O_MINI":           "O_gpt-4o-mini",
    "GEMINI_FLASH":         "GG_gemini-flash-latest",
    "GEMINI_25_FLASH":      "GG_gemini-2.5-flash",
}
