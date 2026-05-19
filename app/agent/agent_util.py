"""
Agent utility — provider map and shared agent helpers.

llm_provider_map is the single source of truth for all registered LLM models.
The env only selects which ones to use (LLM_PROVIDER_CHAIN) — model definitions
never live in .env.

Provider key format:  PREFIX_model-name
  O_   → OpenAI        (ChatOpenAI)
  GG_  → Google GenAI  (ChatGoogleGenerativeAI)
"""

# Master list of all available LLM providers.
# To add a model: add an entry here — no env changes needed.
# To activate it: include its key in LLM_PROVIDER_CHAIN in .env.
llm_provider_map: dict[str, str] = {
    "GEMINI_31_FLASH_LITE": "GG_gemini-3.1-flash-lite",   # default — langchain-google-genai 3.x preserves thought_signatures
    "GPT4O_MINI":           "O_gpt-4o-mini",
    "GEMINI_FLASH":         "GG_gemini-flash-latest",
    "GEMINI_25_FLASH":      "GG_gemini-2.5-flash",
}
