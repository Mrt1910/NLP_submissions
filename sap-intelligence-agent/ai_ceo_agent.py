import json
import re
import chromadb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "sap_intelligence"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# The exact Hugging Face model ID for Llama 3.1 8B
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Maximum number of NEW tokens the model is allowed to generate in its response.
MAX_NEW_TOKENS = 512

# How many documents to retrieve as context for each question.
NUM_DOCUMENTS_TO_RETRIEVE = 8

COMPANY_NAME = "SAP"



# Load the embedding model, connect to ChromaDB, and load the LLM itself once, at module load time, rather than reloading them on every single question.
print("Loading embedding model (used for retrieval)...")
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Connecting to ChromaDB...")
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_collection(COLLECTION_NAME)

if torch.backends.mps.is_available():
    _device = "mps"
elif torch.cuda.is_available():
    _device = "cuda"
else:
    _device = "cpu"
print(f"Using device: {_device}")

print(f"Loading {LLM_MODEL_NAME} (this can take a while the first time - "
      f"downloading ~16GB if not already cached)...")
try:
    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    if _device == "cuda":
        # 4-bit quantization: compresses the model's weights to use
        # roughly a QUARTER of the memory that normal half-precision
        # (float16) would need (~4-5GB instead of ~16GB for this model).
    
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        _llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        _llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            dtype=torch.float16,
        ).to(_device)

    print("LLM loaded successfully.")
except Exception as e:
    # A friendly, specific error instead of a confusing stack trace
    print("ERROR: Could not load the Llama 3.1 8B model.")
    print("This usually means one of the following:")
    print("  1. You haven't been granted access to the gated model yet.")
    print("     Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("     and submit the access request form, then wait for approval.")
    print("  2. You aren't logged in locally. Run: huggingface-cli login")
    print(f"\nOriginal error: {e}")
    raise


# STEP 1: RETRIEVE

def retrieve_relevant_documents(query, n_results=NUM_DOCUMENTS_TO_RETRIEVE):
    query_embedding = _embedding_model.encode([query]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    documents = []
    for doc_text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        documents.append({
            "text": doc_text,
            "source_type": metadata.get("source_type", ""),
            "title": metadata.get("title", ""),
            "link": metadata.get("link", ""),
        })

    return documents


# STEP 2: BUILD PROMPT

def build_analysis_prompt(query, documents):
    """
    Builds the full prompt we send to the LLM. This combines:
    - Clear instructions about its role (AI CEO advisor)
    - The retrieved documents as context/evidence
    - The user's question
    - STRICT instructions about the exact JSON format we want back

    Being very explicit about the JSON shape (including showing an
    example) significantly improves how reliably the model produces
    valid, parseable JSON.
    """
    # Number each document so the LLM can refer back to specific ones as evidence (e.g. "Document 3 shows...").
    context_lines = []
    for i, doc in enumerate(documents, 1):
        context_lines.append(f"Document {i} [{doc['source_type']}]: {doc['text']}")
    context_text = "\n".join(context_lines)

    prompt = f"""You are a Strategic Intelligence Agent acting as an advisor to the CEO of {COMPANY_NAME}.

Below are real, recently collected documents (news articles, official company announcements, and technical community discussions) about {COMPANY_NAME} and its market.

{context_text}

Based ONLY on the documents above, answer this question:
"{query}"

Respond with ONLY a JSON object, and nothing else - no explanation before or after it. Use exactly this structure:

{{
  "recommendation": "A single, specific, actionable recommendation sentence",
  "priority": "High",
  "supporting_evidence": [
    "A short piece of evidence from the documents, referencing which document number it came from",
    "Another piece of evidence"
  ],
  "expected_impact": "A short description of the expected business impact",
  "risk_level": "Medium"
}}

Rules:
- "priority" must be exactly one of: "High", "Medium", "Low"
- "risk_level" must be exactly one of: "High", "Medium", "Low"
- "supporting_evidence" must contain 2-3 short items, each referencing a specific document
- Do not include any text outside the JSON object
"""
    return prompt



# BUILD A MONITOR PROMPT (multiple items at once)
# This is used for the Opportunity Monitor and Risk Monitor dashboard sections. Instead of asking for one recommendation, the LLM is asked to identify several distinct items
# (opportunities or risks), each with its own title, impact/severity level, evidence, and a confidence score.

def build_monitor_prompt(item_type, documents, num_items=4):
    """
    item_type should be either "opportunity" or "risk". This changes the
    wording of the prompt and the field name used for severity
    ("impact_level" for opportunities, "severity_level" for risks).
    """
    context_lines = []
    for i, doc in enumerate(documents, 1):
        context_lines.append(f"Document {i} [{doc['source_type']}]: {doc['text']}")
    context_text = "\n".join(context_lines)

    level_field = "impact_level" if item_type == "opportunity" else "severity_level"
    # "Add an s" pluralization doesnt work for "opportunity" -> "opportunitys" instead of "opportunities", so we handle that case explicitly.
    item_type_plural = "opportunities" if item_type == "opportunity" else f"{item_type}s"

    prompt = f"""You are a Strategic Intelligence Agent acting as an advisor to the CEO of {COMPANY_NAME}.

Below are real, recently collected documents about {COMPANY_NAME} and its market.

{context_text}

Identify the {num_items} most important distinct {item_type_plural} for {COMPANY_NAME} based ONLY on the documents above.

Respond with ONLY a JSON object, and nothing else. Use exactly this structure:

{{
  "items": [
    {{
      "title": "A short, specific title for this {item_type}",
      "{level_field}": "High",
      "evidence": "A short piece of evidence from the documents, referencing which document number it came from",
      "confidence_score": 75
    }}
  ]
}}

Rules:
- Return exactly {num_items} items in the "items" list, each a DIFFERENT {item_type}
- "{level_field}" must be exactly one of: "High", "Medium", "Low"
- "confidence_score" must be a number from 0 to 100, reflecting how strongly the evidence supports this {item_type}
- Do not include any text outside the JSON object
"""
    return prompt

# Generates dictonary with items for the risk and opportunity monitor
def generate_monitor_items(item_type, query, num_items=4):
    documents = retrieve_relevant_documents(query)
    prompt = build_monitor_prompt(item_type, documents, num_items)
    raw_response = call_llm(prompt)
    parsed = parse_llm_response(raw_response)

    if parsed is None:
        return []

    return parsed.get("items", [])




def call_llm(prompt):
    """
    Sends a prompt to the locally loaded Llama 3.1 8B model and returns
    its text response.

    How this works, step by step:
    1. Llama 3.1 expects "chat" input in a specific format (it was
       trained on conversations, not raw text). We wrap our prompt as a
       single user message and apply the model's official chat template,
       which formats it with the exact special tokens the model expects
       (e.g. marking where the user's turn starts and ends).
    2. The tokenizer converts that formatted text into numbers (token
       IDs) the model can actually process.
    3. model.generate() runs the model forward, predicting one token at
       a time, until it either produces an end-of-message token or hits
       MAX_NEW_TOKENS.
    4. We decode the generated token IDs back into text, and strip off
       the original prompt so we're left with only the NEW text the
       model generated.
    """
    messages = [{"role": "user", "content": prompt}]

    # apply_chat_template formats my message using Llama 3.1's expected
    # structure, and converts it to model-ready input.
    #
    # Pass return_dict=True explicitly, so I always get back a dictionary-like object with separate "input_ids" and
    # "attention_mask" tensors, rather than relying on the library's default behavior (which has changed between transformers versions
    model_inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_llm_model.device)

    input_ids = model_inputs["input_ids"]

    with torch.no_grad():  # disables gradient tracking - we're not training, just generating
        output_ids = _llm_model.generate(
            input_ids=input_ids,
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,  # low temperature = more focused, consistent output
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # output_ids includes the original prompt tokens and the newly generated ones.
    # Therefroe slice off everything up to the length of the original input, keeping only the model's new output.
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response_text



# STEP 4: PARSE THE RESPONSE

def parse_llm_response(raw_text):
    text = raw_text.strip()

    # Remove markdown code fences if present (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # If there's still text before/after the JSON object, extract just the part between the first '{' and the last '}'.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  WARNING: could not parse LLM response as JSON: {e}")
        print(f"  Raw response was: {raw_text[:300]}")
        return None



# STEP 5: PUT IT ALL TOGETHER

def generate_recommendation(query):

    documents = retrieve_relevant_documents(query)
    prompt = build_analysis_prompt(query, documents)
    raw_response = call_llm(prompt)
    parsed = parse_llm_response(raw_response)

    if parsed is not None:
        # Attach the source documents too, so the dashboard can show links back to the original evidence if needed.
        parsed["query"] = query
        parsed["source_documents"] = documents

    return parsed


# STEP 6: CEO BRIEFING
#
# The CEO Briefing is a short narrative summary three specific questions from the project brief: What happened? Why does it matter? What should management do next?

def build_briefing_prompt(documents):
    context_lines = []
    for i, doc in enumerate(documents, 1):
        context_lines.append(f"Document {i} [{doc['source_type']}]: {doc['text']}")
    context_text = "\n".join(context_lines)

    prompt = f"""You are a Strategic Intelligence Agent writing a brief executive
summary for the CEO of {COMPANY_NAME}, based on recent news, official
announcements, and community/technical discussion.

{context_text}

Write a concise executive briefing answering exactly these three questions.
Respond with ONLY a JSON object, and nothing else, using exactly this structure:

{{
  "what_happened": "A 2-3 sentence summary of the most important recent developments",
  "why_it_matters": "A 2-3 sentence explanation of the business significance",
  "what_to_do_next": "A 2-3 sentence statement of the recommended management action"
}}

Rules:
- Each field should be 2-3 full sentences, written in a professional executive tone
- Base everything ONLY on the documents above
- Do not include any text outside the JSON object
"""
    return prompt


def generate_ceo_briefing():
    documents = retrieve_relevant_documents(
        f"recent important news and developments about {COMPANY_NAME}",
        n_results=10,  # a slightly wider net, since this is a broad overview
    )
    prompt = build_briefing_prompt(documents)
    raw_response = call_llm(prompt)
    parsed = parse_llm_response(raw_response)

    if parsed is not None:
        parsed["source_documents"] = documents

    return parsed


# QUICK TEST: running this file directly tries a few example questions

if __name__ == "__main__":
    example_questions = [
        f"What are the biggest opportunities for {COMPANY_NAME} right now?",
        f"What are the biggest risks facing {COMPANY_NAME}?",
        f"What should {COMPANY_NAME} prioritize in the next quarter?",
    ]

    for question in example_questions:
        print(f"QUESTION: {question}")

        result = generate_recommendation(question)

        if result is None:
            print("Failed to generate a recommendation for this question.")
            continue

        print(f"Recommendation: {result.get('recommendation')}")
        print(f"Priority: {result.get('priority')}")
        print(f"Risk level: {result.get('risk_level')}")
        print(f"Expected impact: {result.get('expected_impact')}")
        print("Supporting evidence:")
        for item in result.get("supporting_evidence", []):
            print(f"  - {item}")