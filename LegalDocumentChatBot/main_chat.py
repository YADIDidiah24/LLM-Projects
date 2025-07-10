# main_chat.py

import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
import torch

# Import your clean retrieval function
from tools import retrieve_legal_context

# Load environment variables (optional)
load_dotenv()

# --- LLM Setup ---
# Use a capable chat model like TinyLlama
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Set a pad token if one doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_pipeline = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    device=0 if torch.cuda.is_available() else -1
)

# LangChain wrapper
llm = HuggingFacePipeline(pipeline=hf_pipeline)

def get_legal_answer(query: str) -> str:
    """
    Answers a legal question using a RAG pipeline.
    """
    # Step 1: Retrieve relevant legal context
    print("Retrieving context...")
    legal_context = retrieve_legal_context(query)
    
    if "Error:" in legal_context or "No relevant context found" in legal_context:
        return legal_context

    # Step 2: Create a precise prompt using the model's chat template
    # This is crucial for getting good responses from chat models.
    chat_prompt = [
        {
            "role": "system", 
            "content": "You are a helpful legal assistant. Based *only* on the provided legal context, answer the user's question clearly and concisely. If the context doesn't contain the answer, say that you cannot find the information in the document."
        },
        {
            "role": "user", 
            "content": f"""
Legal Context:
---
{legal_context}
---

Question: {query}
"""
        }
    ]
    
    final_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)

    # Step 3: Generate response
    print("Generating answer...")
    response = llm.invoke(final_prompt)
    
    # Step 4: Clean up the response robustly
    # The pipeline output includes the prompt, so we extract just the assistant's reply.
    assistant_marker = "<|assistant|>"
    if assistant_marker in response:
        return response.split(assistant_marker)[-1].strip()
    else:
        # Fallback for unexpected response format
        return "Could not parse the model's response."


if __name__ == "__main__":
    print("Legal QA Bot (CLI Version). Type 'exit' to quit.")
    while True:
        query = input("\nPlease enter your question: ")
        if query.lower() == 'exit':
            break
        
        answer = get_legal_answer(query)
        print("\n--- Answer ---")
        print(answer)
        print("----------------\n")