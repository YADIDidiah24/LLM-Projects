import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from autogen import AssistantAgent, register_function
from transformers import pipeline
import torch

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)
text_generator = pipeline(
    "text-generation",
    model="distilgpt2",  # Free conversational model
    tokenizer="distilgpt2",
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    max_length=512,
    do_sample=True,
    temperature=0.7,
    pad_token_id=50256
)


def retrieve_legal_context(query: str) -> str:

    try:
        db = FAISS.load_local("rag_faiss", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"An error occurred while loading the FAISS index: {e}")
        return "Failed to load the FAISS index."
    
    # Perform similarity search
    try:
        docs = db.similarity_search(query, k=3)
        print(docs)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"An error occurred during similarity search: {e}")
        return "Failed to perform similarity search."
    
def generate_legal_response(messages):

    try:
        # Extract the user message
        user_message = messages[-1]["content"] if messages else ""
        
        # Create a prompt for the model
        prompt = f"Legal Assistant: {user_message}\n\nResponse:"
        
        # Generate response
        response = text_generator(
            prompt,
            max_new_tokens=200,
            num_return_sequences=1,
            pad_token_id=text_generator.tokenizer.eos_token_id
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        
        # Clean up the response (remove the prompt part)
        if "Response:" in generated_text:
            answer = generated_text.split("Response:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def chat_with_legal_assistant(query: str):

    try:
        # First retrieve context
        context = retrieve_legal_context(query)
        
        # Create a prompt with context
        prompt = f"""
        Based on the following legal document context, please answer the user's question:
        
        Context:
        {context}
        
        Question: {query}
        
        Please provide a clear and accurate answer based on the context above.
        """
        
        # Get response from the assistant
        response = generate_legal_response(
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response
        
    except Exception as e:
        return f"Error in chat: {str(e)}"

# Alternative implementation using a different free model
def setup_alternative_model():
    """
    Setup an alternative free model (FLAN-T5 is good for instructions)
    """
    global text_generator
    
    print("Loading alternative model (FLAN-T5)...")
    text_generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if torch.cuda.is_available() else -1,
        max_length=512
    )

def generate_response_with_flan_t5(prompt: str) -> str:
    """
    Generate response using FLAN-T5 model
    """
    try:
        # FLAN-T5 works better with instruction-style prompts
        instruction_prompt = f"Answer the following question based on the legal context provided: {prompt}"
        
        response = text_generator(
            instruction_prompt,
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
        
        return response[0]['generated_text']
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Test the function
if __name__ == "__main__":
    # Test the retrieval function
    test_query = "Are pets allowed in the property?"
    print("="*50)
    print("Testing Legal Document Retrieval")
    print("="*50)
    
    context = retrieve_legal_context(test_query)
    print("Retrieved Context:")
    print("-" * 30)
    print(context)
    
    # Test with different queries
    test_queries = [
        "What are the payment terms?",
        "What is the notice period for termination?",
        "Are there any penalties for late payment?",
        "What are the maintenance responsibilities?"
    ]
    
    print("\n" + "="*50)
    print("Testing Multiple Queries")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        context = retrieve_legal_context(query)
        print(context[:200] + "..." if len(context) > 200 else context)
        print("\n" + "-" * 40)
    
    # Test the chat function
    print("\n" + "="*50)
    print("Testing Chat Function")
    print("="*50)
    
    test_chat_query = "What are the payment terms?"
    response = chat_with_legal_assistant(test_chat_query)
    print(f"Query: {test_chat_query}")
    print(f"Response: {response}")