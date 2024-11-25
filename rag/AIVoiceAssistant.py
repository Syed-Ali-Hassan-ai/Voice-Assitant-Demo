import openai
import os
import faiss
import numpy as np
from typing import Optional, Dict
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Document:
    def __init__(self, text: str, metadata: Optional[Dict] = None):
        self.text = text
        self.metadata = metadata

class AIVoiceAssistant:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.vector_store = None
        self._prompt = self._get_prompt()
        self.customer_name = None  # Track customer name
        self.customer_contact = None  # Track customer contact
        self._create_kb()
    
    def _create_kb(self):
        try:
            with open(r"rag\restaurant_file.txt", 'r', encoding='utf-8') as f:
                text = f.read()
            paragraphs = text.split('\n\n')
            self.documents = [Document(text=p.strip()) for p in paragraphs if p.strip()]
            self._create_embeddings()
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
    
    def _create_embeddings(self):
        texts = [doc.text for doc in self.documents]
        embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = openai.Embedding.create(
                input=batch_texts,
                engine="text-embedding-3-small"
            )
            batch_embeddings = [data['embedding'] for data in response['data']]
            embeddings.extend(batch_embeddings)
        self.embeddings = np.array(embeddings, dtype='float32')
        self.vector_store = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.vector_store.add(self.embeddings)
    
    def interact_with_llm(self, customer_query):
        # Embed the customer query
        response = openai.Embedding.create(
            input=customer_query,
            engine="text-embedding-3-small"
        )
        query_embedding = np.array(response['data'][0]['embedding'], dtype='float32').reshape(1, -1)
        distances, indices = self.vector_store.search(query_embedding, 3)
        relevant_docs = [self.documents[idx].text for idx in indices[0]]
        prompt = self._construct_prompt(customer_query, relevant_docs)
        
        # Generate response using OpenAI's ChatCompletion API
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            n=1,
            temperature=0.7,
        )
        answer = completion.choices[0].message['content'].strip()
        self._update_context(customer_query)  # Update context after response
        return answer
    
    def _construct_prompt(self, customer_query, relevant_docs):
        context = "\n".join(relevant_docs)
        prompt = f"Context:\n{context}\n\nCustomer Query: {customer_query}\n\nAnswer:"
        return prompt

    def _get_prompt(self):
        return """
You are a professional and friendly receptionist at **Bangalore Kitchen**, a renowned restaurant located in Whitefield, Bangalore.

**Objectives:**
- Greet the customer warmly **once at the beginning** of the conversation.
- Provide information about the menu when asked, including popular dishes and recommendations.
- If the customer decides to order, confirm the order details clearly.
- **Politely ask for the customer's name and contact number only after the order is confirmed.**
- **Do not repeat questions or greetings that have already been provided.**
- Use the customer's name in subsequent responses to personalize the conversation.
- Conclude the conversation with a friendly closing once all necessary information is gathered.

**Guidelines:**
- **Avoid starting responses with "Hello" or repeating the restaurant's name after the initial greeting.**
- Keep responses concise, friendly, and under 20 words.
- Maintain a professional yet warm tone throughout the interaction.
- If you don't know the answer to a question, respond with "I'm sorry, but I'm not sure about that."
- Do not mention that you are an AI assistant.

**Remember:**
- **Do not repeat the greeting or ask for the same information multiple times.**
- Use the customer's name after they provide it.
- Adjust your responses based on the customer's input.
    """


    def _update_context(self, customer_query):
        if "name" in customer_query.lower():
            self.customer_name = customer_query.split()[-1]  # Simple extraction
        if "contact" in customer_query.lower() or customer_query.startswith("03"):
            self.customer_contact = customer_query  # Assume any starting with '03' is a contact

    def _custom_response(self):
        if not self.customer_name:
            return "Hello! May I have your name, please?"
        elif not self.customer_contact:
            return f"Thank you, {self.customer_name}. What's your contact number?"
        else:
            return "Thank you for providing your details! How can I assist you?"

# # Example usage:
# if __name__ == "__main__":
#     assistant = AIVoiceAssistant()
#     while True:
#         customer_query = input("Customer: ")
#         if customer_query.lower() in ['exit', 'quit']:
#             break
#         response = assistant._custom_response() if assistant.customer_name or assistant.customer_contact else assistant.interact_with_llm(customer_query)
#         print(f"Assistant: {response}")
