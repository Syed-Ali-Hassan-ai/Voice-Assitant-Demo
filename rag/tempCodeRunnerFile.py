# Example usage:
if __name__ == "__main__":
    assistant = AIVoiceAssistant()
    while True:
        customer_query = input("Customer: ")
        if customer_query.lower() in ['exit', 'quit']:
            break
        response = assistant._custom_response() if assistant.customer_name or assistant.customer_contact else assistant.interact_with_llm(customer_query)
        print(f"Assistant: {response}")
