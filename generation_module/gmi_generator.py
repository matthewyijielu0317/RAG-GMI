import requests
import json

class GMIGenerator:
    def __init__(self, api_key, organization_id=None, model_name="Qwen/Qwen3-235B-A22B-FP8"):
        """
        Initialize the GMIGenerator with API credentials and model settings.
        
        Args:
            api_key (str): The GMI API key for authentication
            organization_id (str, optional): The organization ID if using multi-org access
            model_name (str, optional): The LLM model to use, defaults to Qwen/Qwen3-235B-A22B-FP8
        """
        self.api_key = api_key
        # Only set organization_id if it's provided and not a placeholder value
        self.organization_id = organization_id if organization_id and not organization_id.startswith("YOUR_") else None
        self.model_name = model_name
        self.base_url = "https://api.gmi-serving.com/v1/chat/completions"
    
    def generate_response(self, user_query, retrieved_context_chunks, system_message=None, max_tokens=2000, temperature=0.7):
        """
        Generate a response from the LLM based on the user query and retrieved context.
        
        Args:
            user_query (str): The user's question or prompt
            retrieved_context_chunks (list): List of text chunks providing context for the answer
            system_message (str, optional): System message to guide LLM behavior
            max_tokens (int, optional): Maximum number of tokens in the response
            temperature (float, optional): Controls randomness in output, higher is more random
            
        Returns:
            str: The generated response text
            dict: Usage statistics (optional)
        """
        # Construct messages array for the API request
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Format the context chunks with the user query
        context_text = "\n---\n".join(retrieved_context_chunks)
        user_content = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nAnswer:"
        
        # Add user message
        messages.append({"role": "user", "content": user_content})
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "context_length_exceeded_behavior": "truncate"
        }
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add organization ID only if it's valid
        if self.organization_id:
            headers["X-Organization-ID"] = self.organization_id
        
        try:
            # Make the API request
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse the response JSON
            response_json = response.json()
            
            # Extract the generated content and usage stats
            generated_content = response_json['choices'][0]['message']['content']
            usage_stats = response_json.get('usage', {})
            
            return generated_content, usage_stats
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
                
            # Return a fallback message when API request fails
            return f"Sorry, I couldn't generate a response due to an API error: {str(e)}", {"error": str(e)} 