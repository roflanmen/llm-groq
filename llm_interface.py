import os
from typing import List
from litellm import completion


class LLMInterface:
    def generate_response(self, query: str, context: List[str], api_key: str) -> str:
        prompt = f"""Based on the provided context, please answer the question.
        
        Context:
        {' '.join(context)}

        Question: {query}

        Please provide an answer and explanation."""
        
        response = completion(
            model="groq/llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key
        )
        
        return response.choices[0].message.content
