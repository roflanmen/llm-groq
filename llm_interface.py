import os
from typing import List
from litellm import completion


class LLMInterface:
    def generate_response(self, query: str, context: List[str], api_key: str) -> str:
        prompt = f"""На основі наданого контексту, дайте відповідь на запитання.
        
        Контекст:
        {' '.join(context)}

        Запитання: {query}

        У відповідь напиши відповідь та пояснення."""
        
        response = completion(
            model="groq/llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key
        )
        
        return response.choices[0].message.content
