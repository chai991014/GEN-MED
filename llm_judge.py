import time
import base64
import io
from prompt_template import get_judge_prompt, get_api_judge_prompt


def encode_image(image):
    """Encodes PIL image to base64 for API."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


class GroqJudge:
    def __init__(self, api_key, model_id=None):
        try:
            from groq import Groq
            key = api_key
            if not key:
                print("⚠️ Groq API Key missing. Judge will fail.")
                self.client = None
            else:
                self.client = Groq(api_key=key)
                self.model_id = model_id
                print(f"✅ Groq Judge initialized: {model_id}")
        except ImportError:
            print("❌ Error: 'groq' library not found. Run `pip install groq`.")
            self.client = None

    def judge_answer(self, image, question, raw_answer):
        if not self.client:
            return "Error"

        # Use the standard judge prompt from your template
        prompt_text = get_api_judge_prompt(question, raw_answer)

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a medical answer classifier. Output only 'YES', 'NO', or 'VISION_REQUIRED'."},
                    {"role": "user", "content": prompt_text}
                ],
                model=self.model_id,
                temperature=0.0,
                max_tokens=5
            )
            verdict = completion.choices[0].message.content.strip()

            # If clear, return immediately
            if "YES" in verdict.upper() or "NO" in verdict.upper():
                return verdict
        except Exception as e:
            print(f"⚠️ Groq Judge Error: {e}")
            return "Error"

        print("   ⚠️ Text ambiguous. spending tokens on Vision Check...")
        time.sleep(2)  # Safety pause to prevent Rate Limit

        prompt_text = get_judge_prompt(question, raw_answer)

        try:
            messages = [
                {"role": "system", "content": "You are a medical answer classifier. Output only 'Yes' or 'No'"},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}"
                        }
                    }
                ]}
            ]

            completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_id,
                temperature=0.0,
                max_tokens=5
            )
            verdict = completion.choices[0].message.content.strip()

            # If clear, return immediately
            if "YES" in verdict.upper() or "NO" in verdict.upper():
                return verdict
        except Exception as e:
            print(f"⚠️ Groq Judge Error: {e}")
            return "Error"
