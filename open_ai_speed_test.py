import openai
import time

def speed_test_openai(api_key: str, model: str = "gpt-3.5-turbo"):
    client = openai.OpenAI(api_key=api_key)

    prompt = "Hello, how fast is this response?"

    start_time = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        duration = time.perf_counter() - start_time
        answer = response.choices[0].message.content.strip()
        print(f"Response: {answer}")
    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"Error during OpenAI API call: {e}")
    finally:
        print(f"OpenAI API call took {duration:.4f} seconds")


if __name__ == "__main__":
    api_key = "your_openai_api_key"
    speed_test_openai(api_key)
