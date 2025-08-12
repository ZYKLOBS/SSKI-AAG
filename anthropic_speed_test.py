import anthropic
import time

def speed_test_anthropic(api_key: str, model: str = "claude-3-5-sonnet-20241022"):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = "Hello, how fast is this response?"

    start_time = time.perf_counter()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        duration = time.perf_counter() - start_time
        print(f"Response: {response.content[0].text}")
    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"Error during Anthropic API call: {e}")
    finally:
        print(f"Anthropic API call took {duration:.4f} seconds")


if __name__ == "__main__":
    api_key = "your_anthropic_api_key"
    speed_test_anthropic(api_key)
