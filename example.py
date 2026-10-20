from model_paths import parse_engine_args
from nanovllm import LLM


def main():
    with LLM(parse_engine_args()) as llm:
        prompts = ["Introduce yourself.", "List all prime numbers within 100."]
        formatted = [llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ) for prompt in prompts]
        outputs = llm.generate(formatted, llm.sampling_params(max_tokens=256))
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}\nCompletion: {output['text']}")


if __name__ == "__main__":
    main()
