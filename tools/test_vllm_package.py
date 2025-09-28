#https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic/README.md
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.1,  # Use only 10% of total GPU memory (~2.4 GiB)
        max_model_len=512,  # Reduce context length to save memory
        enforce_eager=True,  # Disable CUDA graphs to save memory
        disable_custom_all_reduce=True,  # Disable custom all-reduce to avoid torch.compile issues
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()