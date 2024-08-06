import sys
from pathlib import Path

# Add the parent directory to sys.path to import the Model class
sys.path.append(str(Path(__file__).parent.parent))

from engine import Engine

def main():
    # Initialize the model
    model = Engine("Mistral7B", quantization="int8")

    # Load the model
    checkpoint_path = Path(r"C:\joev\models\OpenHermes-2.5-Mistral-7B\model_int8.pth")
    model.load(checkpoint_path)

    # Set the prompt
    prompt = "The tallest mountain in the world is "

    # Prefill with the prompt
    tokens = model.prefill(prompt)

    print(f"Initial prompt: {prompt}")
    print(f"Tokenized prompt: {tokens}")

    # Generate 10 tokens
    for i in range(10):
        logits = model.forward(tokens)
        next_token = model.sample()
        tokens.append(next_token)
        
        # Decode and print the new token
        new_text = model.tokenizer.decode([next_token])
        print(f"Token {i+1}: {new_text}")

    # Print the final generated text
    generated_text = model.tokenizer.decode(tokens)
    print("\nFinal generated text:")
    print(generated_text)

    # Unload the model
    model.unload()

if __name__ == "__main__":
    main()  