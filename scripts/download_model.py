from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name on Hugging Face Hub
model_name = "microsoft/phi-2"

print(f"Attempting to download tokenizer for {model_name}...")
# This will download and cache the tokenizer
# trust_remote_code=True is required for this model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer downloaded successfully!")

print(f"\nAttempting to download model for {model_name}...")
# This will download and cache the model's architecture and weights
# Note: This is a ~5.5 GB download and may take time.
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Model downloaded successfully!")

# Save them locally so you don't need to download them again
tokenizer.save_pretrained("./models/phi-2-tokenizer")
model.save_pretrained("./models/phi-2-model")
print("\nModel and tokenizer saved locally to the './models' directory.")
