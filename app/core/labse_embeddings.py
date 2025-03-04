from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load pre-trained tokenizer and model for LaBSE
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')
model = AutoModel.from_pretrained('sentence-transformers/LaBSE')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define pooling type and batch size
pooling = 'mean'  # choose 'cls' for CLS pooling, or 'mean' for mean pooling
batch_size = 32

def generate_sentence_embeddings(sentences: list[str]):
    """
    Generate sentence embeddings using the LaBSE model.
    Processes sentences in batches and returns a NumPy array of embeddings.
    """
    print("Generating LaBSE embeddings...")
    all_embeddings = []  # To store all batch embeddings

    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize the batch of sentences
        inputs = tokenizer(batch_sentences, return_tensors='pt',
                           padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Forward pass through the model
            outputs = model(**inputs)

        # Extract the last hidden states
        last_hidden_states = outputs.last_hidden_state

        # Perform pooling over the token embeddings
        if pooling == 'cls':
            # CLS pooling: use the first token's embedding
            batch_embeddings = last_hidden_states[:, 0, :]
        elif pooling == 'mean':
            # Mean pooling: average token embeddings, ignoring padding tokens
            attention_mask = inputs['attention_mask']
            masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            batch_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            raise ValueError("Invalid pooling method. Choose 'cls' or 'mean'.")

        # Move embeddings to CPU and convert to NumPy array
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings_np)

        # Optionally free GPU memory
        torch.cuda.empty_cache()

        print(f"Processed batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}")

    # Concatenate all embeddings into a single NumPy array
    return np.vstack(all_embeddings)

# Example usage:
# sentences = ["Hello world!", "Bonjour le monde!"]
# embeddings = generate_labse_embeddings(sentences)
# print(embeddings.shape)
