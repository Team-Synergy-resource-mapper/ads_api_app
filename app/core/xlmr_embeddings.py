from transformers import AutoTokenizer, XLMRobertaModel
import torch
import numpy as np

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
model = XLMRobertaModel.from_pretrained('FacebookAI/xlm-roberta-base')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
pooling = 'mean'
batch_size = 32

def generate_sentence_embeddings(sentences: list[str]):
    
    print("Generating sentence embeddings...")
    all_embeddings = []  # Store all embeddings

    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize the batch of sentences
        inputs = tokenizer(batch_sentences, return_tensors='pt',
                           padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs)

        # Extract last hidden states
        last_hidden_states = outputs.last_hidden_state

        # Perform pooling
        if pooling == 'cls':
            # CLS pooling: use the [CLS] token embedding
            # Shape: (batch_size, hidden_size)
            batch_embeddings = last_hidden_states[:, 0, :]
        elif pooling == 'mean':
            # Mean pooling: average token embeddings, excluding padding tokens
            attention_mask = inputs['attention_mask']
            masked_embeddings = last_hidden_states * \
                attention_mask.unsqueeze(-1)
            batch_embeddings = masked_embeddings.sum(
                dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            raise ValueError("Invalid pooling method. Choose 'cls' or 'mean'.")

        # Move embeddings to CPU and convert to NumPy
        batch_embeddings_np = batch_embeddings.cpu().numpy()

        # Append to list
        all_embeddings.append(batch_embeddings_np)

        # Free GPU memory
        torch.cuda.empty_cache()

        print(f"Processed batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}")

    # Concatenate all batch embeddings into a single NumPy array
    return np.vstack(all_embeddings)
