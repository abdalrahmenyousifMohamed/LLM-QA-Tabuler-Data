

===
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import torch
import collections
import numpy as np
# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')


def compute_prediction_sequence(model, data, device):
  """Computes predictions using model's answers to the previous questions."""

  # prepare data
  input_ids = data["input_ids"].to(device)
  attention_mask = data["attention_mask"].to(device)
  token_type_ids = data["token_type_ids"].to(device)

  all_logits = []
  prev_answers = None

  num_batch = data["input_ids"].shape[0]

  for idx in range(num_batch):

    if prev_answers is not None:
        coords_to_answer = prev_answers[idx]
        # Next, set the label ids predicted by the model
        prev_label_ids_example = token_type_ids_example[:,3] # shape (seq_len,)
        model_label_ids = np.zeros_like(prev_label_ids_example.cpu().numpy()) # shape (seq_len,)

        # for each token in the sequence:
        token_type_ids_example = token_type_ids[idx] # shape (seq_len, 7)
        for i in range(model_label_ids.shape[0]):
          segment_id = token_type_ids_example[:,0].tolist()[i]
          col_id = token_type_ids_example[:,1].tolist()[i] - 1
          row_id = token_type_ids_example[:,2].tolist()[i] - 1
          if row_id >= 0 and col_id >= 0 and segment_id == 1:
            model_label_ids[i] = int(coords_to_answer[(col_id, row_id)])

        # set the prev label ids of the example (shape (1, seq_len) )
        token_type_ids_example[:,3] = torch.from_numpy(model_label_ids).type(torch.long).to(device)

    prev_answers = {}
    # get the example
    input_ids_example = input_ids[idx] # shape (seq_len,)
    attention_mask_example = attention_mask[idx] # shape (seq_len,)
    token_type_ids_example = token_type_ids[idx] # shape (seq_len, 7)
    # forward pass to obtain the logits
    outputs = model(input_ids=input_ids_example.unsqueeze(0),
                    attention_mask=attention_mask_example.unsqueeze(0),
                    token_type_ids=token_type_ids_example.unsqueeze(0))
    logits = outputs.logits
    all_logits.append(logits)

    # convert logits to probabilities (which are of shape (1, seq_len))
    dist_per_token = torch.distributions.Bernoulli(logits=logits)
    probabilities = dist_per_token.probs * attention_mask_example.type(torch.float32).to(dist_per_token.probs.device)

    # Compute average probability per cell, aggregating over tokens.
    # Dictionary maps coordinates to a list of one or more probabilities
    coords_to_probs = collections.defaultdict(list)
    prev_answers = {}
    for i, p in enumerate(probabilities.squeeze().tolist()):
      segment_id = token_type_ids_example[:,0].tolist()[i]
      col = token_type_ids_example[:,1].tolist()[i] - 1
      row = token_type_ids_example[:,2].tolist()[i] - 1
      if col >= 0 and row >= 0 and segment_id == 1:
        coords_to_probs[(col, row)].append(p)

    # Next, map cell coordinates to 1 or 0 (depending on whether the mean prob of all cell tokens is > 0.5)
    coords_to_answer = {}
    for key in coords_to_probs:
      coords_to_answer[key] = np.array(coords_to_probs[key]).mean() > 0.5
    prev_answers[idx+1] = coords_to_answer

  logits_batch = torch.cat(tuple(all_logits), 0)

  return logits_batch


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def score_chunk_relevance(chunk, query, vectorizer):
    # Combine the chunk's text into a single string
    # print(chunk)
    chunk_text = str(chunk)
    query_text = str(query)
    
    # Transform texts to TF-IDF vectors
    texts = [chunk_text, query_text]
    tfidf_matrix = vectorizer.transform(texts)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return cosine_sim[0][0]

# Initialize TF-IDF Vectorizer
# print(chunk)
vectorizer = TfidfVectorizer().fit([" ".join(large_table.fillna('').values.flatten())])
def chunk_table_with_context(table, chunk_size=50, overlap=1):
    chunks = []
    for i in range(0, table.shape[0], chunk_size):
        start_idx = max(0, i - overlap)
        end_idx = min(table.shape[0], i + chunk_size)
        chunks.append(table.iloc[start_idx:end_idx])
    return chunks
# Streamlit app
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import torch
from io import StringIO
from dask import dataframe as dd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_chunk(model, tokenizer, chunk, queries):
    model.to(device)
    # Check relevance
    relevance_score = score_chunk_relevance(chunk, queries, vectorizer)
    if relevance_score < 0.66:  # Threshold for relevance
        return None
    df = pd.read_csv(StringIO(chunk))
    inputs = tokenizer(table=df, queries=queries, padding='max_length', return_tensors="pt", truncation=True)
    logits = compute_prediction_sequence(model, inputs, device)
    predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, logits.cpu().detach())

    if predicted_answer_coordinates[0]:
        row, col = predicted_answer_coordinates[0][0]
        chunk = pd.read_csv(StringIO(chunk))
        return chunk.iloc[row, col]

# Assuming `large_table` and `chunk_table_with_context` are defined elsewhere

# Chunk the large table with context preservation
chunks = chunk_table_with_context(large_table)

# Convert to Dask DataFrame
dask_chunks = dd.from_pandas(pd.DataFrame({'chunks': chunks}), npartitions=len(chunks))

# Define queries
queries = ["actor 100 number of movies"]

# Process each chunk in parallel
results = dask_chunks.map_partitions(lambda df: df.apply(lambda row: process_chunk(model, tokenizer, row['chunks'], queries), axis=1))

# Compute the results
# computed_results = results.compute()

# Filter out None values and print results
answers = [res for res in results if res is not None]
print(f"Predicted answers: {answers}")

        st.write(f"Predicted answers: {answers}")
