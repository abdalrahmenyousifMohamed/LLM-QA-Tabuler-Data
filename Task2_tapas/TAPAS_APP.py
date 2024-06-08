import streamlit as st
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch , ast
from io import StringIO
from dask import dataframe as dd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import collections
import numpy as np
from langchain_community.llms import Ollama
from langchain import PromptTemplate # Added

llm = Ollama(model="llama3" , temperature=0.3, num_predict=100) # Added stop token

def get_model_response(user_prompt, system_prompt):
    # NOTE: No f string and no whitespace in curly braces
    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    # Added prompt template
    prompt = PromptTemplate(
        input_variables=["system_prompt", "user_prompt"],
        template=template
    )
    
    # Modified invoking the model
    response = llm(prompt.format(system_prompt=system_prompt, user_prompt=user_prompt))
    
    return response
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')

def compute_prediction_sequence(model, data, device):
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    token_type_ids = data["token_type_ids"].to(device)

    all_logits = []
    prev_answers = None
    num_batch = data["input_ids"].shape[0]

    for idx in range(num_batch):
        if prev_answers is not None:
            coords_to_answer = prev_answers[idx]
            prev_label_ids_example = token_type_ids_example[:,3]
            model_label_ids = np.zeros_like(prev_label_ids_example.cpu().numpy())

            token_type_ids_example = token_type_ids[idx]
            for i in range(model_label_ids.shape[0]):
                segment_id = token_type_ids_example[:,0].tolist()[i]
                col_id = token_type_ids_example[:,1].tolist()[i] - 1
                row_id = token_type_ids_example[:,2].tolist()[i] - 1
                if row_id >= 0 and col_id >= 0 and segment_id == 1:
                    model_label_ids[i] = int(coords_to_answer[(col_id, row_id)])

            token_type_ids_example[:,3] = torch.from_numpy(model_label_ids).type(torch.long).to(device)

        prev_answers = {}
        input_ids_example = input_ids[idx]
        attention_mask_example = attention_mask[idx]
        token_type_ids_example = token_type_ids[idx]
        outputs = model(input_ids=input_ids_example.unsqueeze(0),
                        attention_mask=attention_mask_example.unsqueeze(0),
                        token_type_ids=token_type_ids_example.unsqueeze(0))
        logits = outputs.logits
        all_logits.append(logits)

        dist_per_token = torch.distributions.Bernoulli(logits=logits)
        probabilities = dist_per_token.probs * attention_mask_example.type(torch.float32).to(dist_per_token.probs.device)

        coords_to_probs = collections.defaultdict(list)
        prev_answers = {}
        for i, p in enumerate(probabilities.squeeze().tolist()):
            segment_id = token_type_ids_example[:,0].tolist()[i]
            col = token_type_ids_example[:,1].tolist()[i] - 1
            row = token_type_ids_example[:,2].tolist()[i] - 1
            if col >= 0 and row >= 0 and segment_id == 1:
                coords_to_probs[(col, row)].append(p)

        coords_to_answer = {}
        for key in coords_to_probs:
            coords_to_answer[key] = np.array(coords_to_probs[key]).mean() > 0.5
        prev_answers[idx+1] = coords_to_answer

    logits_batch = torch.cat(tuple(all_logits), 0)
    return logits_batch

def score_chunk_relevance(chunk, query, vectorizer):
    chunk_text = str(chunk)
    query_text = str(query)
    texts = [chunk_text, query_text]
    tfidf_matrix = vectorizer.transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def chunk_table_with_context(table, chunk_size=50, overlap=1):
    chunks = []
    for i in range(0, table.shape[0], chunk_size):
        start_idx = max(0, i - overlap)
        end_idx = min(table.shape[0], i + chunk_size)
        chunks.append(table.iloc[start_idx:end_idx])
    return chunks

def process_chunk(model, tokenizer, chunk, queries,vectorizer):
    model.to(device)
    relevance_score = score_chunk_relevance(chunk, queries, vectorizer)
    if relevance_score < 0.5:
        return None
    df = pd.read_csv(StringIO(chunk))
    inputs = tokenizer(table=df, queries=queries, padding='max_length', return_tensors="pt", truncation=True)
    logits = compute_prediction_sequence(model, inputs, device)
    predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, logits.cpu().detach())
    if predicted_answer_coordinates[0]:
        row, col = predicted_answer_coordinates[0][0]
        chunk = pd.read_csv(StringIO(chunk))
        return str((chunk.iloc[row, col], relevance_score))

st.title("TAPAS Table Question Answering")
st.write("Upload a CSV file and enter your query to get answers from the table using TAPAS model.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    st.write("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    query = st.text_input("Enter your query")

    if query:
        st.write(f"Query: {query}")

        # vectorizer = TfidfVectorizer().fit([" ".join(df.fillna('').values.flatten())])
        vectorizer = TfidfVectorizer().fit([" ".join(df.astype(str).fillna('').values.flatten())])


        chunks = chunk_table_with_context(df)
        columns = df.columns.tolist()

        dask_chunks = dd.from_pandas(pd.DataFrame({'chunks': chunks}), npartitions=len(chunks))

        results  = dask_chunks.map_partitions(lambda df: df.apply(lambda row: process_chunk(model, tokenizer, row['chunks'], query,vectorizer), axis=1))


        filtered_results = [res for res in results if res is not None and not pd.isna(res)]

        # Convert strings back to tuples
        def parse_result(res):
            try:
                return ast.literal_eval(res)
            except (SyntaxError, ValueError):
                return None

        parsed_results = [parse_result(res) for res in filtered_results if res]
        print(parsed_results)

        # Ensure only valid tuples are processed
        # answers = [(ans, score) for ans, score in parsed_results if isinstance(parsed_results, tuple)]
        for entry in parsed_results:
                res = entry[0]
                score = entry[1]
                print("res =", res)
                print("score =", score)
                print()
                st.write(f"Answer: {res}")
                st.write(f"Relevance Score: {score}")

        user_prompt = query
        print(columns)

        with st.spinner('Processing...'):
                system_prompt = (
            "Use the following pieces of information to answer the user's question. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            "and here the order of columns {columns} like in the context , Explain answer in the context of the table and all columns\n\n"
            f"Context: {parsed_results}\n\n"
            f"Question: {query}\n\n"
            "Answer the question and provide additional helpful information, "
            "based on the pieces of information, if applicable. Be succinct. "
            "Responses should be properly formatted to be easily read."
            "Example: Query: what is the age of actor 10 ? Answer: 10 Actor 10 30 20  actor 10 has 30 age and 20 movies\n\n"
        )
                response = get_model_response(user_prompt, system_prompt)
                st.write(f" {response}")
