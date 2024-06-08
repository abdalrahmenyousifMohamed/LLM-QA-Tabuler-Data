# _VOIS

This repository contains code for two distinct tasks leveraging state-of-the-art Language Models (LLMs). Below is an explanation of each task along with the approach used and the models involved.

---

## Task 1: Semantic Embedding with LLM (Which Embedding To Choose)

**Objective:**  
The goal of this task is to utilize semantic embedding techniques along with a Language Model to understand and represent textual data in a meaningful vector space.

**Approach:**
1. **Data Preprocessing:** Clean and preprocess the textual data, including tokenization and removing stop words.
2. **Model Selection:** Employ a suitable Language Model with strong semantic understanding capabilities, such as BGE, Llama, or GPT-3.
3. **Fine-tuning:** Fine-tune the chosen model on specific downstream tasks if required, to enhance its understanding of the domain-specific language.
4. **Embedding Generation:** Utilize the trained model to generate contextualized embeddings for the input text.
5. **Evaluation:** Assess the quality of the embeddings through various tasks such as similarity analysis, clustering, or downstream tasks like sentiment analysis or text classification.

**Using Llama_Index & OpenAI (DEMO):**

https://github.com/abdalrahmenyousifMohamed/VOIS_Tasks/assets/73138953/d084c8f2-0f72-454c-a5d9-3c30d7bae1a7



**Using Open Source LLM & Embedding (DEMO):**

https://github.com/abdalrahmenyousifMohamed/VOIS_Tasks/assets/73138953/c64d0f18-fea9-43d6-bbfd-a92c78ee8e46




## Task 2: QA Tabular Data with TAPAS(Filter)

**Objective:**  
This task focuses on utilizing the TAPAS (Table Parsing) model, which is a specialized LLM designed for question answering on tabular data.

**Approach:**
1. **Data Preparation:** Format the tabular data along with associated questions for training the TAPAS model. Each row in the table corresponds to a potential question-answer pair.
2. **Model Selection:** Utilize the TAPAS model, which has been pre-trained on large-scale tabular data and fine-tuned for question answering tasks.
3. **Fine-tuning:** Fine-tune the TAPAS model on domain-specific tabular data if necessary, to improve performance on specific types of questions or tables.
4. **Inference:** Deploy the fine-tuned model to answer questions on new tabular data. The model can handle complex questions requiring logical reasoning over the table contents.
5. **Evaluation:** Evaluate LLMs through intrinsic, extrinsic, human, and robustness testing to gauge linguistic understanding, performance in downstream tasks, text quality, and reliability.
   

**TAPAS Fine-Tuned and Additional Case (DEMO):**

https://github.com/abdalrahmenyousifMohamed/VOIS_Tasks/assets/73138953/31479fd5-93a2-45d5-88c3-1ae2c38259b5



