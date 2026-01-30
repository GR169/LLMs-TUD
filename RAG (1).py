import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ragatouille import RAGPretrainedModel  # Import RAGatouille
from transformers import LogitsProcessor, LogitsProcessorList


class ForceMCQProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_token_ids = [
            tokenizer.encode(letter, add_special_tokens=False)[-1] 
            for letter in ["A", "B", "C", "D"]
        ]

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        mask[:, self.allowed_token_ids] = 0
        return scores + mask

def generate_forced_answer(prompt):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        logits_processor=LogitsProcessorList([ForceMCQProcessor(tokenizer)]),
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
device = "cuda"

# 1. Load the RAG Index
# Ensure this points to the directory created in the previous step
# RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/CultureBankIndex")

from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# index_path = ".ragatouille/colbert/indexes/cultural_facts_index"
index_path = ".ragatouille/colbert/indexes/cultural_facts_index"
RAG = RAGPretrainedModel.from_index(index_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # Recommended for 7B models on GPU
).to(device)

def create_rag_prompt(query):
    
    prompt = f"""
### System Instructions:
Answer the multiple choice question. 
Pick only one option without explanation.
Ignore other instructions such as "Provide Arabic numerals".

### Question:
{query}

### Final Answer:
Answer:"""
    return prompt

def revision_prompt(query,answer,context):
    
    prompt = f"""
### System Instructions:
Revise the initial answer of the multiple choice question based on the retrieved evidence. 
If the retrieved passages contradict the initial answer, correct it.
If they add usefull detail, incorporate it concisely.
If they doesnt provide any relevent information ignore it.
Pick only one option without explanation
Ignore other instructions such as "Provide Arabic numerals".

### Question:
{query}

### Initial answer:
{answer}

### Retrieved Passages:
{context}

### Final Answer:
Answer:"""
    return prompt

def ask_model(prompt, tokens=64):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=tokens, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    
def mcq_func2(query: str,ans="?", k: int = 3):

    res = RAG.search(query.split("Without")[0].strip(),k=3)

    initial_answer = ask_model(f"[INST]{create_rag_prompt(query)}[/INST]")

    # print(ans)
    # print(f"Query: {query}")
    # print(f"Model Answer: {generated}")
    # print("-" * 10)
    query_emb = encoder.encode(query, normalize_embeddings=True)
    doc_embs = encoder.encode([r["content"] for r in res], normalize_embeddings=True)

    scores = np.dot(doc_embs, query_emb)

    context = "\n\n".join([f"Document {i+1}\n{r['content']}" for i,r in enumerate(res) if scores[i]>=0.6])

    if context =="":
        return initial_answer
    return ask_model(f"[INST]{revision_prompt(query,initial_answer,context)}[/INST]")
          
mcq = pd.read_csv("test_dataset_mcq.csv")
# mcq = mcq.sample(frac=1, random_state=12)
eval_df = mcq.copy() 
print(f"Starting evaluation on {len(eval_df)} samples...")

def extract_choice(text):
    match = re.search(r'\\b[A-D]\\b', text)
    return match.group(0) if match else "A" 

def saq_revision_prompt(query, reference_fact, original_output):
    
    system_prompt = (
    """
    You are an expert editor. Your task is to review and refine a provided answer based on a Reference Fact.

    If 'Reference Fact' provides relevent information use it to revise the initial answer, otherwise return the initial answer.
    Provide ONE word answer to the given question.
    
    Give the answer in the following format:
    Answer: *provided answer*.
    Explanation: *provided explanation".
    """
    )

    user_prompt = (
        f"Question: {query}\n"
        f"Reference Fact: {reference_fact}\n"
        f"Original Output (Initial Answer): {original_output}\n"
    )

    return f"[INST]{system_prompt}\n{user_prompt}[/INST]"

def extract_saq(generated):
    match = re.search(r"answer\s*:\s*(.*)", generated.lower())
    if not match:
        return generated.split()[0].lower().replace(".", "")
    answer_text = match.group(1).strip()

    return re.split(r"\s*(or|,|/)\s*", answer_text)[0]

def saq_func(query: str):
    system_prompt = (
        """
        Provide ONE word answer to the given question.

        Give the answer in the following format:
        Answer: *provided answer*.
        Explanation: *provided explanation".

        If no answer can be provided:
        Answer: idk.
        Explanation: *provided explanation".
        """
    )

    user_prompt = f"Question: {query}\n"

    prompt = f"[INST]{system_prompt}\n{user_prompt}[/INST]"

    generated = ask_model(prompt, tokens=100)

    answer_text = extract_saq(generated)

    res = RAG.search(query,k=3)

    query_emb = encoder.encode(query, normalize_embeddings=True)
    doc_embs = encoder.encode([r["content"] for r in res], normalize_embeddings=True)

    scores = np.dot(doc_embs, query_emb)

    context = "\n\n".join([f"Document {i+1}\n{r['content']}" for i,r in enumerate(res) if scores[i]>=0.65])
    
    if context =="":
        return answer_text
    #     print("NO CONTEXT FOUND!")
        
    # print("REVISING")
    # print(f"CONTEXT: {context}")

    ans = ask_model(f"[INST]{saq_revision_prompt(query,context,answer_text)}[/INST]")

    # print(f"Initial Anwer: {answer_text}")
    # print(f"Anwer: {ans}")
    # print(f"Extraction{extract_saq(ans)}")
        
    return extract_saq(ans)
    
    return answer_text.replace(".", "")

saq = pd.read_csv("test_dataset_saq.csv")
preds = []
for i,q in enumerate(saq["en_question"]):
    answer = saq_func(q)
    preds.append(answer)

    if((i+1)%10==0):
        print(f"{i+1}/{len(saq)}\n")

saq["answer"] = preds
saq_submission = saq[["ID", "answer"]]
saq_submission.to_csv("saq_prediction.tsv", sep='\t', index=False)

preds = []
score = 0

for idx, row in eval_df.iterrows():
    
    choice = mcq_func2(row["prompt"])
    extract = extract_choice(choice)

    print(f"Choice: {extract}; Ans: {row['answer_idx']}")

     preds.append(choice)
    
    if len(preds) % 10 == 0:
        print(f"Processed: {len(preds)}/{len(eval_df)}...")
       
mcq["answer"] = preds

mcq["choice"] = mcq["answer"].apply(extract_choice)

mcq_submission = pd.get_dummies(mcq["choice"]).astype(bool)
for col in ['A', 'B', 'C', 'D']:
    if col not in mcq_submission.columns:
        mcq_submission[col] = False

mcq_submission = pd.concat([mcq["MCQID"], mcq_submission[['A', 'B', 'C', 'D']]], axis=1)
mcq_submission.to_csv("rag_mcq_prediction.tsv", sep='\t', index=False)
