#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('curl -O https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py')


# In[14]:


import pandas as pd
import minsearch


# ## Ingestion

# In[37]:


df = pd.read_csv('../data/data.csv')


# In[38]:


df.columns


# In[39]:


df


# In[45]:


documents = df.to_dict(orient='records')


# In[46]:


documents


# In[47]:


index = minsearch.Index(
    text_fields = ['exercise_name', 'type_of_activity', 'type_of_equipment', 'body_part',
       'type', 'muscle_groups_activated', 'instructions'],
    keyword_fields = ['id']
)


# In[48]:


index.fit(documents)


# In[21]:


query = "give me leg exercises for hamstrings"


# In[49]:


index.search(query, num_results=10)


# ## RAG Flow

# In[23]:


from openai import OpenAI


# In[24]:


client = OpenAI()


# In[27]:


response = client.chat.completions.create(
    model = 'gpt-4o-mini', 
    messages = [{'role':'user', 'content':query}]
)

response.choices[0].message.content # It Gave the General response without any context knowledge


# In[50]:


def search(query):
    # search function (minsearch)
    boost = {} #to give the imporatnace to each keywords by default it's 1.
    
    results = index.search(
             query=query,
             filter_dict={}, #to filter only DE results
             boost_dict = boost,
             num_results = 5
            )
    
    return results


# In[29]:


prompt_template = """
    You are a fitness coach instructor. Answer the QUESTION based on the CONTEXT from our exercises database. 
    Use only facts form the CONTEXT while answering the QUESTION.
    If the CONTEXT doesn't contain the answer, output NONE.

    QUESTION : {question}

    CONTEXT : {context}
    """.strip()

entry_template = """
'exercise_name': {exercise_name}
'type_of_activity': {type_of_activity}
'type_of_equipment': {type_of_equipment}
'body_part': {body_part}
'type': {type}
'muscle_groups_activated': {muscle_groups_activated}
'instructions': {instructions}
""".strip()


def build_prompt(query, search_result):
    # function to build the prompt
    
    context = ""
    for doc in search_result:
        context = context + entry_template.format(**doc) + "\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[33]:


def llm(prompt):
    response = client.chat.completions.create(
        model = 'gpt-4o-mini', 
        messages = [{'role':'user', 'content':prompt}]
    )
    return response.choices[0].message.content


# In[34]:


def rag(query):
    search_result = search(query)
    prompt = build_prompt(query, search_result)
    answer = llm(prompt)

    return answer


# In[54]:


answer = rag(query)
print(answer)


# In[36]:


answer = rag('I want some core exercises to help the backbending')
print(answer)


# In[55]:


answer = rag('I want some exercises to help me with the pull ups')
print(answer)


# In[57]:


question = "Which body part primarily benefits from doing push-ups?"
answer = rag(question)
print(answer)


# ## Retrieval Evaluation:

# In[58]:


df_questions = pd.read_csv('../data/ground-truth-retrieval.csv')


# In[59]:


df_questions


# In[60]:


ground_truth = df_questions.to_dict(orient='records')


# In[61]:


ground_truth[0]


# In[62]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


# In[63]:


def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[64]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[66]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


# In[67]:


from tqdm.auto import tqdm


# In[68]:


evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# ## Finding the best parameters
# 

# In[70]:


df_validation = df_questions[:100]
df_test = df_questions[100:]


# In[71]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)
        
        # Evaluate the objective function
        current_score = objective_function(current_params)
        
        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params
    
    return best_params, best_score


# In[72]:


gt_val = df_validation.to_dict(orient='records')


# In[73]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[74]:


param_ranges = {
    'exercise_name': (0.0, 3.0),
    'type_of_activity': (0.0, 3.0),
    'type_of_equipment': (0.0, 3.0),
    'body_part': (0.0, 3.0),
    'type': (0.0, 3.0),
    'muscle_groups_activated': (0.0, 3.0),
    'instructions': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[75]:


simple_optimize(param_ranges, objective, n_iterations=20)


# In[76]:


def minsearch_improved(query):
    boost = {
        'exercise_name': 2.11,
        'type_of_activity': 1.46,
        'type_of_equipment': 0.65,
        'body_part': 2.65,
        'type': 1.31,
        'muscle_groups_activated': 2.54,
        'instructions': 0.74
    }

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

evaluate(ground_truth, lambda q: minsearch_improved(q['question']))


# ## RAG Evaluation

# In[77]:


prompt2_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[78]:


len(ground_truth)


# In[81]:


record = ground_truth[0]
record


# In[84]:


record = ground_truth[0]
question = record['question']
answer_llm = rag(question)


# In[85]:


print(answer_llm)


# In[87]:


prompt = prompt2_template.format(question=question, answer_llm=answer_llm)
print(prompt)


# In[89]:


print(llm(prompt))


# In[90]:


evaluations = {}


# In[91]:


import json


# In[98]:


df_sample = df_questions.sample(n=200, random_state=1)
sample = df_sample.to_dict(orient='records')


# In[99]:


evaluations = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question) 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))
    


# In[100]:


evaluations


# In[101]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])
df_eval


# In[102]:


df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])


# In[103]:


del df_eval['record']
del df_eval['evaluation']


# In[104]:


df_eval.relevance.value_counts(normalize=True)


# In[105]:


df_eval.relevance.value_counts()


# In[106]:


df_eval.to_csv('../data/rag-eval-gpt-4o-mini.csv', index=False)


# In[107]:


df_eval[df_eval.relevance == 'NON_RELEVANT']

