import ingest
from openai import OpenAI
from time import time


client = OpenAI()
index = ingest.load_index()


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


def llm(prompt, model):
    response = client.chat.completions.create(
        model = model, 
        messages = [{'role':'user', 'content':prompt}]
    )
    return response.choices[0].message.content


def rag(query, model = 'gpt-4o-mini'):
    t0 = time()
    search_result = search(query)
    prompt = build_prompt(query, search_result)
    answer = llm(prompt, model)
    t1 = time()

    took = t1 - t0
    answer_data = {
        "answer" : answer,
        "model_used" : model,
        "response_time" : took,
        "relevance" : "RELEVANT",
        "relevance_explanation" : "None",
        "prompt_tokens" : len(prompt.split()),
        "completion_tokens" : len(answer.split()), 
        "total_tokens" :  len(prompt.split()) + len(answer.split()),
        "eval_prompt_tokens" : 0,
        "eval_completion_tokens" : 0, 
        "eval_total_tokens" : 0,
        "openai_cost" : 0,
    }

    return answer_data

