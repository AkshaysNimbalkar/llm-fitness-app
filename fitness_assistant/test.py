import requests
import pandas as pd

df = pd.read_csv("../data/ground-truth-retrieval.csv")
question = df.sample(n=1).iloc[0]['question']

print("question: ", question)

url = "http://localhost:5000/question"


data = {"question": question}

response = requests.post(url, json=data)
#print(response.content)

try:
    print(response.json())
except ValueError as e:  # JSON decoding error
    print("Error decoding JSON:", str(e))
    print("Response text:", response.text) 


