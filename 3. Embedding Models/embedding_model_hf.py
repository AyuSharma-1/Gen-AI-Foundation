from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text="my name is ayush sharma, i am a software developer"

vector=embedding.embed_query(text)
print(vector)