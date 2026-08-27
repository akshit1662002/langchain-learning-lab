from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings( model="text-embedding-3-large" , dimensions=32)


document = ["Delhi is the capital of india" , "lucknow is the capital of uttarpradesh" , "kolkata is the capital of west Bengal"]


result = embedding.embed_documents(document)

print(result)