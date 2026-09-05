from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",  
    temperature=0.7,
    max_tokens=2000,
)

model = llm

prompt = PromptTemplate(
    template='write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('9_document_loader/cricket.txt' , encoding='utf-8')

docs = loader.load()

# print(docs)
# print(type(docs))
# print(docs[0])


chain = prompt | model | parser
result = chain.invoke({'poem' : docs[0].page_content})

print(result)