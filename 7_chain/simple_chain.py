
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=2000,
    do_sample=False,
)
model = ChatHuggingFace(llm = llm) 


template = PromptTemplate(
    template='Generate five interesting fact about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()


chain = template | model | parser

final_result = chain.invoke({'topic' : 'India'})

print(final_result)