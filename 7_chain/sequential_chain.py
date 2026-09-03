
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


prompt1 = PromptTemplate(
    template='Generate a detail report on  \n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 poiter summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
final_result = chain.invoke({'topic' : 'hindu culter'})

print(final_result)