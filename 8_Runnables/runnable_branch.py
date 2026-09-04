from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda , RunnableBranch
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",  
    temperature=0.7,
    max_tokens=2000,
)

# def word_count(text):
#     return c(text.split())

model = llm

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template='Write a detail report on  {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
) 


report_generation_chain = RunnableSequence(prompt1 , model , parser)


branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 10 , RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain , branch_chain)
result = final_chain.invoke({'topic' : 'weather'})
print(result)