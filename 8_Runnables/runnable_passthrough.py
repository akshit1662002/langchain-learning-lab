# from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",  
    temperature=0.7,
    max_tokens=2000,
)

model = llm

template = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Explain the following joke- {text}',
    input_variables=['text'],
)

parser = StrOutputParser()



#here we generate joke 
joke_generator = RunnableSequence( template ,  model ,parser)

# here we generate joke as well as explanation of joke because i want both joke and explaination of joke 
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explain' :  RunnableSequence(template2 | model | parser)
})

final_chain = RunnableSequence(joke_generator , parallel_chain)

result = final_chain.invoke({'topic' : 'cricket'})

print(result)