from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=2000,
    do_sample=False,
)

model = ChatHuggingFace(llm = llm) 

template = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Explain the following joke- {text}',
    input_variables=['text'],
)

parser = StrOutputParser()


chain = RunnableSequence(template ,model , parser , template2 , model , parser)

result = chain.invoke({'topic'  : 'Frontend Developer'})
print(result)