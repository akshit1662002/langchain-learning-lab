from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",  
    temperature=0.7,
    max_tokens=2000,
)

def word_count(text):
    return len(text.split())

model = llm

parser = StrOutputParser()


template = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

joke_generate_chain = RunnableSequence(template , model , parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'Word_count' : RunnableLambda(word_count)
})


# parallel_chain = RunnableParallel({
#     'joke' : RunnablePassthrough(),
#     'Word_count' : RunnableLambda(lambda x : len(x.split()))
# })
 


final_chain = RunnableSequence(joke_generate_chain , parallel_chain)

result = final_chain.invoke({'topic' : 'cricket'})

print(result)
print(result['joke'])
print(result['Word_count'])

