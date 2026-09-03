from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()
llm = HuggingFaceEndpoint(
   repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    # max_new_tokens=1500,
    # do_sample=False
)
model = ChatHuggingFace(llm = llm) 


# 1st prompt  -> Detail report

template1 = PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=['topic']
)
# 2nd prompt  -> summary 
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)


prompt1 = template1.invoke({'topic' : 'Black hole'})

result = model.invoke(prompt1)


prompt2 = template2.invoke({'text' : result.content})

finalResult = model.invoke(prompt2)

print(finalResult.content)