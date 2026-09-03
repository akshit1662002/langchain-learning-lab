from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import   RunnableBranch , RunnableLambda
from pydantic import BaseModel , Field
from typing import Literal

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=2000,
    do_sample=False,
)

model = ChatHuggingFace(llm = llm) 

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
# result = classifier_chain.invoke({'feedback' : 'this is the  good product'})

# print(result)
# print(result.sentiment)

# branch_chain =  RunnableBranch(
#     (condition1 , chain1)
#     (condition2 , chain2)
#     (default chain)
# )


branch_chain =  RunnableBranch(
    (lambda x:x.sentiment == 'positive'  , prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative'  , prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not found sentiment' )
)


final_chain = classifier_chain | branch_chain

result  = final_chain.invoke({'feedback' : 'this is the beautiful  phone'})

print(result)