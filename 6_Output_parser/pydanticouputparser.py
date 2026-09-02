
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=2000,
    do_sample=False,
)
model = ChatHuggingFace(llm = llm) 


class Person(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(gt=18 ,description='Age of the person')
    city : str = Field(description='Name of the city the person belong to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of the fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={ 'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place' : 'indian'})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser
final_result  = chain.invoke({'place' : 'indian'})
print(final_result)