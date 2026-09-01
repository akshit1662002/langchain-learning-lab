# the biggest disadvantage of json output parser is it does not enforce shcema means how model return result
# in which json format 

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()
llm = HuggingFaceEndpoint(
   repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    # max_new_tokens=1500,
    # do_sample=False
)
model = ChatHuggingFace(llm = llm) 


parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age, city of the fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

template = PromptTemplate(
    template='Give me 5 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# print(final_result)
# print(final_result['name'])
# print(type((final_result )))

#so we can also do with the help of chain 
chain = template | model | parser
result = chain.invoke({'topic' : 'black hole'})
print(result)

