# disadvantage of structure output is Data validation

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#this is not working 
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# now use this 
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema,
)


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=2000,
    do_sample=False,
)
model = ChatHuggingFace(llm = llm) 

schema = [
    ResponseSchema(name="fact_1", description="First factual statement about the topic."),
    ResponseSchema(name="fact_2", description="Second factual statement about the topic."),
    ResponseSchema(name="fact_3", description="Third factual statement about the topic."),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template=(
        "Give exactly three accurate facts about {topic}.\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)
# prompt = template.invoke({'topic' : 'Black Hole'})
# result = model.invoke(prompt)
# print("RAW RESPONSE:")
# print(repr(result.content))
# final_result = parser.parse(result.content)

#with the help of chain 
chain = template | model | parser
result = chain.invoke({'topic' : 'Black Hole'})


print(result)