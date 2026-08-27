from langchain_core.messages import SystemMessage , AIMessage , HumanMessage 
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    # max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
    # provider="auto"
)
model = ChatHuggingFace(llm = llm) 


messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="tell me about langchain"),
     
]


result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)