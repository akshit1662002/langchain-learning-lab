from langchain_core.prompts import ChatPromptTemplate 


# this will not work in case of chatprompt template 
# chat_template = ChatPromptTemplate([
#     SystemMessage(content='You are a helpful {domain} expert'), 
#     HumanMessage(content='Explain in simple terms , what is {topic}')
# ])


chat_template = ChatPromptTemplate([
    ('system' , 'You are a helpful {domain} expert'),
    ('human' , 'Explain in simple terms , What is {topic}')
])


prompt = chat_template.invoke({
    'domain' : 'cricket',
    'topic' : 'Wicket'
})


print(prompt)