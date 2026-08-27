# i also use huggingface because they have free llm  and also i use openai if i have credit 

# here we gave user to write static prompt -> so user have more control on the prompt 
# so here we prepare a template for research paper (dynamic prompt) -> mtlb hamne prompt likh rakha hai uske ander jo key value hai vo user 
# apko provide kar rha hai 

# benefits of using prompt template over f strings

# 1- default validation -->> by default kuch validation hai jisse code development time par hi fat jayega run time par nhi fatega 
# 2- resuable -->> 
# 3- langchain ecosystem


# from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt

load_dotenv()

# model = ChatOpenAI(model="gpt-4" )

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    # max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
    # provider="auto"
)


model = ChatHuggingFace(llm = llm)


st.header("Research tool")


paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


template =load_prompt('template.json')


# # fill the placeholder
# prompt = template.invoke({
#     'paper_input' : paper_input, 
#     'style_input' : style_input,
#     'length_input' :length_input
# })



# # user_input = st.text_input("enter your prompt")

# if st.button('Summarize'):
#     result = model.invoke(prompt)
#     st.write(result.content)
#     # st.write("hello")


# here we create a chain


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
    'paper_input' : paper_input, 
    'style_input' : style_input,
    'length_input' :length_input
    }) 
    st.write(result.content)
