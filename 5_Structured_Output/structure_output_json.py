from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from  typing import TypedDict , Annotated , Optional
load_dotenv()
llm = HuggingFaceEndpoint(
   repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=1500,
    do_sample=False
)
model = ChatHuggingFace(llm = llm) 

#schema
# with typedict
# class Review(TypedDict):
#     key_themes : Annotated[list[str], "Write down all the keys themes discussed in the review in the list"]
#     summary : Annotated[str , "A brief summary of the review"]
#     sentiment : Annotated[str , "Return sentiment of the review either negative positive or neutral"]
#     pros : Annotated[Optional[list[str]] , "Write down all the pros inside the list"]
#     cons : Annotated[Optional[list[str]] , "Write down all the cons inside the list"]
#     name : Annotated[Optional[str] , "Write the name of the reviewer"]

#with pydantic
# class Review(BaseModel):
#     key_themes : list[str] = Field(description='Write down all the keys themes discussed in the review in the list')
#     summary  :  str =  Field(description='A brief summary of the review')
#     sentiment : Literal['pos', 'neg'] = Field(description='Return sentiment of the review either negative positive or neutral')
#     pros : Optional[list[str]] = Field(default=None, description='Write down all the pros inside the list')
#     cons : Optional[list[str]] = Field(default=None, description='Write down all the cons inside the list')
#     name : Optional[str] = Field(default=None , description='Write the name of the reviewer')


#with json

# schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(
 json_schema
)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Akshit tyagi
""")

# this will not work here , as in video they use openai , and i use hugging face
# print(type(result))
# print(result['summary'])
# print(result['sentiment'])

# print(result)
print(result)
