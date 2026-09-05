from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('9_document_loader/dl-curriculum.pdf')
doc =loader.load()

print(doc)
# print(len(doc))
# print(type(doc))
print(doc[0].page_content)
print(doc[2].metadata)