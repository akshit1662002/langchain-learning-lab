from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader = DirectoryLoader(
    path='9_document_loader/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

doc = loader.load()

print(doc)