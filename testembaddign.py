import chromadb
col = chromadb.PersistentClient(path="index/chroma").get_collection("docs")
print(col.count())
print(col.peek(1)["metadatas"])