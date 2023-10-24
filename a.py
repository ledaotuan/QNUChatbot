from build_db import collection,client
results = collection.query(
        query_texts="Xin chào bạn",
        include=["documents"],
        n_results=5,
    )
results