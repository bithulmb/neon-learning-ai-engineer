from dotenv import load_dotenv  
from sentence_transformers import SentenceTransformer,util


load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

#manual similarity search function
def find_closest_sentence(query, sentences):
    query_embedding = model.encode([query])
    sentence_embeddings = model.encode(sentences)

    closest_index = 0
    closest_similarity = -1

    for index,embedding in enumerate(sentence_embeddings):
        similarity = util.cos_sim(query_embedding, embedding)
        if similarity > closest_similarity:
            closest_similarity = similarity
            closest_index = index

    return sentences[closest_index]
    

    


def main():
    sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]

    query = input("Enter a query: ")

    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])

    result = util.semantic_search(query_embedding, sentence_embeddings, top_k=2)
    print(result)
    print("Top 2 closest sentences:")
    for match in result[0]:
        print(f"Sentence: {sentences[match['corpus_id']]}, Score: {match['score']:.4f}")    


if __name__ == "__main__":
    main()