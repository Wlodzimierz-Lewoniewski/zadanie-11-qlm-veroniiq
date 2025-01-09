import math
from collections import Counter

def tokenize(text):
    return text.lower().replace(',', '').replace('.', '').split()

def qlm_with_smoothing(documents, query, lambda_val=0.5):
    # Tokenizacja zapytania
    query_tokens = tokenize(query)

    # Zbieranie unikalnych słów we wszystkich dokumentach
    all_tokens = []
    for doc in documents:
        all_tokens.extend(tokenize(doc))

    # Zbiór unikalnych słów (słownik V)
    vocab = set(all_tokens)
    vocab_size = len(vocab)

    # Przechowywanie wyników prawdopodobieństwa dla każdego dokumentu
    doc_probs = []

    for i, doc in enumerate(documents):
        # Tokenizacja dokumentu
        doc_tokens = tokenize(doc)
        doc_length = len(doc_tokens)

        # Liczenie wystąpień słów w dokumencie
        doc_counter = Counter(doc_tokens)

        # Obliczenie prawdopodobieństwa generowania zapytania z tego dokumentu
        prob = 0
        for word in query_tokens:
            word_count = doc_counter.get(word, 0)
            # P(w|D) z wygładzaniem
            word_prob = (word_count + lambda_val) / (doc_length + lambda_val * vocab_size)
            prob += math.log(word_prob)

        # Zapisujemy indeks dokumentu oraz jego prawdopodobieństwo
        doc_probs.append((i, prob))

    # Sortowanie dokumentów według prawdopodobieństwa malejąco, przy zachowaniu kolejności przy równych wartościach
    doc_probs.sort(key=lambda x: x[1], reverse=True)

    # Zwracamy posortowaną listę indeksów dokumentów
    return [doc[0] for doc in doc_probs]

if __name__ == "__main__":
    n_documents = int(input().strip())
    documents = [input().strip() for _ in range(n_docs)]
    query = input().strip()

    result = qlm_with_smoothing(documents, query, lambda_val=0.5)
    print(result)