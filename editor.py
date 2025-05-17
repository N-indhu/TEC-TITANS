from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, set_seed

# ---------- Autocompletion Classes ----------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def _autocomplete(self, node, prefix, results):
        if node.is_end:
            results.append(prefix)
        for ch, child in node.children.items():
            self._autocomplete(child, prefix + ch, results)

    def autocomplete(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results = []
        self._autocomplete(node, prefix, results)
        return results

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(list)

    def train(self, corpus):
        tokens = corpus.split()
        for i in range(len(tokens) - self.n):
            key = tuple(tokens[i:i+self.n-1])
            self.ngrams[key].append(tokens[i+self.n-1])

    def predict(self, context):
        key = tuple(context.split()[-(self.n-1):])
        return self.ngrams.get(key, [])

class TFIDFAutocompleter:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.docs = documents
        self.X = self.vectorizer.fit_transform(documents)

    def suggest(self, query):
        query_vec = self.vectorizer.transform([query])
        similarities = (self.X * query_vec.T).toarray()
        ranked = sorted(zip(similarities, self.docs), reverse=True)
        return [doc for sim, doc in ranked if sim > 0]

class GPT2Autocompleter:
    def __init__(self):
        self.generator = pipeline('text-generation', model='gpt2')
        set_seed(42)

    def generate(self, prompt):
        results = self.generator(prompt, max_length=30, num_return_sequences=1)
        return results[0]['generated_text']

# ---------- CLI Fallback (since tkinter not supported) ----------
def main():
    corpus = [
        "hello world",
        "hi there",
        "how are you doing",
        "hello how are you",
        "house on the hill",
        "hover over the map",
        "helicopter is flying"
    ]

    trie = Trie()
    for sentence in corpus:
        for word in sentence.split():
            trie.insert(word)

    ngram = NgramModel(3)
    ngram.train(" ".join(corpus))

    tfidf = TFIDFAutocompleter(corpus)
    gpt2 = GPT2Autocompleter()

    print("\nChoose Autocompletion Method:\n1. Trie\n2. N-gram\n3. TF-IDF\n4. GPT-2\n")

    while True:
        choice = input("Enter method (1-4) or 'q' to quit: ").strip()
        if choice == 'q':
            break

        query = input("Enter your query: ").strip()

        if choice == '1':
            print("Suggestions:", trie.autocomplete(query))
        elif choice == '2':
            print("Next word suggestions:", ngram.predict(query))
        elif choice == '3':
            print("Relevant completions:", tfidf.suggest(query))
        elif choice == '4':
            print("Generated completion:", gpt2.generate(query))
        else:
            print("Invalid choice. Try again.")

if __name__ == '__main__':
    main()