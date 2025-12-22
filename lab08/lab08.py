import numpy as np
from collections import Counter
import re

movies = [
    "Terminator: A cyborg assassin from the future hunts a woman destined to lead human resistance.",
    "The Matrix: A hacker discovers his reality is a simulation controlled by machines.",
    "Stargate: Military discovers ancient portal to other worlds and alien civilizations.",
    "The Fifth Element: Cab driver protects cosmic power from ancient evil in futuristic New York.",
    "Back to the Future: Teen travels through time in DeLorean, must fix timeline.",
    "Blade Runner: Future detective hunts rogue replicants in dystopian Los Angeles reality.",
    "Aliens: Colonial marines fight alien hive in future war on distant planet.",
    "Edge of Tomorrow: Soldier relives same day fighting alien invasion through time loop.",
    "Star Wars: Young pilot joins rebel alliance fighting evil empire in galactic civil war."
]

def simple_tokenize(text):
    return re.findall(r'\w+', text.lower())

# Build global vocab from all movies
all_words = set(w for movie in movies for w in simple_tokenize(movie))
vocab = list(all_words)
word_to_idx = {w: i for i, w in enumerate(vocab)}

# Create TF vectors (term frequency, normalized)
def tf_vector(text):
    words = simple_tokenize(text)
    if not words:
        return np.zeros(len(vocab))
    counts = Counter(words)
    vec = np.zeros(len(vocab))
    for w, c in counts.items():
        if w in word_to_idx:
            vec[word_to_idx[w]] = c / len(words)
    return vec

# Precompute all movie vectors
movie_vectors = np.array([tf_vector(movie) for movie in movies])

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def recommend(query, top_k=2):
    query_vec = tf_vector(query)
    sims = [cosine_sim(query_vec, movie_vectors[i]) for i in range(len(movies))]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    
    print("Recommendations:")
    for idx in top_idx:
        print(f"- {movies[idx]} (sim: {sims[idx]:.3f})")

# Test queries
print("=== Space war queries ===")
recommend("galactic war rebels empire")

print("\n=== Alien invasion ===")
recommend("alien invasion fight war")

print("\n=== Time + future ===")
recommend("time travel future war")
