from sentence_transformers import SentenceTransformer
import torch

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

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(movies, convert_to_tensor=True)  # Shape: (5, 384)

def recommend(query, k=2):
    # Compute similarities (1D tensor of shape (num_movies,))
    similarities = torch.cosine_similarity(
        model.encode(query, convert_to_tensor=True).unsqueeze(0), 
        embeddings, 
        dim=1
    )
    
    # Get top-k results in one go
    top_indices = torch.topk(similarities, k).indices
    
    print(f"Recommendations for '{query}':")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {movies[idx]} ({similarities[idx].item():.3f})")


# Test queries
print("=== Space war queries ===")
recommend("galactic war rebels empire")

print("\n=== Alien invasion ===")
recommend("alien invasion fight war")

print("\n=== Time + future ===")
recommend("time travel future war")
