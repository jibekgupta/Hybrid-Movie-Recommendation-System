
"""
Hybrid Movie Recommendation System

A movie recommender that combines content-based and collaborative filtering
to provide better recommentaions than either approach alone.
"""

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def clean_title(title):
    """Clean up movie titles for better matching."""
    if not title:
        return ""
    
    # Convert to lowercase and remove punctuation
    cleaned = title.lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


def extract_year(title):
    """Extract year from movie title like 'Movie Name (1999)'."""
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return match.group(1)
    return ""


def find_common_genres(genres1, genres2, max_genres=3):
    """Find genres shared between two movies."""
    if pd.isna(genres1) or pd.isna(genres2):
        return ""
    
    # Split genres and find intersection
    set1 = set(str(genres1).split("|"))
    set2 = set(str(genres2).split("|"))
    common = set1.intersection(set2)
    
    # Remove empty or invalid genres
    common = [g for g in common if g and g != "(no genres listed)"]
    
    if common:
        return ", ".join(sorted(common)[:max_genres])
    else:
        return ""


def load_movie_data():
    """
    Load movie data from CSV files or use demo data if files don't exist.
    
    Returns:
        tuple: (movies DataFrame, ratings DataFrame)
    """
    try:
        # Try to load actual MovieLens data
        if os.path.exists("movies.csv") and os.path.exists("ratings.csv"):
            print("Loading MovieLens data from CSV files...")
            movies_df = pd.read_csv("movies.csv")
            ratings_df = pd.read_csv("ratings.csv")
            print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
            return movies_df, ratings_df
            
    except Exception as e:
        print(f"Error loading CSV files: {e}")
    
    # Fall back to demo data
    print("Using built-in demo data...")
    
    # Create small demo dataset for testing
    demo_movies = pd.DataFrame({
        "movieId": range(1, 11),
        "title": [
            "Toy Story (1995)",
            "Jumanji (1995)", 
            "Heat (1995)",
            "Sabrina (1995)",
            "GoldenEye (1995)",
            "Seven (Se7en) (1995)",
            "Usual Suspects, The (1995)",
            "Batman Forever (1995)",
            "Grumpier Old Men (1995)",
            "Father of the Bride Part II (1995)"
        ],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Action|Crime|Thriller",
            "Comedy|Romance",
            "Action|Adventure|Thriller",
            "Crime|Mystery|Thriller",
            "Crime|Mystery|Thriller", 
            "Action|Adventure|Comedy|Crime",
            "Comedy|Romance",
            "Comedy"
        ]
    })
    
    demo_ratings = pd.DataFrame({
        "userId": [1,1,1,2,2,2,3,3,3,3,4,4,5,5],
        "movieId": [1,2,6,2,3,7,1,6,7,5,8,2,4,9],
        "rating": [4.0,3.5,4.5,4.0,4.0,4.5,5.0,4.0,4.5,4.0,3.0,2.5,4.0,3.0]
    })
    
    return demo_movies, demo_ratings


def build_content_features(movies_df):
    """
    Build TF-IDF features from movie genres.
    
    Args:
        movies_df: DataFrame with movie information
        
    Returns:
        tuple: (TfidfVectorizer, feature matrix)
    """
    # Prepare genre text for TF-IDF
    genre_docs = movies_df["genres"].fillna("").str.replace("|", " ")
    
    # Create TF-IDF vectorizer
    # Using simple settings that work well for short genre lists
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),  # Include single words and pairs
        min_df=1,            # Keep all terms (small dataset)
        max_df=0.9,          # Remove very common terms
        token_pattern=r'\b\w+\b'  # Match whole words
    )
    
    # Fit and transform
    feature_matrix = tfidf.fit_transform(genre_docs)
    
    return tfidf, feature_matrix


class CollaborativeFilter:
    """
    Item-based collaborative filtering using cosine similarity.
    """
    
    def __init__(self, ratings_df, movie_mapping, neighbors=50):
        """
        Initialize collaborative filter.
        
        Args:
            ratings_df: DataFrame with user ratings
            movie_mapping: Dict mapping movieId to matrix index
            neighbors: Maximum number of neighbors to consider
        """
        self.movie_mapping = movie_mapping
        
        # Create user mapping for consistency
        unique_users = ratings_df["userId"].unique()
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        
        # Map ratings to matrix indices
        mapped_ratings = ratings_df.copy()
        mapped_ratings["movie_idx"] = mapped_ratings["movieId"].map(movie_mapping)
        mapped_ratings["user_idx"] = mapped_ratings["userId"].map(self.user_mapping)
        
        # Remove unmapped entries
        mapped_ratings = mapped_ratings.dropna(subset=["movie_idx", "user_idx"])
        mapped_ratings = mapped_ratings.astype({
            "movie_idx": int, 
            "user_idx": int
        })
        
        # Build item-user matrix (sparse for efficiency)
        from scipy.sparse import coo_matrix
        
        num_movies = len(movie_mapping)
        num_users = len(self.user_mapping)
        
        item_user_matrix = coo_matrix(
            (mapped_ratings["rating"], 
             (mapped_ratings["movie_idx"], mapped_ratings["user_idx"])),
            shape=(num_movies, num_users)
        ).tocsr()
        
        self.item_user_matrix = item_user_matrix
        
        # Set up nearest neighbors model
        # Handle case where we have fewer movies than requested neighbors
        actual_neighbors = min(neighbors, max(1, num_movies))
        
        self.nn_model = NearestNeighbors(
            metric='cosine', 
            algorithm='brute',
            n_neighbors=actual_neighbors
        )
        self.nn_model.fit(item_user_matrix)
    
    def get_similar_movies(self, movie_id, num_similar=20):
        """
        Find movies similar to the given movie.
        
        Args:
            movie_id: ID of the target movie
            num_similar: Number of similar movies to return
            
        Returns:
            list: Tuples of (movie_id, similarity_score)
        """
        movie_idx = self.movie_mapping.get(movie_id)
        if movie_idx is None:
            return []
        
        # Find similar items
        num_neighbors = min(num_similar + 1, self.item_user_matrix.shape[0])
        distances, indices = self.nn_model.kneighbors(
            self.item_user_matrix[movie_idx], 
            n_neighbors=num_neighbors
        )
        
        # Convert distances to similarities and create results
        results = []
        reverse_mapping = {idx: mid for mid, idx in self.movie_mapping.items()}
        
        for distance, idx in zip(distances.flatten(), indices.flatten()):
            if idx == movie_idx:  # Skip the movie itself
                continue
                
            similarity = 1.0 - distance  # Convert distance to similarity
            similar_movie_id = reverse_mapping[idx]
            results.append((similar_movie_id, similarity))
        
        return results[:num_similar]


class HybridMovieRecommender:
    """
    Main recommender system combining content-based and collaborative filtering.
    """
    
    def __init__(self, movies_df, ratings_df):
        """
        Initialize the hybrid recommender.
        
        Args:
            movies_df: DataFrame with movie information
            ratings_df: DataFrame with user ratings
        """
        # Clean and prepare data
        self.movies = movies_df.drop_duplicates("movieId").copy()
        self.movies = self.movies[["movieId", "title", "genres"]].reset_index(drop=True)
        self.ratings = ratings_df[["userId", "movieId", "rating"]].copy()
        
        # Create movie index mapping
        self.movie_to_index = {
            movie_id: idx for idx, movie_id in enumerate(self.movies["movieId"])
        }
        self.index_to_movie = {idx: mid for mid, idx in self.movie_to_index.items()}
        
        print("Building content-based features...")
        self.tfidf_vectorizer, self.content_features = build_content_features(self.movies)
        
        # Set up content-based nearest neighbors
        max_neighbors = min(100, len(self.movies))
        self.content_nn = NearestNeighbors(
            metric='cosine',
            algorithm='brute', 
            n_neighbors=max_neighbors
        )
        self.content_nn.fit(self.content_features)
        
        # Set up collaborative filtering
        print("Building collaborative filtering model...")
        if len(self.ratings) > 0:
            self.cf_model = CollaborativeFilter(
                self.ratings, 
                self.movie_to_index,
                neighbors=100
            )
        else:
            self.cf_model = None
            print("No ratings data available - content-only mode")
        
        # Add normalized titles for search
        self.movies["normalized_title"] = self.movies["title"].apply(clean_title)
        
        print("Recommender system ready!")
    
    def find_content_similar(self, movie_id, num_results=50):
        """Get content-based recommendations."""
        movie_idx = self.movie_to_index.get(movie_id)
        if movie_idx is None:
            return []
        
        num_neighbors = min(num_results + 1, self.content_features.shape[0])
        distances, indices = self.content_nn.kneighbors(
            self.content_features[movie_idx],
            n_neighbors=num_neighbors
        )
        
        results = []
        for distance, idx in zip(distances.flatten(), indices.flatten()):
            if idx == movie_idx:
                continue
                
            similarity = 1.0 - distance
            similar_movie_id = self.index_to_movie[idx]
            results.append((similar_movie_id, similarity))
        
        return results[:num_results]
    
    def find_collaborative_similar(self, movie_id, num_results=50):
        """Get collaborative filtering recommendations."""
        if self.cf_model is None:
            return []
        return self.cf_model.get_similar_movies(movie_id, num_results)
    
    def get_hybrid_recommendations(self, movie_id, num_recommendations=20, content_weight=0.5):
        """
        Get hybrid recommendations combining content and collaborative filtering.
        
        Args:
            movie_id: Target movie ID
            num_recommendations: Number of recommendations to return
            content_weight: Weight for content-based scores (0-1)
            
        Returns:
            list: Tuples of (movie_id, combined_score, explanation)
        """
        # Get recommendations from both approaches
        content_recs = dict(self.find_content_similar(movie_id, 200))
        collab_recs = dict(self.find_collaborative_similar(movie_id, 200))
        
        # Combine candidate movies
        all_candidates = set(content_recs.keys()) | set(collab_recs.keys())
        
        if not all_candidates:
            return []
        
        # Normalize scores to 0-1 range
        def normalize_scores(score_dict):
            if not score_dict:
                return {}
            max_score = max(score_dict.values())
            if max_score == 0:
                return {k: 0 for k in score_dict}
            return {k: v / max_score for k, v in score_dict.items()}
        
        content_normalized = normalize_scores(content_recs)
        collab_normalized = normalize_scores(collab_recs)
        
        # Calculate hybrid scores
        hybrid_scores = {}
        for candidate_id in all_candidates:
            content_score = content_normalized.get(candidate_id, 0)
            collab_score = collab_normalized.get(candidate_id, 0)
            
            # Weighted combination
            hybrid_score = (content_weight * content_score + 
                          (1 - content_weight) * collab_score)
            hybrid_scores[candidate_id] = hybrid_score
        
        # Sort by score and prepare results
        ranked_movies = sorted(hybrid_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        
        # Add explanations
        target_movie = self.movies[self.movies["movieId"] == movie_id]
        if len(target_movie) == 0:
            return []
            
        target_genres = target_movie["genres"].iloc[0]
        
        results = []
        for candidate_id, score in ranked_movies[:num_recommendations]:
            candidate_movie = self.movies[self.movies["movieId"] == candidate_id]
            if len(candidate_movie) == 0:
                continue
                
            candidate_genres = candidate_movie["genres"].iloc[0]
            common_genres = find_common_genres(target_genres, candidate_genres)
            
            if common_genres:
                explanation = f"Shared genres: {common_genres}"
            else:
                explanation = "Similar user preferences"
                
            results.append((candidate_id, float(score), explanation))
        
        return results
    
    def search_movies(self, query, max_results=50):
        """Search for movies by title."""
        if not query.strip():
            return self.movies.head(max_results)
        
        normalized_query = clean_title(query)
        matches = self.movies[
            self.movies["normalized_title"].str.contains(normalized_query, na=False)
        ]
        
        return matches.head(max_results)
    
    def find_movie_by_title(self, title):
        """Find a movie ID by its title."""
        normalized_title = clean_title(title)
        
        # Try exact match first
        exact_matches = self.movies[
            self.movies["normalized_title"] == normalized_title
        ]
        
        if len(exact_matches) == 1:
            return int(exact_matches["movieId"].iloc[0])
        
        # Fall back to partial match
        partial_matches = self.search_movies(title, max_results=1)
        if len(partial_matches) > 0:
            return int(partial_matches["movieId"].iloc[0])
        
        return None
    
    def get_user_rated_movies(self, user_id):
        """Get set of movies already rated by a user."""
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        return set(user_ratings["movieId"].tolist())


class MovieRecommenderGUI:
    """
    Tkinter-based GUI for the movie recommender system.
    """
    
    def __init__(self, recommender):
        """
        Initialize the GUI.
        
        Args:
            recommender: HybridMovieRecommender instance
        """
        self.recommender = recommender
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Hybrid Movie Recommender")
        self.root.geometry("1000x700")
        
        self._setup_ui()
        self._populate_movie_list()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Left panel for search and controls
        left_panel = ttk.Frame(self.root)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Search section
        ttk.Label(left_panel, text="Search Movies", 
                 font=("Arial", 12, "bold")).pack(anchor="w")
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(left_panel, textvariable=self.search_var, width=45)
        search_entry.pack(anchor="w", pady=(5, 10))
        search_entry.bind("<KeyRelease>", self._on_search_changed)
        
        # Movie list
        self.movie_listbox = tk.Listbox(left_panel, height=15, width=60)
        self.movie_listbox.pack(anchor="w", pady=(0, 10))
        self.movie_listbox.bind("<<ListboxSelect>>", self._on_movie_selected)
        
        # Control panel
        controls = ttk.LabelFrame(left_panel, text="Recommendation Settings")
        controls.pack(anchor="w", fill=tk.X, pady=(0, 10))
        
        # Number of recommendations
        ttk.Label(controls, text="Number of recommendations:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.num_recs = tk.IntVar(value=10)
        num_spinner = ttk.Spinbox(controls, from_=5, to=50, 
                                 textvariable=self.num_recs, width=10)
        num_spinner.grid(row=0, column=1, sticky="w", padx=5)
        
        # Content weight slider
        ttk.Label(controls, text="Content vs Collaborative (α):").grid(
            row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.content_weight = tk.DoubleVar(value=0.5)
        weight_scale = ttk.Scale(controls, from_=0.0, to=1.0, 
                               variable=self.content_weight, 
                               orient=tk.HORIZONTAL, length=200)
        weight_scale.grid(row=1, column=1, sticky="w", padx=5)
        
        # User filtering
        ttk.Label(controls, text="User ID (optional):").grid(
            row=2, column=0, sticky="w", padx=5, pady=5)
        
        self.user_id_var = tk.StringVar()
        user_entry = ttk.Entry(controls, textvariable=self.user_id_var, width=15)
        user_entry.grid(row=2, column=1, sticky="w", padx=5)
        
        self.filter_rated = tk.BooleanVar(value=True)
        filter_check = ttk.Checkbutton(
            controls, 
            text="Hide movies user has already rated",
            variable=self.filter_rated
        )
        filter_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(controls)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Get Recommendations", 
                  command=self._get_recommendations).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Results", 
                  command=self._clear_results).pack(side=tk.LEFT, padx=5)
        
        # Right panel for results
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Selected movie display
        self.selected_movie = tk.StringVar(value="Selected Movie: None")
        ttk.Label(right_panel, textvariable=self.selected_movie,
                 font=("Arial", 11, "bold")).pack(anchor="w")
        
        # Status display
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right_panel, textvariable=self.status_var,
                 foreground="gray").pack(anchor="w", pady=(0, 10))
        
        # Results table
        columns = ("Title", "Year", "Score", "Reason")
        self.results_tree = ttk.Treeview(right_panel, columns=columns, 
                                        show="headings", height=25)
        
        # Configure columns
        self.results_tree.heading("Title", text="Movie Title")
        self.results_tree.heading("Year", text="Year") 
        self.results_tree.heading("Score", text="Score")
        self.results_tree.heading("Reason", text="Why Recommended")
        
        self.results_tree.column("Title", width=350, anchor="w")
        self.results_tree.column("Year", width=70, anchor="center")
        self.results_tree.column("Score", width=80, anchor="e")
        self.results_tree.column("Reason", width=200, anchor="w")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, 
                                 command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _populate_movie_list(self):
        """Fill the movie list with all available movies."""
        movies = self.recommender.movies.head(200)  # Show first 200 movies
        self.movie_listbox.delete(0, tk.END)
        
        for _, movie in movies.iterrows():
            movie_id = int(movie["movieId"])
            title = movie["title"]
            self.movie_listbox.insert(tk.END, f"{movie_id} — {title}")
    
    def _on_search_changed(self, event=None):
        """Handle search text changes."""
        query = self.search_var.get()
        search_results = self.recommender.search_movies(query, max_results=100)
        
        self.movie_listbox.delete(0, tk.END)
        for _, movie in search_results.iterrows():
            movie_id = int(movie["movieId"])
            title = movie["title"]
            self.movie_listbox.insert(tk.END, f"{movie_id} — {title}")
    
    def _on_movie_selected(self, event=None):
        """Handle movie selection from the list."""
        selection = self.movie_listbox.curselection()
        if not selection:
            return
        
        selected_text = self.movie_listbox.get(selection[0])
        title = selected_text.split("—", 1)[1].strip()
        self.selected_movie.set(f"Selected Movie: {title}")
        self.status_var.set("Ready to get recommendations")
    
    def _clear_results(self):
        """Clear the results table."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.status_var.set("Results cleared")
    
    def _get_recommendations_async(self):
        """Get recommendations in a separate thread to avoid freezing UI."""
        try:
            # Get selected movie
            selection = self.movie_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a movie first!")
                self.status_var.set("No movie selected")
                return
            
            selected_text = self.movie_listbox.get(selection[0])
            movie_id = int(selected_text.split("—")[0].strip())
            
            # Get parameters
            num_recs = self.num_recs.get()
            weight = self.content_weight.get()
            
            # Get recommendations
            recommendations = self.recommender.get_hybrid_recommendations(
                movie_id, 
                num_recommendations=num_recs * 2,  # Get extra for filtering
                content_weight=weight
            )
            
            # Apply user filtering if requested
            user_id_text = self.user_id_var.get().strip()
            rated_movies = set()
            
            if user_id_text and self.filter_rated.get():
                try:
                    user_id = int(user_id_text)
                    rated_movies = self.recommender.get_user_rated_movies(user_id)
                except ValueError:
                    pass  # Invalid user ID, ignore filtering
            
            # Clear previous results
            self._clear_results()
            
            # Show results
            count = 0
            for movie_id, score, reason in recommendations:
                if user_id_text and movie_id in rated_movies:
                    continue  # Skip movies user has already rated
                
                movie_info = self.recommender.movies[
                    self.recommender.movies["movieId"] == movie_id
                ]
                
                if len(movie_info) == 0:
                    continue
                    
                movie_row = movie_info.iloc[0]
                title = movie_row["title"]
                year = extract_year(title)
                
                self.results_tree.insert("", tk.END, values=(
                    title, year, f"{score:.3f}", reason
                ))
                
                count += 1
                if count >= num_recs:
                    break
            
            if count == 0:
                self.status_var.set("No recommendations found")
            else:
                self.status_var.set(
                    f"Found {count} recommendations (α={weight:.2f})"
                )
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error getting recommendations: {e}")
    
    def _get_recommendations(self):
        """Start recommendation process."""
        self.status_var.set("Getting recommendations...")
        # Run in separate thread to keep UI responsive
        thread = threading.Thread(target=self._get_recommendations_async)
        thread.daemon = True
        thread.start()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the application."""
    print("Loading movie data...")
    movies_df, ratings_df = load_movie_data()
    
    print("Cleaning data...")
    # Basic data cleaning
    movies_df = movies_df.dropna(subset=["movieId", "title"])
    ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"])
    
    # Ensure correct data types
    movies_df["movieId"] = movies_df["movieId"].astype(int)
    ratings_df["movieId"] = ratings_df["movieId"].astype(int)
    ratings_df["userId"] = ratings_df["userId"].astype(int)
    ratings_df["rating"] = ratings_df["rating"].astype(float)
    
    print("Building recommender system...")
    recommender = HybridMovieRecommender(movies_df, ratings_df)
    
    print("Starting GUI...")
    app = MovieRecommenderGUI(recommender)
    app.run()


if __name__ == "__main__":
    main()