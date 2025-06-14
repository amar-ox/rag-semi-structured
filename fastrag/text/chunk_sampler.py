#    FastRAG: Efficient Retrieval Augmented Generation for Semi-structured Data
#    Copyright (C) 2024–2025 Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <https://www.gnu.org/licenses/>.


# fastrag/chunk_sampler.py

import string
import math
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from .utils import ensure_nltk_resources

ensure_nltk_resources()

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    words = word_tokenize(text)
    return words


def extract_keywords(text, n_clusters, n_top):
    documents = []
    for doc in text.split('\n'):
        documents.append(" ".join(preprocess_text(doc)))

    # Vectorize the documents
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # Extract terms from cluster centers
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    # Collect keywords for each cluster
    keywords = []
    for i in range(n_clusters):
        cluster_keywords = [terms[ind] for ind in order_centroids[i, :n_top]]
        keywords.extend(cluster_keywords)

    return set(keywords)


def tokenize_for_sampling(chunks, keywords):
    tokenized_chunks = []
    for chunk in chunks:
        tokens = preprocess_text(chunk)
        filtered_tokens = [token for token in tokens if token in keywords]
        tokenized_chunks.append(" ".join(filtered_tokens))
    return tokenized_chunks


def entropy(text):
    tokenized_text = text.split()
    word_counts = Counter(tokenized_text)
    total_words = sum(word_counts.values())
    entropy_value = -sum(
        (count / total_words) * math.log(count / total_words, 2)
        for count in word_counts.values()
    )
    return entropy_value


def select_minimum_chunks(tfidf_matrix, entropies, threshold=1.0):
    num_chunks, num_terms = tfidf_matrix.shape
    selected_indices = []
    covered_terms = set()

    while len(covered_terms) / num_terms < threshold:
        # Compute coverage gain for each chunk
        coverage_gains = []
        for i in range(num_chunks):
            if i not in selected_indices:
                new_terms = set(np.nonzero(tfidf_matrix[i].toarray())[1]) - covered_terms
                coverage_gain = len(new_terms) * entropies[i]
                coverage_gains.append((coverage_gain, i))

        # Select the chunk with the highest coverage gain
        if coverage_gains:
            best_chunk = max(coverage_gains)[1]
            selected_indices.append(best_chunk)
            covered_terms.update(np.nonzero(tfidf_matrix[best_chunk].toarray())[1])
        else:
            break

    return selected_indices


def get_sample_chunks(combined_chunks, threshold, keywords, evals=False):
    tokenized_chunks = tokenize_for_sampling(combined_chunks, keywords)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_chunks)
    entropies = np.array([entropy(chunk) for chunk in tokenized_chunks])
    selected_indices = select_minimum_chunks(tfidf_matrix, entropies, threshold)
    selected_chunks = [combined_chunks[i] for i in selected_indices]

    evaluation_chunks = []
    if evals and (len(combined_chunks) > len(selected_indices)):
        remaining_indices = [i for i in range(len(combined_chunks)) if i not in selected_indices]
        evaluation_indices = np.random.choice(remaining_indices, len(selected_chunks), replace=True)
        evaluation_chunks = [combined_chunks[i] for i in evaluation_indices]
    else:
        print(f"Info: Not enough remaining chunks ({len(combined_chunks) - len(selected_indices)}) to select evaluation chunks")

    return selected_chunks, evaluation_chunks

#############################
def find_optimal_parameters(chunks, desired_sample_size, n_clusters_range, n_top_range, return_evals=False):
    for n_clusters in n_clusters_range:
        for n_top in n_top_range:
            # print(f"Try parameters: n_clusters={n_clusters}, n_top={n_top}")
            keywords = extract_keywords("\n".join(chunks), n_clusters=n_clusters, n_top=n_top)
            sample_chunks_result, evals_chunks = get_sample_chunks(chunks, 1.0, keywords, return_evals)
            if len(sample_chunks_result) == desired_sample_size:
                return n_clusters, n_top, sample_chunks_result, evals_chunks
    return None, None, None, None

def sample_chunks(chunks, desired_sample_size, return_evals=False):
    n_clusters_range = range(2, 10)
    n_top_range = range(3, 10)

    n_clusters, n_top, sample_chunks_result, evals_chunks = find_optimal_parameters(
        chunks, desired_sample_size, n_clusters_range, n_top_range, return_evals
    )

    if sample_chunks_result is not None:
        print(f"Found parameters: n_clusters={n_clusters}, n_top={n_top}")
        return sample_chunks_result, evals_chunks
    else:
        return None, None

def sample_chunks_per_section(section_combined_chunks, desired_sample_size):
    n_clusters_range = range(1, 6)
    n_top_range = range(1, 10)
    #optimal_results = {}
    section_sample_chunks = {}

    for section, chunks in section_combined_chunks.items():
        n_clusters, n_top, sample_chunks_result, _ = find_optimal_parameters(
            chunks, desired_sample_size, n_clusters_range, n_top_range
        )

        if sample_chunks_result is not None:
            section_sample_chunks[section] = sample_chunks_result
            #optimal_results[section] = {
            #    "n_clusters": n_clusters,
            #    "n_top": n_top,
            #}
            print(f"Found parameters for section {section}: n_clusters={n_clusters}, n_top={n_top}")
        #else:
        #    optimal_results[section] = {"n_clusters": None, "n_top": None}

    return section_sample_chunks
