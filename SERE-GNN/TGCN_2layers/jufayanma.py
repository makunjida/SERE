import os
import pickle
import string
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
from tqdm import tqdm

def calculate_distance(dependencies, num_words):
    distances = [[float('inf')] * num_words for _ in range(num_words)]
    for i in range(num_words):
        distances[i][i] = 0  # 自距离为0
    
    for dep in dependencies:
        rel, gov, dep = dep
        # 确保索引在有效范围内
        if 0 < gov <= num_words and 0 < dep <= num_words:
            gov_index = gov - 1
            dep_index = dep - 1
            distances[gov_index][dep_index] = 1
            distances[dep_index][gov_index] = 1  # 双向，如果需要无向距离

    # 使用Floyd-Warshall算法计算所有对最短路径
    for k in range(num_words):
        for i in range(num_words):
            for j in range(num_words):
                if distances[i][j] > distances[i][k] + distances[k][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
    return distances


def create_mask_matrix(distances, threshold=4):
    num_words = len(distances)
    mask_matrix = [[0 if distances[i][j] > threshold else 1 for j in range(num_words)] for i in range(num_words)]
    return mask_matrix