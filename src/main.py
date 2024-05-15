
import os
from preprocessing.iris_segmentation import segment_iris
from preprocessing.normalization import normalize_iris
from feature_extraction.gabor_filters import apply_gabor_filters
from feature_extraction.deep_features import DeepFeatureExtractor
from matching.hamming_distance import hamming_distance
from matching.neural_matching import NeuralMatcher
from utils.image_utils import load_image, save_image

def main(image_path1, image_path2):
    # Load images
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    
    # Segment iris
    iris1 = segment_iris(image1)
    iris2 = segment_iris(image2)
    
    # Normalize iris
    norm_iris1 = normalize_iris(iris1)
    norm_iris2 = normalize_iris(iris2)
    
    # Feature extraction
    gabor_features1 = apply_gabor_filters(norm_iris1)
    gabor_features2 = apply_gabor_filters(norm_iris2)
    
    deep_extractor = DeepFeatureExtractor()
    deep_features1 = deep_extractor.extract_features(norm_iris1)
    deep_features2 = deep_extractor.extract_features(norm_iris2)
    
    # Matching
    hamming_dist = hamming_distance(gabor_features1, gabor_features2)
    neural_matcher = NeuralMatcher()
    neural_matcher.eval()
    similarity = neural_matcher(deep_features1.unsqueeze(0), deep_features2.unsqueeze(0))
    
    # Output results
    print(f"Hamming Distance: {hamming_dist}")
    print(f"Neural Network Similarity: {similarity.item()}")

if __name__ == "__main__":
    main("path/to/first/image.jpg", "path/to/second/image.jpg")
