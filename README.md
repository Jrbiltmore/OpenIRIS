
# OpenIRIS

OpenIRIS is an advanced iris authentication system that leverages modern techniques in computer vision, machine learning, and optimization to provide superior accuracy and speed.

## Directory Structure

OpenIRIS/
├── data/
│ ├── raw/
│ ├── processed/
│ └── models/
├── src/
│ ├── preprocessing/
│ │ ├── iris_segmentation.py
│ │ └── normalization.py
│ ├── feature_extraction/
│ │ ├── gabor_filters.py
│ │ └── deep_features.py
│ ├── matching/
│ │ ├── hamming_distance.py
│ │ └── neural_matching.py
│ ├── utils/
│ │ ├── logger.py
│ │ └── image_utils.py
│ └── main.py
├── tests/
│ ├── test_segmentation.py
│ ├── test_feature_extraction.py
│ ├── test_matching.py
│ └── test_pipeline.py
├── README.md
├── requirements.txt
└── setup.py

## Installation

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

To run the main pipeline, execute:
```bash
python src/main.py path/to/first/image.jpg path/to/second/image.jpg
```

## Testing

To run tests, execute:
```bash
pytest tests/
```
