import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohammadrahdanmofrad/landslide-risk-assessment-factors")

print("Path to dataset files:", path)