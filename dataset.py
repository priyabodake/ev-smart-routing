import kagglehub

# Download latest version
path = kagglehub.dataset_download("markbai/traffic-metr-la")

print("Path to dataset files:", path)