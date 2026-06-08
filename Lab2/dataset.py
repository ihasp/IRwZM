import kagglehub
import os

# Download latest version
folder_path = kagglehub.dataset_download("zfturbo/measurements-of-urine-ph")
print("Path to dataset files:", folder_path)
print(os.listdir(folder_path))

path = os.path.join(folder_path, "ph_v1_days.csv")
