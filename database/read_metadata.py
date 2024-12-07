import faiss
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python inspect_faiss.py <path_to_faiss_file>")
    sys.exit(1)

faiss_file = sys.argv[1]
# Check if the file exists
if not os.path.isfile(faiss_file):
    print(f"File {faiss_file} does not exist.")
    sys.exit(1)

# Get the file size
file_size = os.path.getsize(faiss_file)
print(f"Size of the FAISS file: {file_size / (1024 * 1024):.2f} MB")

try:
    # Load the FAISS index
    index = faiss.read_index(faiss_file)
    
    # Print index information
    print(f"FAISS index loaded from {faiss_file}")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Dimension of vectors: {index.d}")
    print(f"Metric type: {index.metric_type}")

except Exception as e:
    print(f"Error reading FAISS file: {e}")
