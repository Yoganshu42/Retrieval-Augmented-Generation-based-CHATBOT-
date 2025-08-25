import subprocess
import sys
from vector_db import search_faiss

def main():
    # print("Hello from my-rag-project!")
    
    # Launching Streamlit app directly with help of subprocess
    print("🚀 Launching Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
