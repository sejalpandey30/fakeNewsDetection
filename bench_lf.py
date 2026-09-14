import os
import time
import pandas as pd
from src.features import linguistic_features as lf

def main():
    print("Testing linguistic feature extraction speed...")
    texts = ["This is a breaking news test article with some clickbait words and exclamation marks!!"] * 50
    t0 = time.time()
    lf.extract_batch(texts)
    dt = time.time() - t0
    print(f"50 texts took {dt:.3f}s -> {dt/50*1000:.2f}ms per text")

if __name__ == '__main__':
    main()
