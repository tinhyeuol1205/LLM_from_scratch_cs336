import os
import typing
from cs336_basics.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm

def preprocess(data_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], output_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", ["<|endoftext|>"])
    chunk_size = 100_000_000 # 100_000_000 tokens
    with open(data_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    with open(data_path, "r", encoding="utf-8") as fin, open(output_path, 'wb') as fout:
        tokens_buffer = []
        fin_with_progress = tqdm(fin, total=total_lines, desc="Encoding lines")
        for tok in tokenizer.encode_iterable(fin_with_progress):
            tokens_buffer.append(tok)
            if len(tokens_buffer) >= chunk_size:
                arr = np.array(tokens_buffer, dtype=np.uint16)
                fout.write(arr.tobytes())
                tokens_buffer.clear()
        
        if len(tokens_buffer) > 0:
            arr = np.array(tokens_buffer, dtype=np.uint16)
            fout.write(arr.tobytes())
        tokens_buffer.clear()
        
