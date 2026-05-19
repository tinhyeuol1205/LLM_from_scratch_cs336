from train_bpe import train_bpe
import os
import json

def main():
    input_path = os.path.join(os.path.dirname(__file__), "/home/arisu/CV/LLM_from_scratch/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(len(vocab))
    print(len(merges))
    # vocab_json = {str(token): str(idx) for idx, token in vocab.items()}
    # merges_json = [[str(p1), str(p2)] for (p1, p2) in merges]
    # # Lưu vocab
    # with open("vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(vocab_json, f, indent=4, ensure_ascii=False)
        
    # # Lưu merges
    # with open("merges.json", "w", encoding="utf-8") as f:
    #     json.dump(merges_json, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()