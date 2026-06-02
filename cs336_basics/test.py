# import requests
# import json

# code = "7154"

# r = requests.get(
#     "https://copy-paste.online/api/v1/paste",
#     params={"code": code}
# )
# print(r.status_code)
# data = json.loads(r.text)
# print(len(data["message"]))
# # with open("vocab_from_api.json", "w") as f:
# #     json.dump(data, f, indent=4)
from cs336_basics.preprocess_data import preprocess

preprocess("data/owt.txt", "data/owt_train.bin")
