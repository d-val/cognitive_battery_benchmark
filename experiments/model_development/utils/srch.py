import torch
from collections import deque


def search():
    queue = deque([torch])
    while len(queue):
        cur = queue.popleft()
        for i in dir(cur):
            if "permute" in i:
                return i
            else:
                if "__" not in i:
                    queue.append(cur.__getattribute__(i))

print(search())