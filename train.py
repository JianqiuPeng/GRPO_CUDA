import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rl_zoo3.train import train

if __name__ == "__main__":
    train()
