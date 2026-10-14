# Shared settings for the whole project
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}   # index 0 is reserved for the CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1

RAW_H, RAW_W = 64, 384     # size of the raw crop we preprocess (6:1 aspect, like a serial-code strip)
IN_H, IN_W = 32, 192       # size fed to the network
MIN_LEN, MAX_LEN = 6, 12   # serial code length range
