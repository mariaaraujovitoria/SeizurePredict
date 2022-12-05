
import os

SAMPLE_RATE_DOWNSAMPLE = int(0.1*int(os.environ["SAMPLING_RATE"]))
PATIENCES = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
LEN_WINDOW = int(os.environ["SAMPLING_RATE"])*int(os.environ["SEC"])
LEN_WINDOW_DOWNSAMPLE = SAMPLE_RATE_DOWNSAMPLE*os.environ["SEC"]
THRESHOLD = 2 * os.environ["SAMPLING_RATE"]
