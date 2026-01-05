import os

DATA_ROOT = "./data"
SAVE_DIR = "./checkpoints"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 600
WARMUP_EPOCHS = 0
LEARNING_RATE = 1e-4

SIGNAL_GAIN = 100.0
GLOBAL_MAX_THRESH = 2000
Z_DIM = 3200
KERNEL_SIZE = 9
PADDING = KERNEL_SIZE // 2

VIDEO_CONFIG = {
    "video_01": {"gaps": [5, 10, 15, 20], "min_th": 50},
    "video_02": {"gaps": [5, 10, 15, 20], "min_th": 80},
    "video_03": {"gaps": [5, 10, 15, 20], "min_th": 40},
    "video_04": {"gaps": [5, 7, 10, 15, 20], "min_th": 230},
    "video_06": {"gaps": [5, 10, 15, 20], "min_th": 266},
    "video_07": {"gaps": [5, 10, 15, 20], "min_th": 60},
    "video_08": {"gaps": [5, 10, 15, 20], "min_th": 88},
    "video_09": {"gaps": [5, 10, 15, 20], "min_th": 163},
    "video_10": {"gaps": [5, 10, 15, 20], "min_th": 350},
    "video_11": {"gaps": [3, 5, 10, 15],  "min_th": 110},
    "video_12": {"gaps": [3, 7, 11, 15, 20], "min_th": 330},
    "video_13": {"gaps": [5, 10, 15, 20], "min_th": 230},
    "video_14": {"gaps": [5, 10, 15, 20], "min_th": 150},
    "video_15": {"gaps": [5, 10, 15, 20], "min_th": 50},
    "video_16": {"gaps": [3, 5, 8, 12, 15], "min_th": 20},
    "video_17": {"gaps": [5, 10, 13, 18], "min_th": 150},
    "video_18": {"gaps": [4, 8, 12, 16],   "min_th": 250},
    "video_19": {"gaps": [5, 10, 15, 20], "min_th": 70},
    "video_20": {"gaps": [5, 7, 10, 15, 20], "min_th": 120},
    "video_21": {"gaps": [5, 10, 15, 20], "min_th": 134},
    "video_22": {"gaps": [5, 10, 15, 20], "min_th": 275},
    "video_23": {"gaps": [5, 10, 15, 20], "min_th": 85},
    "video_24": {"gaps": [5, 10, 15, 20], "min_th": 100},
    "video_25": {"gaps": [5, 10, 15, 20], "min_th": 123},
    "video_26": {"gaps": [5, 10, 15, 20], "min_th": 320},
    "video_27": {"gaps": [5, 10, 15, 20], "min_th": 140},
    "video_28": {"gaps": [5, 10, 15, 20], "min_th": 230},
    "video_29": {"gaps": [5, 10, 15, 20], "min_th": 270},
    "video_30": {"gaps": [5, 10, 15, 20], "min_th": 160},
    "video_31": {"gaps": [5, 10, 15, 20], "min_th": 120},
    "video_32": {"gaps": [5, 7, 10, 15, 20], "min_th": 92},
    "video_34": {"gaps": [5, 10, 15, 20], "min_th": 180},
    "video_35": {"gaps": [5, 10, 15, 20], "min_th": 36},
    "video_36": {"gaps": [5, 10, 15, 20], "min_th": 360},
    "video_37": {"gaps": [5, 10, 15, 20], "min_th": 193},
    "video_38": {"gaps": [3, 5, 10, 15, 20], "min_th": 500},
    "video_39": {"gaps": [5, 10, 15, 20], "min_th": 70},
    "video_40": {"gaps": [5, 10, 15, 20], "min_th": 100},
    "video_41": {"gaps": [5, 10, 15, 20], "min_th": 165},
    "video_42": {"gaps": [5, 7, 10, 15, 20], "min_th": 315},
    "video_43": {"gaps": [5, 10, 15, 20], "min_th": 120},
    "video_44": {"gaps": [5, 7, 10, 15, 20], "min_th": 400},
    "video_45": {"gaps": [5, 10, 15, 20], "min_th": 160},
    "video_46": {"gaps": [5, 7, 10, 15, 20], "min_th": 28},
    "video_47": {"gaps": [5, 7, 10, 15, 20], "min_th": 37},
    "video_48": {"gaps": [5, 10, 15, 20], "min_th": 243},
    "video_49": {"gaps": [5, 10, 15, 20], "min_th": 190},
    "video_50": {"gaps": [5, 10, 15, 20], "min_th": 370},
    "video_51": {"gaps": [5, 10, 15, 20], "min_th": 350},
    "video_52": {"gaps": [5, 10, 15, 20], "min_th": 300},
    "video_54": {"gaps": [5, 10, 15, 20], "min_th": 340},
    "video_55": {"gaps": [5, 10, 15, 20], "min_th": 219},
    "video_56": {"gaps": [5, 10, 15, 20], "min_th": 227},
    "video_57": {"gaps": [5, 10, 15, 20], "min_th": 173},
    "video_58": {"gaps": [5, 10, 15, 20], "min_th": 220},
    "video_60": {"gaps": [5, 10, 15, 20], "min_th": 280},
    "video_61": {"gaps": [5, 10, 15, 20], "min_th": 398},
    "video_62": {"gaps": [5, 10, 15, 20], "min_th": 540},
    "video_63": {"gaps": [5, 10, 15, 20], "min_th": 135},
    "video_64": {"gaps": [5, 10, 15, 20], "min_th": 104},
}