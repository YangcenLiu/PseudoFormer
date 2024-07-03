import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import scienceplots
import matplotlib.font_manager as fm
fm.fontManager.addfont('/data0/lixunsong/liuyangcen/CVPR2024/timesi.ttf')
fm.fontManager.addfont('/data0/lixunsong/liuyangcen/CVPR2024/times.ttf')
fm.fontManager.addfont('/data0/lixunsong/liuyangcen/CVPR2024/timesbd.ttf')

def main(path, name):
    plt.rcParams["font.family"] = "Times New Roman"
    with open(path, "r",) as f:
        content = json.load(f)
    
    vid_lens = []
    seg_lens = []
    for vid in content["database"].values():
        vid_lens.append(vid["duration"])
        for seg in vid["annotations"]:
            seg_lens.append(seg["segment"][1]-seg["segment"][0])
    
    print(len(vid_lens))
    print(len(seg_lens))

    # Sort segment lengths
    seg_lens.sort()

    max_length = max(seg_lens)
    
    # Plot histogram with consistent bin size for all three segments
    # plt.hist(seg_lens, bins=np.linspace(0, max_length, num=500), color='green', alpha=0.5, label='All Segments')
    
    # Calculate segment lengths for each third
    total_length = len(seg_lens)
    one_third = total_length // 3
    
    # Divide segments into three categories
    long_seg = seg_lens[2 * one_third:]
    middle_seg = seg_lens[one_third:2 * one_third]
    short_seg = seg_lens[:one_third]
    print(long_seg[-1])
    print(long_seg[0])
    print(short_seg[0])
    print(short_seg[-1])
    exit(0)
    # Plot histograms for each segment category
    plt.hist(long_seg, bins=np.linspace(0, max_length, num=500), color='red', alpha=0.5, label='Long ({:.2f})'.format(sum(long_seg)/len(long_seg)))
    plt.hist(middle_seg, bins=np.linspace(0, max_length, num=500), color='orange', alpha=0.5, label='Middle ({:.2f})'.format(sum(middle_seg)/len(middle_seg)))
    plt.hist(short_seg, bins=np.linspace(0, max_length, num=500), color='green', alpha=0.5, label='Short ({:.2f})'.format(sum(short_seg)/len(short_seg)))
    
    # Add labels and title
    plt.xlabel('Duration')
    plt.ylabel('Count')
    plt.title('Histogram of Segment Durations of ' + name)
    
    # Add legend
    plt.legend()

    plt.savefig("/data0/lixunsong/liuyangcen/TriDet/outputs/" + name + ".png")
    plt.close()

def main1(paths, name):
    plt.rcParams["font.family"] = "Times New Roman"
    content = []
    
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                content.append(json.loads(line))
    
    vid_lens = []
    seg_lens = []
    for vid in content:
        vid_lens.append(vid["duration"])
        for seg in vid["relevant_windows"]:
            seg_lens.append(seg[1]-seg[0])
    
    print(len(vid_lens))
    print(len(seg_lens))

    # Sort segment lengths
    seg_lens.sort()

    max_length = max(seg_lens)
    
    # Plot histogram with consistent bin size for all three segments
    # plt.hist(seg_lens, bins=np.linspace(0, max_length, num=500), color='green', alpha=0.5, label='All Segments')
    
    # Calculate segment lengths for each third
    total_length = len(seg_lens)
    one_third = total_length // 3
    
    # Divide segments into three categories
    long_seg = seg_lens[2 * one_third:]
    middle_seg = seg_lens[one_third:2 * one_third]
    short_seg = seg_lens[:one_third]
    
    # Plot histograms for each segment category
    plt.hist(long_seg, bins=np.linspace(0, max_length, num=200), color='red', alpha=0.5, label='Long ({:.2f})'.format(sum(long_seg)/len(long_seg)))
    plt.hist(middle_seg, bins=np.linspace(0, max_length, num=200), color='orange', alpha=0.5, label='Middle ({:.2f})'.format(sum(middle_seg)/len(middle_seg)))
    plt.hist(short_seg, bins=np.linspace(0, max_length, num=200), color='green', alpha=0.5, label='Short ({:.2f})'.format(sum(short_seg)/len(short_seg)))
    
    # Add labels and title
    plt.xlabel('Duration')
    plt.ylabel('Count')
    plt.title('Histogram of Segment Durations of ' + name)
    
    # Add legend
    plt.legend()

    plt.savefig("/data0/lixunsong/liuyangcen/TriDet/outputs/" + name + ".png")
    plt.close()

if __name__ == '__main__':
    main("/data0/lixunsong/Datasets/thumos/annotations/thumos14.json", "thumos14")
    main("/data0/lixunsong/Datasets/anet_1.3/annotations/anet1.3_i3d_filtered.json", "anet13")
    main("/data0/lixunsong/Datasets/HACS/hacs.json", "hacs")


    main1(["/data0/lixunsong/Datasets/moment_anno/highlight_train_release.jsonl",
    "/data0/lixunsong/Datasets/moment_anno/highlight_val_release.jsonl"],
     "highlight")

    main1(["/data0/lixunsong/Datasets/moment_anno/charades_sta/charades_sta_test_tvr_format.jsonl",
    "/data0/lixunsong/Datasets/moment_anno/charades_sta/charades_sta_train_tvr_format.jsonl"],
     "charades")

    main1(["/data0/lixunsong/Datasets/moment_anno/tacos/test.jsonl",
    "/data0/lixunsong/Datasets/moment_anno/tacos/train.jsonl",
    "/data0/lixunsong/Datasets/moment_anno/tacos/val.jsonl"],
     "tacos") 