# python imports
import argparse
import os
import glob
import time
from pprint import pprint
import json

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import sub_valid_one_epoch, ANETdetection, fix_random_seed

def snippet_mAP(results, gt):

    Acc = []
    Attn = []

    print(results.keys()) # dict_keys(['video-id', 't-start', 't-end', 'label', 'score'])

    print(len(results['video-id']))
    print(len(results['label']))
    # gt = gt.groupby('video-id')
    # results = results.groupby('video-id')

    # print(results[0].shape)

    class_mapping = {29:3,65:10,70:11,79:12,68:13,127:14,153:15,41:18} # a2t 测试thumos2anet
    # class_mapping = {3:29,10:65,11:70,12:79,13:68,14:127,15:153,18:41} # t2a 测试anet2thumos

    for vname in results['video_id']:
        cas = results["labels"]

        attn = results[vname]["attn"]
        proposals = gt[gt["video-id"] == vname]

        gt_p = torch.zeros_like(cas) + 100

        for index, row in proposals.iterrows():
            start = row["t-start"]
            end = row["t-end"]
            label = row["label"]
            if start >= cas.size(1):
                continue

            if end >= cas.size(1):
                end = cas.size(1)-1
            
            for t in range(start, end+1):
                gt_p[:,t,0] = class_mapping[label]
                # print(label)
        
        for t in range(cas.size(1)):
            if gt_p[:,t,0] == 20: # 是否考虑后景
                continue
            if cas[:,t,0] == gt_p[:,t,0]:
                Acc.append(1)
            else:
                Acc.append(0)
            Attn.append(attn[:,t,0])
    print(sum(Attn)/len(Attn))
    exit()
    print(sum(Acc)/len(Acc)*100)

    # Define the number of bins or intervals
    num_bins = 10

    # Initialize lists to store accuracy for each interval
    accuracy_by_bin = []

    # Initialize the bin edges
    bin_edges = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    print(bin_edges)

    # Calculate accuracy in each bin
    for i in range(num_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        in_bin = [Acc[j] for j in range(len(Attn)) if lower_bound <= Attn[j] < upper_bound]
        accuracy = sum(in_bin) / len(in_bin) * 100 if len(in_bin) > 0 else 0
        accuracy_by_bin.append(accuracy)
    print(accuracy_by_bin)

    exit()

def main(args):
    if args.mode=="t2a":
        main_t2a(args)
    elif args.mode=="a2t":
        main_a2t(args)

def main_t2a(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    
    if os.path.isfile(args.crd_config):
        crd_cfg = load_config(args.crd_config)
    else:
        raise ValueError("Config file does not exist.")

    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
        crd_cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    crd_val_dataset = make_dataset(
        crd_cfg['dataset_name'], False, crd_cfg['val_split'], **crd_cfg['dataset']
    )

    # set bs = 1, and disable shuffle
    crd_val_loader = make_data_loader(
        crd_val_dataset, False, None, 1, crd_cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # SmD test
    """5. load class mapping"""
    with open(args.class_mapping_path, "r") as file:
        class_mapping = json.load(file)
        t_index2name = {
            int(k): v["anet name"] for k, v in class_mapping.items()
        }
        a_index2name = {
                int(v["anet idx"]) : v["thu name"] for k, v in class_mapping.items()
            }
        a_index2t_index = {
                int(v["anet idx"]) : int(k) for k, v in class_mapping.items()
            }
        t_index2a_index = {
                int(k) : int(v["anet idx"]) for k, v in class_mapping.items()
            }

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds'],
            target_class_list=t_index2name.keys(),
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')
    """6. Test the model (SmD test)"""
    print("\nSmD test: model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP, results = sub_valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        target_class_list=t_index2name.keys(),
        num_classes=20,
        return_results=True,
    )
    end = time.time()
    # print("All done! Total time: {:0.2f} sec".format(end - start))

    """7. Test the model (CrD test)"""
    # CrD test
    # set up evaluator
    crd_det_eval, output_file = None, None
    if not args.saveonly:
        crd_val_db_vars = crd_val_dataset.get_attributes()
        crd_det_eval = ANETdetection(
            crd_val_dataset.json_file,
            crd_val_dataset.split[0],
            tiou_thresholds=crd_val_db_vars['tiou_thresholds'],
            target_class_list=a_index2name.keys(),
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    print("\nCrD test: model {:s} ...".format(cfg['model_name']))

    start = time.time()
    mAP, results = sub_valid_one_epoch(
        crd_val_loader,
        model,
        -1,
        evaluator=crd_det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        target_class_list=a_index2name.keys(),
        t_index2a_index=t_index2a_index,
        num_classes=20,
        return_results=True,
    )

    return

def main_a2t(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    
    if os.path.isfile(args.crd_config):
        crd_cfg = load_config(args.crd_config)
    else:
        raise ValueError("Config file does not exist.")

    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
        crd_cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    crd_val_dataset = make_dataset(
        crd_cfg['dataset_name'], False, crd_cfg['val_split'], **crd_cfg['dataset']
    )

    # set bs = 1, and disable shuffle
    crd_val_loader = make_data_loader(
        crd_val_dataset, False, None, 1, crd_cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # SmD test

    """5. load class mapping"""
    with open(args.class_mapping_path, "r") as file:
        class_mapping = json.load(file)
        t_index2name = {
            int(k): v["thu name"] for k, v in class_mapping.items()
        }
        a_index2name = {
                int(v["thu idx"]) : v["anet name"] for k, v in class_mapping.items()
            }
        a_index2t_index = {
                int(v["thu idx"]) : int(k) for k, v in class_mapping.items()
            }
        t_index2a_index = {
                int(k) : int(v["thu idx"]) for k, v in class_mapping.items()
            }

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds'],
            target_class_list=t_index2name.keys(),
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """6. Test the model (SmD test)"""
    print("\nSmD test: model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = sub_valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        target_class_list=t_index2name.keys(),
        num_classes=200,
    )
    end = time.time()
    # print("All done! Total time: {:0.2f} sec".format(end - start))

    """7. Test the model (CrD test)"""
    # SmD test
    # set up evaluator
    crd_det_eval, output_file = None, None
    if not args.saveonly:
        crd_val_db_vars = crd_val_dataset.get_attributes()
        crd_det_eval = ANETdetection(
            crd_val_dataset.json_file,
            crd_val_dataset.split[0],
            tiou_thresholds=crd_val_db_vars['tiou_thresholds'],
            target_class_list=a_index2name.keys(),
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    print("\nCrD test: model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = sub_valid_one_epoch(
        crd_val_loader,
        model,
        -1,
        evaluator=crd_det_eval,
        output_file=output_file,
        ext_score_file=crd_cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        target_class_list=a_index2name.keys(),
        t_index2a_index=t_index2a_index,
        num_classes=200,
    )


    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('mode', type=str,
                        help='t2a, a2t, t2h, a2h')
    parser.add_argument('class_mapping_path', type=str,
                        help='path to class mapping file')
    parser.add_argument('crd_config', type=str,
                        help='path to crd config file')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
