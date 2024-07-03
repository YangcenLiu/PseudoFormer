import numpy as np
import torch
import torch.nn as nn
# from scipy.interpolate import interp1d


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def str2ind(categoryname, classlist):
    return [
        i for i in range(len(classlist))
        if categoryname == classlist[i].decode("utf-8")
    ][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)],
                  axis=0)


def idx2multihot(id_list, num_class):
    return np.sum(np.eye(num_class)[id_list], axis=0)


def random_extract(feat, t_max):
    # ind = np.arange(feat.shape[0])
    # splits = np.array_split(ind, t_max)
    # nind = np.array([np.random.choice(split, 1)[0] for split in splits])
    # return feat[nind]

    # ind = np.random.choice(feat.shape[0], size=t_max)
    # ind = sorted(ind)
    # return feat[ind]
    r = np.random.randint(len(feat) - t_max)
    return feat[r: r + t_max]


def pad(feat, min_len):
    if feat.shape[0] <= min_len:
        return np.pad(
            feat,
            ((0, min_len - feat.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        return feat


def fn_normalize(x):
    return (x - np.mean(x, 0, keepdims=True)) / \
           (np.std(x, 0, keepdims=True) + 1e-10)


def process_feat(feat, length=None, normalize=False):
    if length is not None:
        if len(feat) > length:
            x = random_extract(feat, length)
        else:
            x = pad(feat, length)
    else:
        x = feat
    if normalize:
        x = fn_normalize(x)
    return x


def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + "-results.log", "a+")
    string_to_write = str(itr)
    # if dmap:
    for item in dmap:
        string_to_write += " " + "%.2f" % item
    string_to_write += " " + "%.2f" % cmap
    fid.write(string_to_write + "\n")
    fid.close()


def soft_nms(dets, iou_thr=0.7, method='gaussian', sigma=0.3):
    dets = np.array(dets)
    x1 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, 1]

    areas = x2 - x1 + 1

    # expand dets with areas, and the second dimension is
    # x1, x2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 1], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1].tolist())

        xx1 = np.maximum(dets[0, 2], dets[1:, 2])
        xx2 = np.minimum(dets[0, 3], dets[1:, 3])

        inter = np.maximum(xx2 - xx1 + 1, 0.0)
        iou = inter / (dets[0, -1] + dets[1:, -1] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 1] *= weight
        dets = dets[1:, :]

    return retained_box


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)  # 交集

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]  # 取出不重叠的
        order = order[inds + 1]

    return keep

def get_proposal_oic_2(tList,
                       wtcam,
                       final_score,
                       c_pred,
                       scale,
                       v_len,
                       num_segments,
                       lambda_=0.25,
                       gamma=0.2,
                       loss_type="oic"): # 一般用的这个2号
    # t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list) # grouping: np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(
                    0, int(grouped_temp_list[j][0] - lambda_ * len_proposal))
                outer_e = min(
                    int(wtcam.shape[0] - 1),
                    int(grouped_temp_list[j][-1] + lambda_ * len_proposal),
                )

                outer_temp_list = list(
                    range(outer_s, int(grouped_temp_list[j][0]))) + list(
                    range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                if loss_type == "oic":
                    c_score = inner_score - outer_score + gamma * final_score[
                        c_pred[i]]
                else:
                    c_score = inner_score
                t_start = grouped_temp_list[j][0]
                t_end = (grouped_temp_list[j][-1] + 1)
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


"""
ramp up
"""


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * linear_rampup(epoch, args.consistency_rampup)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def get_cls_score(element_cls, dim=-2, rat=20, ind=None):
    topk_val, _ = torch.topk(element_cls,
                             k=max(1, int(element_cls.shape[-2] // rat)),
                             dim=-2)
    instance_logits = torch.mean(topk_val, dim=-2)
    pred_vid_score = torch.softmax(
        instance_logits, dim=-1)[..., :-1].squeeze().data.cpu().numpy()
    return pred_vid_score

def filter_segments(segment_predict, vn):
    ambilist = './Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        # s[j], e[j], np.max(seg)+0.7*c_s[c],c]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)

@torch.no_grad()
def multiple_threshold_hamnet(vid_name, data_dict, fps):
    elem = data_dict['cas']
    element_atn = data_dict['attn']
    act_thresh_cas = np.arange(0.1, 0.9, 10)
    element_logits = elem * element_atn
    pred_vid_score = get_cls_score(element_logits, rat=10)
    score_np = pred_vid_score.copy()
    # score_np[score_np < 0.2] = 0
    # score_np[score_np >= 0.2] = 1
    cas_supp = element_logits[..., :-1]
    cas_supp_atn = element_atn

    pred = np.where(pred_vid_score >= 0.2)[0]

    # NOTE: threshold
    act_thresh = np.linspace(0.1, 0.9, 10)

    prediction = None
    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])
    cas_pred = cas_supp[0].cpu().numpy()[:, pred]
    num_segments = cas_pred.shape[0]
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]

    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))

    proposal_dict = {}

    for i in range(len(act_thresh)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
            seg_list.append(pos)

        proposals = get_proposal_oic_2(seg_list,
                                             cas_temp,
                                             pred_vid_score,
                                             pred,
                                             None,
                                             num_segments,
                                             num_segments,)

        for j in range(len(proposals)):
            try:
                class_id = proposals[j][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += proposals[j]
            except IndexError:
                logger.error(f"Index error")
    final_proposals = []

    for class_id in proposal_dict.keys():
        final_proposals.append(soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))

    video_lst, t_start_lst, t_end_lst = [], [], []
    cls_idxs_all, scores_all = [], []
    # [c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end, c_score, c_pred])

    segment_predict = np.array(segment_predict)
    # segment_predict = filter_segments(segment_predict, vid_name)

    segs_all = []
    cls_idxs_all, scores_all = [], []
    for i in range(np.shape(segment_predict)[0]):
        segs_all.append(torch.tensor([[segment_predict[i, 0], segment_predict[i, 1]]]))
        scores_all.append(torch.tensor([segment_predict[i, 2]]))
        cls_idxs_all.append(torch.tensor([segment_predict[i, 3]], dtype=int))

    if len(segs_all) > 0:
        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {
            "segments": segs_all,
            "labels": cls_idxs_all,
            "scores": scores_all,
        }
    else:
        results = None

    return results