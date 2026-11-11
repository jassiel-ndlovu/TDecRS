import numpy as np 
import torch
from device import get_device
import math
import time

def RMSE(model, model_name, dataloader):
    RMSE = np.array([], dtype=np.float32)
    for batch in dataloader:
        conf = None
        if len(batch) == 4:
            features, feature_values, label, conf = batch
            conf = conf.to(get_device())
        else:
            features, feature_values, label = batch
        features = features.to(get_device())
        feature_values = feature_values.to(get_device())
        label = label.to(get_device())

        if conf is None:
            prediction = model(features, feature_values)
        else:
            prediction = model(features, feature_values, conf)
        prediction = prediction.clamp(min=-1.0, max=1.0)
        SE = (prediction - label).pow(2)
        RMSE = np.append(RMSE, SE.detach().cpu().numpy())

    return np.sqrt(RMSE.mean())
def pre_ranking(user_feature, item_feature, item_category, num_groups, train_dict, valid_dict, test_dict):
    '''prepare for the ranking: construct input data'''

    dev = get_device()
    item_idx = list(item_feature.keys())

    # Every user is scored against the same item table, so the item half of the
    # feature matrix is built once instead of being replicated per user.  The
    # original layout materialised a (num_users x num_items x num_features)
    # tensor on the device -- ~6.9 GiB for ml-1m -- which OOMs a 16 GiB card
    # once model and optimiser state are resident.  Model inputs are unchanged:
    # the per-user rows are re-assembled by expand+cat inside Ranking().
    all_item_features = torch.tensor(
        np.array([item_feature[itemID][0] for itemID in item_idx]), dtype=torch.long, device=dev)
    all_item_feature_values = torch.tensor(
        np.array([item_feature[itemID][1] for itemID in item_idx], dtype=np.float32), device=dev)

    item_pos = {itemID: pos for pos, itemID in enumerate(item_idx)}

    user_rank_feature = {}
    for userID in test_dict:
        his_items = train_dict[userID]
        mask = np.zeros(len(item_idx), dtype=np.float32)
        for itemID in his_items: # set as -inf if it's in training set
            if itemID in item_pos:
                mask[item_pos[itemID]] = -999.0

        user_feat = torch.tensor(
            np.array(user_feature[userID][0]), dtype=torch.long, device=dev)
        user_feat_values = torch.tensor(
            np.array(user_feature[userID][1], dtype=np.float32), device=dev)
        user_rank_feature[userID] = [user_feat, user_feat_values, torch.tensor(mask, device=dev)]

    user_kl_score = {}
    max_kl_score = 0
    min_kl_score = 999
    for userID in test_dict:
        train_set = train_dict[userID]
        valid_set = valid_dict[userID]
        kl_score = kl_score_calc(train_set, valid_set, item_category, num_groups)
        user_kl_score[userID]=kl_score
        if kl_score>max_kl_score:
            max_kl_score = kl_score
        if kl_score<min_kl_score:
            min_kl_score = kl_score
    user_kl_score['max_kl_score'] = max_kl_score
    user_kl_score['min_kl_score'] = min_kl_score

    return user_rank_feature, user_kl_score, all_item_features, all_item_feature_values


def Ranking(model, model_name, valid_dict, test_dict, train_dict, item_feature, user_rank_feature, user_kl_score,
            all_item_features, all_item_feature_values,
            FM_score, batch_size, topN, num_groups, alpha, user_confounder=None,
            return_pred=False):
    """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
    dev = get_device()
    user_gt_test = []
    user_gt_valid = []
    user_pred_dict = {}
    user_item_top1k = {}
    item_idx = list(item_feature.keys())
    num_items = all_item_features.size(0)
    item_idx_tensor = torch.tensor(item_idx, device=dev)

    for userID in test_dict:
        user_feat, user_feat_values, mask = user_rank_feature[userID]
        # DCR: the user's confounder after their whole training history
        conf_u = None if user_confounder is None else torch.tensor(
            np.asarray(user_confounder[userID], dtype=np.float32), device=dev)

        chunks = []
        st = 0
        while st < num_items:
            ed = min(st + batch_size, num_items)
            n = ed - st
            batch_feature = torch.cat(
                [user_feat.unsqueeze(0).expand(n, -1), all_item_features[st: ed]], 1)
            batch_feature_values = torch.cat(
                [user_feat_values.unsqueeze(0).expand(n, -1), all_item_feature_values[st: ed]], 1)
            if conf_u is None:
                prediction = model(batch_feature, batch_feature_values)
            else:
                prediction = model(batch_feature, batch_feature_values,
                                   conf_u.unsqueeze(0).expand(n, -1))
            chunks.append(prediction + mask[st: ed])
            st = ed
        all_predictions = chunks[0] if len(chunks) == 1 else torch.cat(chunks, 0)

        user_gt_valid.append(valid_dict[userID])
        user_gt_test.append(test_dict[userID])
        _, indices = torch.topk(all_predictions, topN[-1])
        pred_items = item_idx_tensor[indices].cpu().numpy().tolist()
        user_item_top1k[userID] = pred_items
        user_pred_dict[userID] = all_predictions.detach().cpu().numpy()

    if FM_score is None:
        # FM / NFM baselines: no inference blending, rank on the raw scores.
        user_pred = [user_item_top1k[userID] for userID in test_dict]
        valid_results = computeTopNAccuracy(user_gt_valid, user_pred, topN)
        test_results = computeTopNAccuracy(user_gt_test, user_pred, topN)
        if return_pred:
            return valid_results, test_results, None, user_pred_dict, user_item_top1k
        return valid_results, test_results, None

    kl_score_gap = user_kl_score['max_kl_score'] - user_kl_score['min_kl_score']

    user_pred = []
    for userID in test_dict:

        FM_item_scores = torch.tensor(FM_score[userID]).to(dev)
        all_predictions = torch.tensor(user_pred_dict[userID]).to(dev)

        lamda = ((user_kl_score[userID] - user_kl_score['min_kl_score'])/kl_score_gap)**alpha

        DecFM_score =  lamda * torch.sigmoid(all_predictions) + (1-lamda) * torch.sigmoid(FM_item_scores)

        _, indices = torch.topk(DecFM_score, topN[-1])
        pred_items = item_idx_tensor[indices].cpu().numpy().tolist()

        user_pred.append(pred_items)

    valid_results = computeTopNAccuracy(user_gt_valid, user_pred, topN)
    test_results = computeTopNAccuracy(user_gt_test, user_pred, topN)
#         calibration_results = calibration(item_feature, test_dict, train_dict, user_pred, topN, num_groups)
    calibration_results = None

    if return_pred: # used in the inference.py
        return valid_results, test_results, calibration_results, user_pred_dict, user_item_top1k
    return valid_results, test_results, calibration_results


def kl_div(p_dis, q_dis, alpha=0.01):
    KL_res = 0
    for index, p_value in enumerate(p_dis):
        if p_value < 1e-5:
            continue
        q_value = (1-alpha) * q_dis[index] + alpha * p_dis[index]
        KL_res += p_value*(np.log(p_value/q_value))
    return KL_res

def kl_score_calc(training_set, validation_set, item_category, num_groups):
    items_list = training_set+validation_set
    items_len = len(items_list)
    training_set_1 = items_list[:items_len//2]
    training_set_2 = items_list[items_len//2:]
    
    training_set_1_dis = [0] * num_groups
    for itemID in training_set_1:
        categories = item_category[itemID]
        for cate in categories:
            training_set_1_dis[cate] += round(1.0/len(categories), 4)
    training_set_1_dis = [x/len(training_set_1) for x in training_set_1_dis]

    training_set_2_dis = [0] * num_groups
    for itemID in training_set_2:
        categories = item_category[itemID]
        for cate in categories:
            training_set_2_dis[cate] += round(1.0/len(categories), 4)
    training_set_2_dis = [x/len(training_set_2) for x in training_set_2_dis]
    
    kl_res_1 = kl_div(training_set_1_dis, training_set_2_dis)
    kl_res_2 = kl_div(training_set_2_dis, training_set_1_dis)
    
    return kl_res_1+kl_res_2

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
    
    
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR


def calibration(item_feature, test_dict, train_dict, user_pred, topN, num_groups):
    C_KL = []
    C_H = []
    C_E = []
    assert len(test_dict) == len(user_pred)
    
    alpha = 0.01 # refer to the setting in "calibrated recommendation"
    for i, userID in enumerate(test_dict):
        
        history_items = train_dict[userID]
        history_cate = np.array([0]*num_groups, dtype=np.float32)
        for itemID in history_items:
            history_cate += np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32)
        history_cate = history_cate/len(history_items)
        
        C_KL_u = []
        C_H_u = []
        C_E_u = []
        for n in topN:
            rec_list = user_pred[i][:n]
            rec_cate = np.array([0]*num_groups, dtype=np.float32)
            for itemID in rec_list:
                rec_cate += np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32)
            rec_cate = rec_cate/len(rec_list)
            C_KL_res = 0
            C_H_res = 0
            C_E_res = 0
            for j, p_cate in enumerate(history_cate):
                C_H_res += (np.sqrt(p_cate) - np.sqrt(rec_cate[j])) * (np.sqrt(p_cate) - np.sqrt(rec_cate[j]))
                C_E_res += (p_cate-rec_cate[j]) * (p_cate-rec_cate[j])
                if p_cate < 1e-5:
                    continue
                q_cate = (1-alpha) * rec_cate[j] + alpha * p_cate
                C_KL_res += p_cate*(np.log(p_cate/q_cate))
            C_KL_u.append(C_KL_res)
            C_H_u.append(np.sqrt(C_H_res)/np.sqrt(2))
            C_E_u.append(np.sqrt(C_E_res))
        C_KL.append(C_KL_u)
        C_H.append(C_H_u)
        C_E.append(C_E_u)
    C_KL = np.around(np.mean(C_KL, 0), 4).tolist()
    C_H = np.around(np.mean(C_H, 0), 4).tolist()
    C_E = np.around(np.mean(C_E, 0), 4).tolist()
    return C_KL, C_H, C_E



def print_results(train_RMSE, valid_result, test_result, calibration_results):
    """output the evaluation results."""
    if train_RMSE is not None:
        print("[Train]: RMSE: {:.4f}".format(train_RMSE))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

    if calibration_results is not None:
        print("[Calibration]: C_KL: {} C_H: {} C_E: {}".format(
                            '-'.join([str(x) for x in calibration_results[0]]), 
                            '-'.join([str(x) for x in calibration_results[1]]), 
                            '-'.join([str(x) for x in calibration_results[2]])))
