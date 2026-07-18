import random
import numpy as np
import re


def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg)
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")

def camel_to_normal(camel_string):
    # 使用正则表达式将驼峰字符串转换为正常字符串
    normal_string = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_string).lower()
    return normal_string

def clean_symbol_in_rel(rel):
    '''
    clean symbol in relation

    Args:
        rel (str): relation name
    '''
    
    rel = rel.strip("_") # Remove heading
    # Replace inv_ with inverse
    # rel = rel.replace("inv_", "inverse ")
    if "/" in rel:
        if "inverse" in rel:
            rel = rel.replace("inverse ", "")
            rel = "inverse " + fb15k_rel_map[rel]
        else:
            rel = fb15k_rel_map[rel]
    # WN-18RR
    elif "_" in rel:
        rel = rel.replace("_", " ") # Replace _ with space
    # UMLS
    elif "&" in rel:
        rel = rel.replace("&", " ") # Replace & with space
    # YAGO 
    else:
        rel = camel_to_normal(rel)
    return rel

def check_prompt_length(prompt, list_of_paths, model):
    '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
    all_paths = "\n".join(list_of_paths)
    all_tokens = prompt + all_paths
    maximun_token = max(1, model.maximun_token - model.args.max_new_tokens)
    if model.token_len(all_tokens) < maximun_token:
        return all_paths
    else:
        # Shuffle the paths
        random.shuffle(list_of_paths)
        new_list_of_paths = []
        # check the length of the prompt
        for p in list_of_paths:
            tmp_all_paths = "\n".join(new_list_of_paths + [p])
            tmp_all_tokens = prompt + tmp_all_paths
            if model.token_len(tmp_all_tokens) > maximun_token:
                return "\n".join(new_list_of_paths)
            new_list_of_paths.append(p)


def ill_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] > pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank

def harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank

def balance_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    if pred[ent2idx[q_t]]!=0:
        pred_ranks = np.argsort(pred)[::-1]    

        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t!=ent2idx[q_t]]

        filtered_ranks = []
        for i in range(len(pred_ranks)):
            idx = pred_ranks[i]
            if idx not in truth:
                filtered_ranks.append(idx)

        rank = filtered_ranks.index(ent2idx[q_t])+1
    else:
        truth = gt[(q_h, q_r)]

        filtered_pred = []

        for i in range(len(pred)):
            if i not in truth:
                filtered_pred.append(pred[i])
        n_non_zero = np.count_nonzero(filtered_pred)
        rank = n_non_zero+1
    return rank

def random_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    truth.append(ent2idx[q_t])
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            if (pred[idx] == pred[ent2idx[q_t]]) and (np.random.uniform() < 0.5):
                filtered_ranks.append(idx)
            else:
                filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank
