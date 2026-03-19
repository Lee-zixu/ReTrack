import datetime
import time
from pathlib import Path


from scipy.optimize import minimize

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump, json_dump_append
from tqdm import tqdm


class TestWebVidCoVR:
    def __init__(self, remove_self_similarity: bool = True, dataset: str = "covr",
                 beta_range: tuple = (0.1, 5), beta_step: float = 0.5, max_iter: int = 20):
        
        self.remove_self_similarity = remove_self_similarity
        self.dataset = dataset
        self.beta_start, self.beta_end = beta_range
        self.beta_step = beta_step
        self.max_iter = max_iter

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation mmembed_withTarget...")
        start_time = time.time()

        tar_img_feats = []
        query_feats = []
        captions = []
        pair_ids = []
        mod_feats = []
        ref_feats = []

        for batch in tqdm(data_loader):
            ref_img = batch["ref_img"]
            tar_img = batch["tar_img"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            ref_description = batch["ref_description"]
            tag_description = batch["tag_description"]
            pair_ids.extend(pair_id.cpu().numpy().tolist())
            captions.extend(caption)

            device = ref_img.device

            
            mod_feat = model.textual_feature(caption,device)
            mod_feats.append(mod_feat.cpu())
            ref_feat = model.target_fea(ref_img, tag_description, fabric, device)
            ref_feats.append(ref_feat[1].cpu())
            query_feat, cu, query_embeds = model.compose_feature(ref_img, caption, ref_description, fabric, device)
            query_feats.append(query_embeds.cpu())

            
            result = model.target_fea(tar_img, tag_description, fabric, device)
            tar_feat = result[1] 
            
            tar_img_feats.append(tar_feat.cpu())


        # 拼接所有结果
        query_feats = torch.cat(query_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)
        mod_feats = torch.cat(mod_feats, dim=0)
        ref_feats = torch.cat(ref_feats, dim=0)

        # 归一化特征
        query_feats = F.normalize(query_feats, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)
        mod_feats = F.normalize(mod_feats, dim=-1)
        ref_feats = F.normalize(ref_feats, dim=-1)

        # 准备图像ID
        ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]
        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)



        if fabric.global_rank == 0:
            sim_q2t_np = model.fine_grained_sim(query_feats, tar_img_feats)

            sim_q2t_np = sim_q2t_np

            # 计算评估指标
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            fabric.print(f"Evaluation time: {total_time_str}")

            recalls = eval_recall(sim_q2t_np)
            recalls["annotation"] = Path(data_loader.dataset.annotation_pth).name
            fabric.print(recalls)
            
            self_sim = "" if self.remove_self_similarity else "_ss"
            json_dump_append(recalls, f"recalls_{self.dataset}{self_sim}.json")

            fabric.print(
                f"Recalls saved in {Path.cwd()}/recalls_{self.dataset}{self_sim}.json"
            )

        fabric.barrier()




@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result
