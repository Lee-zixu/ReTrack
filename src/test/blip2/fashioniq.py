import datetime
import time
from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabulate import tabulate
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning

from src.tools.files import json_dump, json_load, json_dump_append
from tqdm import tqdm
import logging

class TestFashionIQ_Ori:
    def __init__(self, category: str):
        self.category = category
        pass

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for validation Original Test...")
        start_time = time.time()
        test_queries, test_targets = data_loader.dataset.test_queries, data_loader.dataset.test_targets

        vl_feats = []
        pair_ids = []

        tar_img_feats = []
        tar_video_embeds = []
        tar_video_token_embeds = []
        query_feats = []
        compose_feature_video_sentence_list = []
        compose_feature_token_list = []
        prob_compose_embedding = []
        prob_target_embedding = []
        captions = []
        device = "cuda"
        with torch.no_grad():   
            all_queries = []
            all_imgs = []
            if test_queries:
                # compute test query features
                imgs = []
                mods = []
                for t in tqdm(test_queries):
                    imgs += [t['source_img_data']]
                    mods += [t['mod']['str']]
                    if len(imgs) >= data_loader.batch_size or t is test_queries[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        _, _, query_feat= model.compose_feature(imgs, mods, [""], fabric, device)
                        query_feats.append(query_feat.cpu())
                        imgs = []
                        mods = []

                # compute all image features
                imgs = []
                for t in tqdm(test_targets):
                    imgs += [t['target_img_data']]
                    if len(imgs) >= data_loader.batch_size or t is test_targets[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        tar_feat = model.target_fea(imgs, [""], fabric, device)[1]
                        tar_img_feats.append(tar_feat.cpu())
                        imgs = []

                
                query_feats = torch.cat(query_feats, dim=0)
                tar_img_feats = torch.cat(tar_img_feats, dim=0)

                query_feats = F.normalize(query_feats, dim=-1)
                tar_img_feats = F.normalize(tar_img_feats, dim=-1)

                sims = model.fine_grained_sim(query_feats, tar_img_feats)
                test_targets_id = []
                for i in test_targets:
                    test_targets_id.append(i['target_img_id'])
                for i, t in enumerate(test_queries):
                    sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


                nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

                # compute recalls
                out = []
                nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
                for k in [1, 10, 50]:
                    r = 0.0
                    for i, nns in enumerate(nn_result):
                        if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                            r += 1
                    r = 100 * r / len(nn_result)
                    out += [('{}_r{}'.format(data_loader.dataset.category, k), r)]
                logging.info(out)
                fabric.barrier()




def recall_at_k_labels(sim, query_lbls, target_lbls, k=10):
    distances = - sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_lbls)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(query_lbls), len(target_lbls)).reshape(
            len(query_lbls), -1
        )
    )
    assert torch.equal(
        torch.sum(labels, dim=-1).int(), torch.ones(len(query_lbls)).int()
    )
    return round((torch.sum(labels[:, :k]) / len(labels)).item() * 100, 2)


def get_recalls_labels(
    sims, query_lbls, target_lbls, ks: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    return {f"R{k}": recall_at_k_labels(sims, query_lbls, target_lbls, k) for k in ks}


def mean_results(dir=".", fabric=None, save=True):
    dir = Path(dir)
    recall_pths = list(dir.glob("recalls_fiq-*.json"))
    recall_pths.sort()
    if len(recall_pths) != 3:
        return

    df = {}
    for pth in recall_pths:
        name = pth.name.split("_")[1].split(".")[0]
        data = json_load(pth)
        df[name] = data

    df = pd.DataFrame(df)

    # FASHION-IQ
    df_fiq = df[df.columns[df.columns.str.contains("fiq")]]
    assert len(df_fiq.columns) == 3
    df_fiq["Average"] = df_fiq.mean(axis=1)
    df_fiq["Average"] = df_fiq["Average"].apply(lambda x: round(x, 2))

    headers = [
        "dress\nR10",
        "dress\nR50",
        "shirt\nR10",
        "shirt\nR50",
        "toptee\nR10",
        "toptee\nR50",
        "Average\nR10",
        "Average\nR50",
    ]
    fiq = []
    for category in ["fiq-dress", "fiq-shirt", "fiq-toptee", "Average"]:
        for recall in ["R10", "R50"]:
            value = df_fiq.loc[recall, category]
            value = str(value).zfill(2)
            fiq.extend([value])
    if fabric is None:
        print(tabulate([fiq], headers=headers, tablefmt="latex_raw"))
        print(" & ".join(fiq))
    else:
        fabric.print(tabulate([fiq], headers=headers))
        fabric.print(" & ".join(fiq))

    if save:
        df_mean = df_fiq["Average"].to_dict()
        df_mean = {k + "_mean": round(v, 2) for k, v in df_mean.items()}
        json_dump(df_mean, "recalls_fiq-mean.json")
