"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import copy

from src.model.blip2.blip2 import Blip2Base, disabled_train


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Contribution_Layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,  # 512, 1, 512
                 activation="relu", normalize_before=False, is_weights=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)  
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)  
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before
        self.is_weights = is_weights

    def forward(self, tgt, memory,
                pos=None,
                query_pos=None):
        tgt = self.norm1(tgt) 
        memory = self.norm2(memory)
        tgt2, atten_weights = self.multihead_attn(tgt, memory, memory,)  
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm4(tgt)

        return tgt, atten_weights

class TransDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                pos=None,
                query_pos=None):
        output = tgt

        intermediate = []
        all_weights = []

        for layer in self.layers:
            output, weights = layer(output, memory,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                all_weights.append(weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(all_weights)
        return output.unsqueeze(0)

class Contribution_decoder(nn.Module):
    def __init__(self, layers=1, heads=1, dim_ftr=512, dim_feedforward=512):
        super().__init__()
        embedding_dim = dim_ftr  # 512
        d_model = dim_ftr
        dim_feedforward = dim_feedforward
        decoder_layer = Contribution_Layer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward)
        self.event_decoder = TransDecoder(decoder_layer, layers, nn.LayerNorm(d_model),
                                          return_intermediate=True)

    def forward(self, query, features):
        query_size = features.shape[1] 
        enco_others = features.permute(1, 0, 2) 
        h_attr = query 

        if len(h_attr.size()) == 2: 
            h_attr = h_attr.unsqueeze(1).repeat(1, query_size, 1)  
            h_attr_batch = h_attr.permute(1,0,2)  
        else:
            h_attr_batch = h_attr.permute(1,0,2) 

        hs, _ = self.event_decoder(h_attr_batch, enco_others)  
        hs = hs[-1].permute(1, 0, 2) 

        return hs


class EvidenceRegularizationLoss(nn.Module):
    def __init__(self, tau):
        super(EvidenceRegularizationLoss, self).__init__()
        self.tau = tau
        self.mse = nn.MSELoss(reduction='mean')

    
    def forward(self, sims, evidence, lambda_=0.01):
        BS = sims.size(0)
        K = evidence.size(1)
        mask = 1 - torch.eye(BS).cuda()
        soft_label = (sims * mask)
        S = torch.sum(evidence, dim=1)
        U = K / S

        scale = (1 - U).mean() / soft_label.mean()
        # ce loss
        loss = self.mse(1 - U, scale * soft_label)
        return loss





class ReTrack(Blip2Base):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="eva_clip_g",
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        temperature=1,
        si_ti_weight=1,
        si_tc_weight=0,

        using_M=64, # Follow Batch
        K=32,  
        lambda_=1, 
        vid_frames=4,
        kappa_=0.5, 
        tau=5,  
    ):
        super().__init__()
        print("Initialized ReTrack")
        self.loss = loss

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = 0.07 

        self.max_txt_len = max_txt_len
        self.num_frames = vid_frames
        self.compose_token_mat_weight = nn.parameter.Parameter(torch.eye(num_query_token * self.num_frames), requires_grad=True)

        self.compose_anchor_mat_weight = nn.parameter.Parameter(torch.eye(num_query_token * self.num_frames), requires_grad=True)


        self.ifFrames = False 

        self.K = K * self.num_frames
        self.tau = tau
        
        self.loss_evi = EvidenceRegularizationLoss(self.tau)  
        self.lambda_ = lambda_
        self.kappa_ = kappa_


        self.contribution_decoder = Contribution_decoder(layers=2, heads=1, dim_ftr=embed_dim, dim_feedforward=embed_dim)
        self.anchor_projector = nn.Linear(256, self.K * 256)

        self.point_factor = nn.Parameter(torch.rand(1), requires_grad=True)
        dim_size = 256


        self.text_anchor_to_k = nn.Sequential(
            nn.Linear(
                dim_size,
                int(dim_size / 2)
            ),
            nn.LeakyReLU(),
            nn.Linear(int(dim_size / 2), self.K),
            nn.Softmax(dim=-1)
        )



        self.point_weights = nn.Sequential(
            nn.Linear(using_M, using_M * 2),
            nn.LeakyReLU(),
            nn.Linear(using_M*2, embed_dim)
        )


        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight


    def target_fea(self, tar_img, description, fabric, device='cuda'):
        if tar_img.dim() == 5:
            self.ifFrames = True

        bs = tar_img.shape[0]
        if self.ifFrames:
            bs, nf, c, h, w = tar_img.shape
            tar_img = tar_img.view(bs * nf, c, h, w)
        

        if self.train_vit:
            tar_img_embs = self.ln_vision(self.visual_encoder(tar_img))
        else:
            with torch.no_grad():
                tar_img_embedding = self.visual_encoder(tar_img)
            tar_img_embs = self.ln_vision(tar_img_embedding)
        tar_img_embs = tar_img_embs.float()

        query_tokens = self.query_tokens.expand(
            bs, -1, -1
        )

        if self.ifFrames:
            tar_img_embs = tar_img_embs.view(bs, -1, tar_img_embs.shape[-1])
            query_tokens = query_tokens.repeat(1, nf, 1)


        tar_img_atts = torch.ones(tar_img_embs.size()[:-1], dtype=torch.long).to(device)
        tar_img_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=tar_img_embs,
            encoder_attention_mask=tar_img_atts,
            return_dict=True,
        )
        vl_embs = tar_img_output.last_hidden_state[:, : query_tokens.size(1), :]


        target_si_feat = F.normalize(self.vision_proj(vl_embs), dim=-1)

        target_si_feat_mean = F.normalize(target_si_feat.mean(dim=1)) 

        return target_si_feat_mean, target_si_feat, None



    def compose_feature(self, ref_img, caption, description, fabric, device='cuda'):
        if ref_img.dim() == 5:
            self.ifFrames = True

        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        if self.ifFrames:
            bs, nf, c, h, w = ref_img.shape
            ref_img = ref_img.view(bs * nf, c, h, w)
        else:
            bs = ref_img.shape[0]

        if self.train_vit:
            ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
        else:
            with torch.no_grad():
                ref_img_embedding = self.visual_encoder(ref_img)
            ref_img_embs = self.ln_vision(ref_img_embedding)

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.ifFrames:
            ref_img_embs = ref_img_embs.view(bs, -1, ref_img_embs.shape[-1])
            query_tokens = query_tokens.repeat(1, nf, 1)
        
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        ###============== Image-text Matching ===================###

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        output = self.Qformer.bert(
            text_tokens.input_ids,  
            query_embeds=query_tokens,  
            attention_mask=attention_mask,  
            encoder_hidden_states=ref_img_embs,  
            encoder_attention_mask=ref_img_atts,  
            return_dict=True,
            output_attentions=True
        )

        vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]


        
        query_embeds = F.normalize(self.text_proj(vl_embs), dim=-1)
        query_si_feat = F.normalize(query_embeds.mean(1)) 



        with torch.no_grad():
            cu = output.cross_attentions


        return query_si_feat, cu, query_embeds  

    def fine_grained_sim(self, compose_feature_token, tar_video_token_embeds):
        compose_feature_token = compose_feature_token.unsqueeze(1).cuda()
        tar_video_token_embeds = tar_video_token_embeds.permute(0, 2, 1).cuda()
        similarity_results = []
        for i in range(compose_feature_token.size(0)):
            single_compose_feature_token = compose_feature_token[i].unsqueeze(0) 
            similarity = torch.matmul(single_compose_feature_token, tar_video_token_embeds) 
            max_similarity = similarity.max(-1)[0]  
            max_similarity = max_similarity.cuda()
            token_level_logits_ = torch.sum(max_similarity \
            * torch.softmax(torch.matmul(torch.softmax(max_similarity / 1e-2, dim=-1), self.compose_token_mat_weight) / 1e-2, dim=-1), dim=-1)
        
            similarity_results.append(token_level_logits_.cpu())
        token_tmp_logits = torch.cat(similarity_results, dim=0)
        logits = token_tmp_logits.cpu().numpy()

        return logits

    def textual_feature(self, caption, device='cuda'):

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        text_features = self.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)

        text_features = text_features[:, 0, :]
        return text_features


    def visual_feature(self, tar_img, device='cuda'):
        if self.train_vit:
            tar_img_embs = self.ln_vision(self.visual_encoder(tar_img))
        else:
            with torch.no_grad():
                tar_img_embs = self.ln_vision(self.visual_encoder(tar_img))
        tar_img_embs = tar_img_embs.float()
        query_tokens = self.query_tokens.expand(
            tar_img_embs.shape[0], -1, -1
        )
        tar_img_atts = torch.ones(tar_img_embs.size()[:-1], dtype=torch.long).to(device)
        tar_img_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=tar_img_embs,
            encoder_attention_mask=tar_img_atts,
            return_dict=True,
        )
        vl_embs = tar_img_output.last_hidden_state[:, : query_tokens.size(1), :]
        target_si_feat = F.normalize(self.vision_proj(vl_embs), dim=-1)

        return target_si_feat.mean(dim=1)

    def generate_anchor(self, feat1, feat2):
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        contribution_q = self.contribution_decoder(feat1, feat2)
        if len(feat2) == 2:
            dir_vec= feat2.unsqueeze(1) - contribution_q
        else:
            dir_vec = feat2 - contribution_q

        
        dir_vec = F.normalize(dir_vec, p=2, dim=2)

        
        similarity = torch.matmul(feat2, feat1.transpose(0, 1)) 

        
        point = self.point_weights(similarity)
        point = torch.exp(point)

        if len(feat2) == 2:
            anchor = feat2.unsqueeze(1) + point * dir_vec
        else:
            anchor = feat2 + point * dir_vec

        k_weights = self.text_anchor_to_k(anchor)   

        k_weights = k_weights.permute(0, 2, 1) 
        dynamic_c_anchor = torch.bmm(k_weights, anchor)  
        dynamic_c_anchor = F.normalize(dynamic_c_anchor, dim=-1)

        return dynamic_c_anchor




    def forward(self, batch, fabric):
        ref_img = batch["ref_img"]
        tar_img = batch["tar_img"]
        caption = batch["edit"]

        ref_description = batch["ref_description"]
        tag_description = batch["tag_description"]
        ref_img.half()
        tar_img.half()

        if ref_img.dim() == 5:
            self.ifFrames = True


        device = ref_img.device
        query_si_feat,_,query_embeds = self.compose_feature(ref_img, caption, ref_description, fabric, device)

        with torch.no_grad():
            text_mod_feat = self.textual_feature(caption, device) 

        tar_img_feat, tar_img_embeds ,_= self.target_fea(tar_img, tag_description, fabric, device)

        with torch.no_grad():
            ref_img_feat, ref_img_embeds, _ = self.target_fea(ref_img, ref_description, fabric, device)


        token_tmp_logits = torch.matmul(query_embeds.unsqueeze(1), tar_img_embeds.permute(0, 2, 1)).max(-1)[0]
        token_level_logits = torch.sum(token_tmp_logits \
            * torch.softmax(torch.matmul(torch.softmax(token_tmp_logits / 1e-2, dim=-1), self.compose_token_mat_weight) / 1e-2, dim=-1), dim=-1)#.t()


        sim_i2t = token_level_logits / self.temp
        bs = token_level_logits.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            query_si_feat.device
        )
        loss_itc = F.cross_entropy(sim_i2t, targets)


        loss = 0
        if self.si_ti_weight > 0:
            loss = loss_itc 
            
        self.r_anchor = self.generate_anchor(ref_img_feat.detach(), query_embeds)
        self.t_anchor = self.generate_anchor(text_mod_feat.detach(), query_embeds)


        r_anchor = self.r_anchor
        t_anchor = self.t_anchor
        
        directional_anchor = F.normalize(r_anchor - query_embeds + t_anchor - query_embeds, dim=-1)#

        anchor_tmp_logits = torch.matmul(directional_anchor.unsqueeze(1), (tar_img_embeds - query_embeds).permute(0, 2, 1)).max(-1)[0]
        anchor_level_logits = torch.sum(anchor_tmp_logits \
            * torch.softmax(torch.matmul(torch.softmax(anchor_tmp_logits / 1e-2, dim=-1), self.compose_anchor_mat_weight) / 1e-2, dim=-1), dim=-1)#.t()

        anchor_sim_i2t = anchor_level_logits / self.temp
        bs = anchor_level_logits.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            query_si_feat.device
        )
        distance_alignment = F.cross_entropy(anchor_sim_i2t, targets)


        tmp_ref_to_t = torch.matmul(r_anchor.unsqueeze(1), tar_img_embeds.permute(0, 2, 1)).max(-1)[0]
        ref_to_t = tmp_ref_to_t.permute(0, 2, 1)
        tmp_text_to_t = torch.matmul(t_anchor.unsqueeze(1), tar_img_embeds.permute(0, 2, 1)).max(-1)[0]
        text_to_t = tmp_text_to_t.permute(0, 2, 1)
        
        
        
        evidence_ref = self.evidence_compute(ref_to_t)
        evidence_text = self.evidence_compute(text_to_t)

        
        evi_loss_ref = self.loss_evi(sim_i2t, evidence_ref)
        evi_loss_text = self.loss_evi(sim_i2t, evidence_text)

        uct_loss = ((evi_loss_ref + evi_loss_text) / 2)

        total_loss = loss + \
                    + self.kappa_ * distance_alignment + \
                     self.lambda_ * uct_loss 


        return total_loss


    def evidence_compute(self, sims):
        E = torch.exp(sims / self.tau)  
        evidence = E + 1
        return evidence




def retrack(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model