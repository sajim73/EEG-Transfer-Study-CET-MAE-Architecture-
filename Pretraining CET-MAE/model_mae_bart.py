# -*- coding: utf-8 -*-
# @Time : 29/11/23 4:02 PM

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartTokenizer, XLMRobertaTokenizer, XLMRobertaModel, T5Model, T5Tokenizer

from Multi_Stream_TransformerEncoder import Multi_Stream_TransformerEncoder, Multi_Stream_TransformerEncoderLayer


def check_nan_inf(x, name="tensor"):
    if torch.isnan(x).any():
        raise ValueError(f"{name} contains NaN")
    if torch.isinf(x).any():
        raise ValueError(f"{name} contains Inf")


def Pooler(encoded_embedding, attention_mask):
    denom = attention_mask.sum(-1).unsqueeze(-1).clamp(min=1)
    return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / denom


def compute_sentencelevel_contrastive_logits(projection_embeddings, inputs_attn_mask_batch, target_input_ids_batch, text_llm):
    batch_size = projection_embeddings.shape[0]
    EEG_features = Pooler(projection_embeddings, inputs_attn_mask_batch)

    logit_scale = nn.Parameter(torch.ones([], device=projection_embeddings.device) * np.log(1 / 0.07))
    Text_features = text_llm(
        input_ids=target_input_ids_batch["input_ids"],
        attention_mask=target_input_ids_batch["attention_mask"]
    ).last_hidden_state
    text_attention_mask = target_input_ids_batch["attention_mask"]
    Sentence_feature = Pooler(Text_features, text_attention_mask)

    EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    Sentence_feature = Sentence_feature / Sentence_feature.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    logits_per_EEG = logit_scale.exp() * EEG_features @ Sentence_feature.t()
    logits_per_text = logit_scale.exp() * Sentence_feature @ EEG_features.t()

    labels = torch.arange(batch_size, device=EEG_features.device).long()
    total_loss = (F.cross_entropy(logits_per_EEG, labels) + F.cross_entropy(logits_per_text, labels)) / 2
    return total_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device, dtype=x.dtype)
        return self.dropout(x)


class CETMAE_project_late_bart(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        eeg_dim=840,
        multi_heads=8,
        feedforward_dim=2048,
        trans_layers=6,
        decoder_embed_dim=840,
        pretrain_path="./models/huggingface/bart-large",
        norm_layer=nn.LayerNorm,
        device=0,
    ):
        super().__init__()
        print("A CET-MAE Model")
        self.device = torch.device(device)
        self.tokenizer = BartTokenizer.from_pretrained(pretrain_path)

        self.fc_eeg = nn.Linear(eeg_dim, embed_dim)
        self.act = nn.GELU()

        self.pos_embed_e = PositionalEncoding(eeg_dim)

        self.eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=eeg_dim,
            nhead=multi_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            norm_first=False,
        )
        self.e_branch = nn.TransformerEncoder(self.eeg_encoder_layer, num_layers=trans_layers)

        self.t_branch = BartModel.from_pretrained(pretrain_path)
        self.t_branch_encoder = self.t_branch.get_encoder()
        for param in self.t_branch.parameters():
            param.requires_grad = False

        self.unify_encoder_layer = Multi_Stream_TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=16,
            dim_feedforward=4096,
            batch_first=True,
            norm_first=False,
        )
        self.unify_branch = Multi_Stream_TransformerEncoder(self.unify_encoder_layer, num_layers=1)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_e = PositionalEncoding(eeg_dim)

        self.eeg_decoder_layers = nn.TransformerEncoderLayer(
            d_model=eeg_dim,
            nhead=multi_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            norm_first=False,
        )
        self.eeg_decoder = nn.TransformerEncoder(self.eeg_decoder_layers, num_layers=1)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_embed_e = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pred_t = nn.Linear(embed_dim, 50265, bias=True)

        self.loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def eeg_masking_preserve_order_last_position(self, x, mask_ratio, attention_mask):
        N, L, D = x.shape
        valid_lens = torch.sum(attention_mask, dim=1).int()
        len_keep = (valid_lens * (1 - mask_ratio)).int()
        len_keep = torch.where(valid_lens > 0, torch.clamp(len_keep, min=1), len_keep)

        masked_x_list = []
        ids_keep_list = []
        ids_restore_list = []
        max_len = 0

        for i in range(N):
            valid_positions = torch.nonzero(attention_mask[i] > 0, as_tuple=False).squeeze(1)
            if valid_positions.numel() == 0:
                ids_keep_i_sorted = valid_positions
                ids_remove_i_sorted = valid_positions
                masked_x_i = x[i][:0]
            else:
                last_attention_index = valid_positions[-1]
                selectable_positions = valid_positions[:-1]

                if selectable_positions.numel() > 0:
                    rand_order = torch.randperm(selectable_positions.numel(), device=x.device)
                    keep_count = min(max(1, len_keep[i].item()), selectable_positions.numel())
                    ids_keep_i = selectable_positions[rand_order[:keep_count]]
                    ids_keep_i_sorted = ids_keep_i[torch.argsort(ids_keep_i)]
                else:
                    ids_keep_i_sorted = valid_positions[:1]


                remove_mask = torch.ones(valid_positions.shape[0], dtype=torch.bool, device=x.device)
                if ids_keep_i_sorted.numel() > 0:
                    keep_match = valid_positions.unsqueeze(1) == ids_keep_i_sorted.unsqueeze(0)
                    remove_mask = ~keep_match.any(dim=1)

                ids_remove_i = valid_positions[remove_mask]
                ids_remove_i_sorted = ids_remove_i[torch.argsort(ids_remove_i)]
                masked_x_i = torch.index_select(x[i], dim=0, index=ids_keep_i_sorted) if ids_keep_i_sorted.numel() > 0 else x[i][:0]

            ids_keep_list.append(ids_keep_i_sorted)
            ids_restore_list.append(ids_remove_i_sorted)
            masked_x_list.append(masked_x_i)
            max_len = max(max_len, masked_x_i.shape[0])

        padded_masked_x_list = []
        for masked_x_i in masked_x_list:
            pad_len = max_len - masked_x_i.shape[0]
            padded_masked_x_list.append(F.pad(masked_x_i, (0, 0, 0, pad_len)))

        masked_x = torch.stack(padded_masked_x_list, dim=0)

        masked_attention_mask = torch.zeros((N, max_len), device=x.device, dtype=attention_mask.dtype)
        masked_attention_mask_invert = torch.ones((N, max_len), device=x.device, dtype=attention_mask.dtype)
        for i in range(N):
            k = ids_keep_list[i].shape[0]
            if k > 0:
                masked_attention_mask[i, :k] = 1
                masked_attention_mask_invert[i, :k] = 0

        return masked_x, ids_keep_list, ids_restore_list, masked_attention_mask, masked_attention_mask_invert

    def mask_batch_text_tokens(self, inputs, tokenizer, mlm_probability=0.15, is_train=True):
        if tokenizer.mask_token is None:
            raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling.")

        if inputs.dim() == 3 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 2:
            raise ValueError(f"Expected 2D input_ids, got shape {tuple(inputs.shape)}")

        inputs = inputs.long()
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, mlm_probability, device=labels.device)

        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
            for seq in labels
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        if tokenizer.pad_token_id is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        inputs = inputs.clone()
        inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        effective_masked_indices = masked_indices & (labels != -100)
        return inputs, labels, effective_masked_indices

    def forward_encoder(self, e, e_attn_mask, t, mask_ratio_e, mlm_probability):
        e = e + self.pos_embed_e(e)

        e_masked, ids_keep, ids_restore, masked_attention_mask, masked_attention_mask_invert = \
            self.eeg_masking_preserve_order_last_position(e, mask_ratio_e, e_attn_mask)

        text_input_ids = t["input_ids"] if isinstance(t, dict) else t.input_ids
        text_attention_mask = t["attention_mask"] if isinstance(t, dict) else t.attention_mask
        text_attention_mask_invert = t["attention_mask_invert"] if isinstance(t, dict) else t.attention_mask_invert

        if text_input_ids.dim() == 3 and text_input_ids.size(1) == 1:
            text_input_ids = text_input_ids.squeeze(1)
        if text_attention_mask.dim() == 3 and text_attention_mask.size(1) == 1:
            text_attention_mask = text_attention_mask.squeeze(1)
        if text_attention_mask_invert.dim() == 3 and text_attention_mask_invert.size(1) == 1:
            text_attention_mask_invert = text_attention_mask_invert.squeeze(1)
        if text_input_ids.dim() == 1:
            text_input_ids = text_input_ids.unsqueeze(0)
        if text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if text_attention_mask_invert.dim() == 1:
            text_attention_mask_invert = text_attention_mask_invert.unsqueeze(0)

        text_input_ids = text_input_ids.long()
        text_attention_mask = text_attention_mask.long()
        text_attention_mask_invert = text_attention_mask_invert.long()

        text_mlm_input_ids, text_mlm_labels, mlm_indices = self.mask_batch_text_tokens(
            text_input_ids, self.tokenizer, mlm_probability=mlm_probability
        )

        e_branch_embeddings = self.e_branch(e_masked, src_key_padding_mask=masked_attention_mask_invert.bool())
        e_branch_embeddings = self.act(self.fc_eeg(e_branch_embeddings))

        t_branch_embeddings = self.t_branch_encoder(
            input_ids=text_mlm_input_ids,
            attention_mask=text_attention_mask
        ).last_hidden_state

        target_dtype = self.fc_eeg.weight.dtype
        e_branch_embeddings = e_branch_embeddings.to(target_dtype)
        t_branch_embeddings = t_branch_embeddings.to(target_dtype)
        masked_attention_mask_invert = masked_attention_mask_invert.to(torch.bool)
        text_attention_mask_invert = text_attention_mask_invert.to(torch.bool)

        unify_embeddings = torch.cat((e_branch_embeddings, t_branch_embeddings), dim=1)
        unify_attention_mask_invert = torch.cat((masked_attention_mask_invert, text_attention_mask_invert), dim=1)

        unify_branch_embeddings = self.unify_branch(
            unify_embeddings,
            src_key_padding_mask=unify_attention_mask_invert,
            modality=None
        )

        _, L_e, _ = e_branch_embeddings.shape
        x_eeg = unify_branch_embeddings[:, :L_e, :]
        x_text = unify_branch_embeddings[:, L_e:, :]

        ce = self.unify_branch(
            e_branch_embeddings,
            src_key_padding_mask=masked_attention_mask_invert,
            modality="e"
        )

        ct = self.unify_branch(
            t_branch_embeddings,
            src_key_padding_mask=text_attention_mask_invert,
            modality="t"
        )

        return x_eeg, x_text, ids_keep, ids_restore, masked_attention_mask, text_mlm_input_ids, text_mlm_labels, mlm_indices, ce, ct

    def forward_decoder(self, masked_e, eeg_attn_mask_invert, ids_keep_list, ids_restore_list):
        e_decoder = self.act(self.decoder_embed_e(masked_e))

        batch_size, _, dim = e_decoder.shape
        full_len = eeg_attn_mask_invert.shape[1]
        restored = torch.zeros(batch_size, full_len, dim, device=e_decoder.device, dtype=e_decoder.dtype)

        for i in range(batch_size):
            keep_indices = ids_keep_list[i]
            remove_indices = ids_restore_list[i]

            if keep_indices.numel() > 0:
                restored[i, keep_indices] = e_decoder[i, :keep_indices.numel()]

            if remove_indices.numel() > 0:
                restored[i, remove_indices] = self.mask_token[0, 0].to(e_decoder.dtype)

        e = restored + self.decoder_pos_embed_e(restored)
        e = self.eeg_decoder(e, src_key_padding_mask=eeg_attn_mask_invert.bool())
        e = self.decoder_norm(e)
        check_nan_inf(e, "decoder_eeg")
        return restored, e

    def Pooler(self, encoded_embedding, attention_mask):
        denom = attention_mask.sum(-1).unsqueeze(-1).clamp(min=1)
        return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / denom

    def text_Pooler(self, encoded_embedding, attention_mask):
        denom = attention_mask.sum(-1).unsqueeze(-1).clamp(min=1)
        return (encoded_embedding * attention_mask.unsqueeze(-1)).sum(1) / denom

    def masked_Pooler(self, encoded_embedding, attention_mask, masked_indices):
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if masked_indices.dim() == 1:
            masked_indices = masked_indices.unsqueeze(0)
        if encoded_embedding.dim() == 2:
            encoded_embedding = encoded_embedding.unsqueeze(0)

        masked_attention_mask = attention_mask.clone()
        masked_attention_mask[masked_indices] = 0
        sum_embed = (encoded_embedding * masked_attention_mask.unsqueeze(-1)).sum(1)
        sum_mask = masked_attention_mask.sum(-1).unsqueeze(-1).clamp(min=1)
        pooled_output = sum_embed / sum_mask
        return pooled_output


    def compute_sentencelevel_contrastive_logits(self, eeg_embeddings, eeg_attention, text_embedddings, text_attention, masked_indices):
        if eeg_attention.dim() == 1:
            eeg_attention = eeg_attention.unsqueeze(0)
        if text_attention.dim() == 1:
            text_attention = text_attention.unsqueeze(0)
        if masked_indices.dim() == 1:
            masked_indices = masked_indices.unsqueeze(0)
        if eeg_embeddings.dim() == 2:
            eeg_embeddings = eeg_embeddings.unsqueeze(0)
        if text_embedddings.dim() == 2:
            text_embedddings = text_embedddings.unsqueeze(0)
        
        batch_size = eeg_embeddings.shape[0]
        EEG_features = self.Pooler(eeg_embeddings, eeg_attention)
        Sentence_feature = self.masked_Pooler(text_embedddings, text_attention, masked_indices)

        EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        Sentence_feature = Sentence_feature / Sentence_feature.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        logit_scale = torch.exp(torch.tensor(np.log(1 / 0.07), device=EEG_features.device, dtype=EEG_features.dtype))
        logits_per_EEG = logit_scale * (EEG_features @ Sentence_feature.t())
        logits_per_text = logit_scale * (Sentence_feature @ EEG_features.t())

        labels = torch.arange(batch_size, device=EEG_features.device).long()
        total_loss = (F.cross_entropy(logits_per_EEG, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss

    def forward_contrastive(self, eeg_embeddings, text_embeddings, bidirect_contrast=False):
        eeg_embeddings = F.normalize(eeg_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        total = torch.mm(eeg_embeddings, text_embeddings.t()) / 0.05

        if not bidirect_contrast:
            nce = -torch.mean(torch.diag(F.log_softmax(total, dim=0)))
            c_acc = torch.sum(
                torch.eq(
                    torch.argmax(F.softmax(total, dim=0), dim=0),
                    torch.arange(0, total.shape[0], device=eeg_embeddings.device)
                )
            ) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(F.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(F.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(
                torch.eq(
                    torch.argmax(F.softmax(total, dim=0), dim=0),
                    torch.arange(0, total.shape[0], device=eeg_embeddings.device)
                )
            ) / total.shape[0]
            c_acc_2 = torch.sum(
                torch.eq(
                    torch.argmax(F.softmax(total.t(), dim=0), dim=0),
                    torch.arange(0, total.shape[0], device=eeg_embeddings.device)
                )
            ) / total.shape[0]
            return (nce_1 + nce_2) / 2, (c_acc_1 + c_acc_2) / 2

    def forward_loss_eeg(self, eeg, pred, eeg_ids_restore_list):
        losses = []
        for i, sample_ids in enumerate(eeg_ids_restore_list):
            if sample_ids.numel() == 0:
                continue
            eeg_sample = eeg[i][sample_ids]
            pred_sample = pred[i][sample_ids]
            loss_sample = ((pred_sample - eeg_sample) ** 2).mean(dim=-1)
            losses.append(loss_sample)

        if len(losses) == 0:
            return torch.tensor(0.0, device=eeg.device, dtype=eeg.dtype)

        losses_tensor = torch.cat(losses)
        return losses_tensor.mean()

    def forward(
        self,
        eeg,
        eeg_attn_mask,
        eeg_attn_mask_invert,
        text,
        mask_ratio_e=0.25,
        mlm_probability=0.5,
        mlm_loss_weight=0.5,
        mae_loss_weight=1.0,
        contrast_loss_weight=0.01,
        sim_loss_weight=0.0,
    ):
        latent_eeg, latent_text, eeg_ids_keep_list, eeg_ids_restore_list, masked_attention_mask, \
        text_mlm_inputs_ids, text_mlm_labels, mlm_indices, latent_c_eeg, latent_c_text = \
            self.forward_encoder(eeg, eeg_attn_mask, text, mask_ratio_e, mlm_probability)

        check_nan_inf(latent_eeg, "latent_eeg")

        project_e, pred_e = self.forward_decoder(
            latent_eeg, eeg_attn_mask_invert, eeg_ids_keep_list, eeg_ids_restore_list
        )
        check_nan_inf(pred_e, "pred_e")

        loss_mae_eeg = self.forward_loss_eeg(eeg, pred_e, eeg_ids_restore_list)
        loss_mae = mae_loss_weight * loss_mae_eeg

        mlm_logits = self.act(self.decoder_pred_t(latent_text))
        loss_mlm = self.loss_mlm(
            input=mlm_logits.view(-1, 50265),
            target=text_mlm_labels.view(-1)
        )
        loss_mlm = mlm_loss_weight * loss_mlm

        eeg_embeddings_whole_words = self.Pooler(project_e, eeg_attn_mask)
        last_one_indices = (torch.sum(eeg_attn_mask, dim=1).long() - 1).clamp(min=0, max=eeg.shape[1] - 1)
        eeg_sentence_embeddings = eeg[torch.arange(eeg.size(0), device=eeg.device), last_one_indices]
        cos_sim = F.cosine_similarity(eeg_embeddings_whole_words, eeg_sentence_embeddings, dim=1)
        loss_sim = sim_loss_weight * (1 - cos_sim.mean())

        text_attention_mask = text["attention_mask"] if isinstance(text, dict) else text.attention_mask
        if text_attention_mask.dim() == 3 and text_attention_mask.size(1) == 1:
            text_attention_mask = text_attention_mask.squeeze(1)

        loss_c = self.compute_sentencelevel_contrastive_logits(
            latent_c_eeg,
            masked_attention_mask,
            latent_c_text,
            text_attention_mask.long(),
            mlm_indices,
        )
        loss_c = contrast_loss_weight * loss_c

        loss = loss_mlm + loss_c + loss_mae
        check_nan_inf(loss, "loss")

        return loss_mae, loss_mlm, loss_c, loss_sim, loss
