from __future__ import annotations

import torch

from .inference_config import StyleTTS2Params
from .random_state import request_generator
from .model_utils import timed_operation

class BatchStyleMixin:
    def _run_model_forward(
        self,
        tokens_tensor: torch.Tensor,
        input_lengths_tensor: torch.Tensor,
        text_mask: torch.Tensor,
        ref_s_batch: torch.Tensor,
        prev_s_list: list,
        styletts2_params_list: list[StyleTTS2Params],
        batch_size: int,
        input_lengths: list[int],
    ) -> tuple[torch.Tensor, list, list[int], torch.Tensor | None]:
        with timed_operation("model_forward"):
            torch.compiler.cudagraph_mark_step_begin()
            
            use_diffusion_list = [p.use_diffusion for p in styletts2_params_list]
            use_diffusion_any = any(use_diffusion_list)

            with timed_operation("text_encoder"):
                t_en = self._model.text_encoder(tokens_tensor, input_lengths_tensor, text_mask)
            
            with timed_operation("bert"):
                bert_dur = self._model.bert(tokens_tensor, attention_mask=(~text_mask).int())
            
            with timed_operation("bert_encoder"):
                d_en = self._model.bert_encoder(bert_dur).transpose(-1, -2)
            
            s_pred = torch.zeros(batch_size, 256, device=self._device)
            
            if use_diffusion_any:
                with timed_operation("diffusion_sampling"):
                    diffusion_indices = [i for i in range(batch_size) if use_diffusion_list[i]]
                    non_diffusion_indices = [i for i in range(batch_size) if not use_diffusion_list[i]]
                    
                    embedding_scales = [styletts2_params_list[i].embedding_scale for i in diffusion_indices]
                    diffusion_steps_list = [styletts2_params_list[i].diffusion_steps for i in diffusion_indices]
                    
                    param_groups: dict[tuple[float, int], list[int]] = {}
                    for idx, i in enumerate(diffusion_indices):
                        key = (embedding_scales[idx], diffusion_steps_list[idx])
                        if key not in param_groups:
                            param_groups[key] = []
                        param_groups[key].append((idx, i))
                    
                    for (embedding_scale, diffusion_steps), group_items in param_groups.items():
                        group_indices_in_batch = [item[1] for item in group_items]
                        group_batch_size = len(group_items)
                        
                        noise = torch.stack(
                            [
                                torch.randn(
                                    (256,),
                                    device=self._device,
                                    generator=request_generator(
                                        styletts2_params_list[index].seed,
                                        self._device,
                                    ),
                                )
                                for index in group_indices_in_batch
                            ]
                        ).unsqueeze(1)
                        bert_dur_group = bert_dur[group_indices_in_batch]
                        ref_s_group = ref_s_batch[group_indices_in_batch]

                        if self._trt.diffusion_enabled:
                            s_pred_group = self._trt.run_diffusion_sampler(
                                noise,
                                bert_dur_group,
                                ref_s_group,
                                embedding_scale=embedding_scale,
                                diffusion_steps=diffusion_steps,
                            ).squeeze(1).clone()
                        else:
                            noise_padded, _ = self._pad_to_batch_size(noise, group_batch_size)
                            bert_dur_padded, _ = self._pad_to_batch_size(bert_dur_group, group_batch_size)
                            ref_s_batch_padded, _ = self._pad_to_batch_size(ref_s_group, group_batch_size)

                            s_pred_group_padded = self._sampler(
                                noise=noise_padded,
                                embedding=bert_dur_padded,
                                embedding_scale=embedding_scale,
                                features=ref_s_batch_padded,
                                num_steps=diffusion_steps
                            ).squeeze(1).clone()

                            s_pred_group = s_pred_group_padded[:group_batch_size]
                        style_norm = torch.linalg.vector_norm(s_pred_group.float(), dim=1)
                        use_style = torch.isfinite(s_pred_group).all(dim=1) & (style_norm <= 10.0)
                        s_pred_group = torch.where(use_style[:, None], s_pred_group, ref_s_group.to(s_pred_group.dtype))
                        for group_idx, batch_idx in enumerate(group_indices_in_batch):
                            s_pred[batch_idx] = s_pred_group[group_idx]
                    
                    if non_diffusion_indices:
                        for i in non_diffusion_indices:
                            s_pred[i] = ref_s_batch[i]
                
                if any(ps is not None for ps in prev_s_list):
                    for i in range(batch_size):
                        if use_diffusion_list[i]:
                            t = styletts2_params_list[i].style_interpolation_factor
                            if prev_s_list[i] is not None:
                                ps = prev_s_list[i].to(self._device).squeeze()
                                s_pred[i] = t * ps + (1 - t) * s_pred[i]
                
                s = s_pred[:, 128:]
                ref = s_pred[:, :128]
                
                for i in range(batch_size):
                    alpha = styletts2_params_list[i].alpha
                    beta = styletts2_params_list[i].beta
                    ref[i] = alpha * ref[i] + (1 - alpha) * ref_s_batch[i, :128]
                    s[i] = beta * s[i] + (1 - beta) * ref_s_batch[i, 128:]
                s_pred = torch.cat([ref, s], dim=-1)
            else:
                s = ref_s_batch[:, 128:]
                ref = ref_s_batch[:, :128]
            
            with timed_operation("predictor"):
                d = self._model.predictor.text_encoder(d_en, s, input_lengths_tensor, text_mask)
                
                x, _ = self._model.predictor.lstm(d)
                
                duration = self._model.predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1)
                speed_tensor = torch.tensor([styletts2_params_list[i].speed for i in range(batch_size)], device=self._device).unsqueeze(1)
                duration = duration / speed_tensor
                pred_dur = torch.round(duration).clamp(min=1)
                pred_dur = pred_dur * (~text_mask).float()
                
            decoder_type = getattr(self._model_config.decoder, 'type', None) if hasattr(self._model_config, 'decoder') else None
            
            with timed_operation("alignment_computation"):
                all_pred_aln_trg = []
                all_total_mel_frames = []
                all_actual_lengths = []
                
                for b in range(batch_size):
                    batch_pred_dur = pred_dur[b]
                    batch_input_length = input_lengths[b]
                    batch_pred_dur_actual = batch_pred_dur[:batch_input_length]
                    
                    total_mel_frames = int(batch_pred_dur_actual.sum().item())
                    all_total_mel_frames.append(total_mel_frames)
                    all_actual_lengths.append(total_mel_frames)
                    
                    if total_mel_frames == 0:
                        pred_aln_trg = torch.zeros(batch_input_length, 0, device=self._device)
                    else:
                        pred_aln_trg = torch.zeros(batch_input_length, total_mel_frames, device=self._device)
                        
                        dur_ints = batch_pred_dur_actual.int()
                        cumsum_dur = torch.cumsum(dur_ints, dim=0)
                        frame_indices = torch.arange(total_mel_frames, device=self._device).unsqueeze(0)
                        
                        start_frames = torch.cat([torch.zeros(1, device=self._device, dtype=torch.long), cumsum_dur[:-1]])
                        end_frames = cumsum_dur
                        
                        mask = (frame_indices >= start_frames.unsqueeze(1)) & (frame_indices < end_frames.unsqueeze(1))
                        pred_aln_trg[mask] = 1.0
                    
                    all_pred_aln_trg.append(pred_aln_trg)
                
                max_mel_frames = max(all_actual_lengths) if all_actual_lengths else 0
                max_input_length = max(input_lengths)
                
                batch_pred_aln_trg = torch.zeros(batch_size, max_input_length, max_mel_frames, device=self._device)
                for b in range(batch_size):
                    aln = all_pred_aln_trg[b]
                    if aln.shape[1] > 0:
                        batch_pred_aln_trg[b, :aln.shape[0], :aln.shape[1]] = aln
                
                all_pred_aln_trg = [aln.cpu().numpy() for aln in all_pred_aln_trg]  
            
            with timed_operation("F0Ntrain"):
                en = torch.bmm(d.transpose(-1, -2), batch_pred_aln_trg)
                if decoder_type == "hifigan":
                    asr_new = torch.zeros_like(en)
                    asr_new[:, :, 0] = en[:, :, 0]
                    asr_new[:, :, 1:] = en[:, :, 0:-1]
                    en = asr_new
                
                F0_pred, N_pred = self._model.predictor.F0Ntrain(en, s)
            
            with timed_operation("prepare_decoder_inputs"):
                asr = torch.bmm(t_en, batch_pred_aln_trg)
                if decoder_type == "hifigan":
                    asr_new = torch.zeros_like(asr)
                    asr_new[:, :, 0] = asr[:, :, 0]
                    asr_new[:, :, 1:] = asr[:, :, 0:-1]
                    asr = asr_new
                
                if self._trt.decoder_enabled:
                    decoder_inputs = self._trt.prepare_decoder_inputs(
                        asr,
                        F0_pred,
                        N_pred,
                        ref,
                        batch_size=batch_size,
                    )
                    asr_padded = decoder_inputs["asr"]
                    F0_pred_padded = decoder_inputs["f0"]
                    N_pred_padded = decoder_inputs["noise"]
                    ref_padded = decoder_inputs["style"]
                else:
                    asr_seq_padded = self._pad_to_multiple(asr, 128, dim=-1)
                    F0_pred_seq_padded = self._pad_to_multiple(F0_pred, 256, dim=-1)
                    N_pred_seq_padded = self._pad_to_multiple(N_pred, 256, dim=-1)

                    asr_padded, _ = self._pad_to_batch_size(asr_seq_padded, batch_size)
                    F0_pred_padded, _ = self._pad_to_batch_size(F0_pred_seq_padded, batch_size)
                    N_pred_padded, _ = self._pad_to_batch_size(N_pred_seq_padded, batch_size)
                    ref_padded, _ = self._pad_to_batch_size(ref, batch_size)
            
            with timed_operation("decoder"):
                if self._trt.decoder_enabled:
                    har = self._model.decoder.generator._preprocess_f0(F0_pred_padded)
                    out_padded = self._trt.run_decoder(
                        {
                            "asr": asr_padded,
                            "f0": F0_pred_padded,
                            "noise": N_pred_padded,
                            "style": ref_padded,
                            "har": har,
                        }
                    )
                else:
                    out_padded = self._model.decoder(asr_padded, F0_pred_padded, N_pred_padded, ref_padded)
            
            out = out_padded[:batch_size]
            
            s_pred_for_results = s_pred if use_diffusion_any else None
            
            return out, all_pred_aln_trg, all_actual_lengths, s_pred_for_results
    
