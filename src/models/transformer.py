import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
import config


class CodeSummarizationTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = config.VOCAB_SIZE,
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        n_encoder_layers: int = config.N_ENCODER_LAYERS,
        n_decoder_layers: int = config.N_DECODER_LAYERS,
        d_ff: int = config.D_FF,
        max_code_len: int = config.MAX_CODE_LENGTH,
        max_summary_len: int = config.MAX_SUMMARY_LENGTH,
        dropout: float = config.DROPOUT,
        pad_idx: int = config.PAD_IDX
    ):
        super().__init__()
        self.pad_idx = pad_idx
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            max_len=max_code_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            max_len=max_summary_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len = tgt.size()
        
        padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        
        mask = padding_mask & causal_mask
        
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
            
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        return decoder_output
    
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = config.MAX_SUMMARY_LENGTH,
        bos_idx: int = config.BOS_IDX,
        eos_idx: int = config.EOS_IDX,
        beam_size: int = config.BEAM_SIZE,
        temperature: float = config.TEMPERATURE
    ) -> torch.Tensor:
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        src_mask = self.create_src_mask(src)
        encoder_output = self.encoder(src, src_mask)
        
        if beam_size == 1:
            return self._greedy_decode(
                encoder_output, src_mask, max_len, bos_idx, eos_idx, device
            )
        else:
            return self._beam_search(
                encoder_output, src_mask, max_len, bos_idx, eos_idx, 
                beam_size, temperature, device, batch_size
            )
    
    def _greedy_decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
        device: torch.device
    ) -> torch.Tensor:
        batch_size = encoder_output.size(0)
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = self.create_tgt_mask(generated)
            output = self.decoder(generated, encoder_output, tgt_mask, src_mask)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == eos_idx).all():
                break
                
        return generated
    
    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
        beam_size: int,
        temperature: float,
        device: torch.device,
        batch_size: int
    ) -> torch.Tensor:
        results = []
        
        for b in range(batch_size):
            enc_out = encoder_output[b:b+1]
            s_mask = src_mask[b:b+1]
            
            enc_out = enc_out.repeat(beam_size, 1, 1)
            s_mask = s_mask.repeat(beam_size, 1, 1, 1)
            
            sequences = torch.full((beam_size, 1), bos_idx, dtype=torch.long, device=device)
            scores = torch.zeros(beam_size, device=device)
            
            for step in range(max_len - 1):
                tgt_mask = self.create_tgt_mask(sequences)
                output = self.decoder(sequences, enc_out, tgt_mask, s_mask)
                logits = output[:, -1, :] / temperature
                log_probs = torch.log_softmax(logits, dim=-1)
                
                if step == 0:
                    top_scores, top_tokens = log_probs[0].topk(beam_size)
                    sequences = torch.cat([
                        sequences[0:1].repeat(beam_size, 1),
                        top_tokens.unsqueeze(1)
                    ], dim=1)
                    scores = top_scores
                else:
                    vocab_size = log_probs.size(-1)
                    next_scores = scores.unsqueeze(1) + log_probs
                    next_scores = next_scores.view(-1)
                    
                    top_scores, top_indices = next_scores.topk(beam_size)
                    beam_indices = top_indices // vocab_size
                    token_indices = top_indices % vocab_size
                    
                    sequences = torch.cat([
                        sequences[beam_indices],
                        token_indices.unsqueeze(1)
                    ], dim=1)
                    scores = top_scores
                
                if (sequences[:, -1] == eos_idx).all():
                    break
            
            best_idx = scores.argmax()
            results.append(sequences[best_idx])
        
        max_result_len = max(r.size(0) for r in results)
        padded_results = torch.zeros(batch_size, max_result_len, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded_results[i, :r.size(0)] = r
            
        return padded_results


