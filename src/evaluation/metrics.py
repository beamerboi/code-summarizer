import torch
import math
from collections import Counter
import nltk
from rouge_score import rouge_scorer
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def compute_bleu(references: list[str], hypotheses: list[str], max_n: int = 4) -> dict:
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def brevity_penalty(ref_len, hyp_len):
        if hyp_len >= ref_len:
            return 1.0
        return math.exp(1 - ref_len / hyp_len)
    
    precisions = []
    total_ref_len = 0
    total_hyp_len = 0
    
    for n in range(1, max_n + 1):
        match_count = 0
        total_count = 0
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()
            
            if n == 1:
                total_ref_len += len(ref_tokens)
                total_hyp_len += len(hyp_tokens)
            
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            hyp_ngrams = Counter(get_ngrams(hyp_tokens, n))
            
            for ngram, count in hyp_ngrams.items():
                match_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += sum(hyp_ngrams.values())
        
        precision = match_count / total_count if total_count > 0 else 0
        precisions.append(precision)
    
    bp = brevity_penalty(total_ref_len, total_hyp_len)
    
    if all(p > 0 for p in precisions):
        log_precision = sum(math.log(p) for p in precisions) / max_n
        bleu = bp * math.exp(log_precision)
    else:
        bleu = 0.0
    
    return {
        'bleu': bleu * 100,
        'bleu-1': precisions[0] * 100,
        'bleu-2': precisions[1] * 100 if len(precisions) > 1 else 0,
        'bleu-3': precisions[2] * 100 if len(precisions) > 2 else 0,
        'bleu-4': precisions[3] * 100 if len(precisions) > 3 else 0,
        'brevity_penalty': bp
    }


def compute_rouge(references: list[str], hypotheses: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge-1': sum(rouge1_scores) / len(rouge1_scores) * 100,
        'rouge-2': sum(rouge2_scores) / len(rouge2_scores) * 100,
        'rouge-L': sum(rougeL_scores) / len(rougeL_scores) * 100
    }


def compute_perplexity(model, data_loader, criterion, device) -> float:
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            code_ids = batch['code_ids'].to(device)
            summary_ids = batch['summary_ids'].to(device)
            
            tgt_input = summary_ids[:, :-1]
            tgt_output = summary_ids[:, 1:]
            
            logits = model(code_ids, tgt_input)
            
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            target_flat = tgt_output.contiguous().view(-1)
            
            loss = torch.nn.functional.cross_entropy(
                logits_flat, 
                target_flat, 
                ignore_index=0,
                reduction='sum'
            )
            
            non_pad_mask = target_flat != 0
            num_tokens = non_pad_mask.sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def evaluate_model(model, data_loader, tokenizer, device, max_samples: int = None) -> dict:
    model.eval()
    
    references = []
    hypotheses = []
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating summaries"):
            code_ids = batch['code_ids'].to(device)
            ref_summaries = batch['summary_texts']
            
            generated = model.generate(code_ids, beam_size=5)
            
            for i in range(generated.size(0)):
                gen_ids = generated[i].tolist()
                hyp = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                references.append(ref_summaries[i])
                hypotheses.append(hyp)
                
                samples_processed += 1
                if max_samples and samples_processed >= max_samples:
                    break
            
            if max_samples and samples_processed >= max_samples:
                break
    
    bleu_scores = compute_bleu(references, hypotheses)
    rouge_scores = compute_rouge(references, hypotheses)
    
    results = {
        **bleu_scores,
        **rouge_scores,
        'num_samples': len(references)
    }
    
    return results, references, hypotheses


