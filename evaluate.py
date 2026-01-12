import torch
from torch.utils.data import DataLoader
import argparse
import config
from src.data.dataset import CodeSummaryDataset, collate_fn
from src.data.tokenizer import CodeTokenizer
from src.models.transformer import CodeSummarizationTransformer
from src.evaluation.metrics import evaluate_model, compute_perplexity
from src.training.trainer import LabelSmoothingLoss


def main():
    parser = argparse.ArgumentParser(description='Evaluate code summarization model')
    parser.add_argument('--checkpoint', type=str, default=str(config.BEST_MODEL_PATH),
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--show-examples', type=int, default=5,
                        help='Number of example outputs to display')
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = CodeTokenizer()
    tokenizer.load(config.TOKENIZER_PATH)
    vocab_size = len(tokenizer)
    
    print(f"Loading {args.split} data...")
    data_path = config.TEST_DATA_PATH if args.split == 'test' else config.VAL_DATA_PATH
    data = torch.load(data_path)
    
    max_samples = getattr(config, 'MAX_TEST_SAMPLES', len(data['codes']))
    
    dataset = CodeSummaryDataset(
        codes=data['codes'][:max_samples],
        summaries=data['summaries'][:max_samples],
        tokenizer=tokenizer
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Evaluation samples: {len(dataset)}")
    
    print("Loading model...")
    model = CodeSummarizationTransformer(vocab_size=vocab_size)
    
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown') + 1}")
    
    print("\nComputing perplexity...")
    criterion = LabelSmoothingLoss(vocab_size, config.PAD_IDX, smoothing=0.0)
    perplexity = compute_perplexity(model, data_loader, criterion, config.DEVICE)
    print(f"Perplexity: {perplexity:.2f}")
    
    print("\nGenerating summaries and computing metrics...")
    results, references, hypotheses = evaluate_model(
        model, data_loader, tokenizer, config.DEVICE, max_samples=args.max_samples
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"\nBLEU Scores:")
    print(f"  BLEU-1: {results['bleu-1']:.2f}")
    print(f"  BLEU-2: {results['bleu-2']:.2f}")
    print(f"  BLEU-3: {results['bleu-3']:.2f}")
    print(f"  BLEU-4: {results['bleu-4']:.2f}")
    print(f"  BLEU (combined): {results['bleu']:.2f}")
    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {results['rouge-1']:.2f}")
    print(f"  ROUGE-2: {results['rouge-2']:.2f}")
    print(f"  ROUGE-L: {results['rouge-L']:.2f}")
    print(f"\nPerplexity: {perplexity:.2f}")
    
    if args.show_examples > 0:
        print("\n" + "="*50)
        print("EXAMPLE OUTPUTS")
        print("="*50)
        
        for i in range(min(args.show_examples, len(references))):
            print(f"\n--- Example {i+1} ---")
            print(f"Reference: {references[i]}")
            print(f"Generated: {hypotheses[i]}")


if __name__ == "__main__":
    main()


