import torch
import argparse
import config
from src.data.tokenizer import CodeTokenizer
from src.data.preprocessing import preprocess_code
from src.models.transformer import CodeSummarizationTransformer


def summarize_code(
    code: str,
    model: CodeSummarizationTransformer,
    tokenizer: CodeTokenizer,
    device: torch.device,
    beam_size: int = config.BEAM_SIZE,
    temperature: float = config.TEMPERATURE
) -> str:
    model.eval()
    
    processed_code = preprocess_code(code)
    
    code_ids = tokenizer.encode(processed_code, max_length=config.MAX_CODE_LENGTH)
    code_tensor = torch.tensor([code_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(
            code_tensor,
            max_len=config.MAX_SUMMARY_LENGTH,
            beam_size=beam_size,
            temperature=temperature
        )
    
    summary_ids = generated[0].tolist()
    summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Generate summary for Python code')
    parser.add_argument('--input', type=str, required=True,
                        help='Python code snippet to summarize')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to Python file to summarize')
    parser.add_argument('--checkpoint', type=str, default=str(config.BEST_MODEL_PATH),
                        help='Path to model checkpoint')
    parser.add_argument('--beam-size', type=int, default=config.BEAM_SIZE,
                        help='Beam size for decoding')
    parser.add_argument('--temperature', type=float, default=config.TEMPERATURE,
                        help='Temperature for sampling')
    args = parser.parse_args()
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            code = f.read()
    else:
        code = args.input
    
    print("Loading tokenizer...")
    tokenizer = CodeTokenizer()
    tokenizer.load(config.TOKENIZER_PATH)
    vocab_size = len(tokenizer)
    
    print("Loading model...")
    model = CodeSummarizationTransformer(vocab_size=vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    
    print("\nInput code:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    
    summary = summarize_code(
        code=code,
        model=model,
        tokenizer=tokenizer,
        device=config.DEVICE,
        beam_size=args.beam_size,
        temperature=args.temperature
    )
    
    print("\nGenerated summary:")
    print("-" * 40)
    print(summary)
    print("-" * 40)


if __name__ == "__main__":
    main()


