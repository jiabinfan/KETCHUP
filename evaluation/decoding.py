import transformers
import torch
import argparse
import tqdm
from torch.cuda.amp import autocast
from utils.template_handler import TemplateHandler, FewshotTemplateHandler
from utils.helper import greedy_search, beam_search
from utils.dataset import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='data/test.src')
parser.add_argument('-o', '--output', type=str, default='ost_pseudo/test.tgt_gen')
parser.add_argument('-bs', '--beam-size', type=int, default=1)
parser.add_argument('-mn', '--model-name', required=True)
parser.add_argument('-lp', '--length-penalty', default=1.0, type=float)
parser.add_argument('-tn', '--tokenizer-name')
parser.add_argument('--max-sentences', type=int, default=2)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--do-sample', action='store_true')
parser.add_argument('--topk', type=int)
parser.add_argument('--top-p', type=float)
parser.add_argument('--max-length-a', type=float, default=1.5)
parser.add_argument('--max-length-b', type=int, default=50)
parser.add_argument('--min-max-length', type=int, default=8)
parser.add_argument('--max-length', type=int, default=512)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--template', type=str, default='templates/translation_T0.txt')
parser.add_argument('-cn', '--config-name', type=str, default='/mnt/nvme/guoqing/t5-base/config.json')
parser.add_argument('--length-ratio', type=float, default=2)
parser.add_argument('--max-tokens', type=int, default=1024)
parser.add_argument('--init-flat', action='store_true')
parser.add_argument('--mask-extra', action='store_true')


def decoding(args,model,tokenizer,device=None):
    set_seed(args.seed)
    model.config.length_penalty = args.length_penalty
    config=transformers.AutoConfig.from_pretrained(args.config_name)
    model.eval()
    finished = False
    bar = tqdm.tqdm()
    if 'gsm8k' in args.input:
        handler = FewshotTemplateHandler(args.template, tokenizer)
    else:
        handler = TemplateHandler(args.template, tokenizer)
    try:
        with open(args.input, encoding='utf8') as fp, \
            open(args.output, 'w', encoding='utf8') as ofp:
            while not finished:
                torch.cuda.empty_cache()
                lines = []

                for _ in range(args.max_sentences):
                    line = fp.readline()
                    if len(line)<=1:
                        finished = True
                        break
                    else:
                        line=' '.join(line.split()[:600])
                        lines.append(line.strip())
                        bar.update()
                
                inputs = tokenizer(handler.process(lines)[0], return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                if getattr(model.config, 'n_positions', None):
                    max_length = model.config.n_positions
                elif getattr(model.config, 'max_position_embeddings', None):
                    max_length = model.config.max_position_embeddings
                else:
                    max_length = 512

                # different max-sentences may have different max_length, and result in different evaluation results    
                max_length =  min(max_length, int(args.max_length_a * input_ids.shape[-1]) + args.max_length_b)
                beam_size = args.beam_size
                # print(args.beam_size)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if 'gsm8k' not in args.input:
                        # if beam_size==1:
                        #     outputs, _, _, _=greedy_search(model, input_ids, device, args, config)

                        # elif beam_size>1:
                        #     outputs, _, _ = beam_search(model, input_ids, device, args, config, beam_size=beam_size)
                        #     outputs =outputs[::beam_size]
                        outputs = model.generate(
                            input_ids,
                            attention_mask=inputs['attention_mask'].to(device),
                            do_sample=False,
                            num_beams=args.beam_size,
                            num_return_sequences=1,
                            # no_repeat_ngram_size=2,
                            max_length=max_length,
                        )
                    elif 'gsm8k' in args.input:
                        outputs = model.generate(
                            input_ids,
                            attention_mask=inputs['attention_mask'].to(device),
                            do_sample=False,
                            num_beams=args.beam_size,
                            num_return_sequences=1,
                            # no_repeat_ngram_size=2,
                            max_length=max_length,
                        )
                if outputs.shape[0] >0:
                    for i in range(outputs.shape[0]):
                        gen=tokenizer.decode(outputs[i], skip_special_tokens=True)
                        ofp.write(gen + '\n')
                    ofp.flush()
                del inputs, input_ids
                torch.cuda.empty_cache()
    except Exception as e:
        print(line)
        print(gen)
        print(e)
    
    return args.output

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda")
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name).to(device)
    
    if model.config.pad_token_id == tokenizer.vocab["<extra_id_1>"]:
            setattr(
                model.config,
                "pad_token_id",
                model.config.decoder_start_token_id,
            )
    decoding(args, model, tokenizer, device)