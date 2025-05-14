import transformers
import torch
import argparse
import tqdm
from torch.cuda.amp import autocast
from utils.template_handler import TemplateHandler, FewshotTemplateHandler
# from utils.helper import greedy_search, beam_search
from utils.dataset import set_seed
from utils.helper import CustomLogitsProcessor, tool_fn

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='data/xsum/test.src')
parser.add_argument('-o', '--output', type=str, default='test.tgt_gen')
parser.add_argument('-bs', '--beam-size', type=int, default=1)
parser.add_argument('-mn', '--model-name', default='/home/gluo/DPO-ST/ft_models/t5/sft-t5-base')
parser.add_argument('-lp', '--length-penalty', default=1.0, type=float)
parser.add_argument('-tn', '--tokenizer-name')
parser.add_argument('--max-sentences', type=int, default=16)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--do-sample', action='store_true')
parser.add_argument('--topk', type=int)
parser.add_argument('--top-p', type=float)
parser.add_argument('--max-length-a', type=float, default=1.5)
parser.add_argument('--max-length-b', type=int, default=50)
parser.add_argument('--min-max-length', type=int, default=8)
parser.add_argument('--max-length', type=int, default=512)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--template', type=str, default='templates/summarization_T0.txt')
parser.add_argument('-cn', '--config-name', type=str, default='/mnt/nvme/guoqing/t5-base/config.json')
parser.add_argument('--length-ratio', type=float, default=2)
parser.add_argument('--max-tokens', type=int, default=1024)
parser.add_argument('--init-flat', action='store_true')
parser.add_argument('--mask-extra', action='store_true')


def decoding(args,model,tokenizer,device=None, logits_proc=None):
    set_seed(args.seed)
    model.config.length_penalty = args.length_penalty
    model.eval()
    finished = False
    bar = tqdm.tqdm()
    if 'gsm8k' in args.input:
        handler = FewshotTemplateHandler(args.template, tokenizer)
    else:
        handler = TemplateHandler(args.template, tokenizer)
    
    gen_kwargs = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    try:
        with open(args.input, encoding='utf8') as fp, \
            open(args.output, 'w', encoding='utf8') as ofp:
            # ofp.write(str(args) + '\n')
            while not finished:
                torch.cuda.empty_cache()
                lines = []

                for _ in range(args.max_sentences):
                    line = fp.readline()
                    if len(line)<=1:
                        finished = True
                        break
                    else:
                        if 'xsum' in args.input:
                            line=' '.join(line.split()[:600])
                        else:
                            line=' '.join(line.split())
                        lines.append(line.strip())
                        bar.update()
                
                # print(handler.process(lines)[0])
                # exit()
                if 'gsm8k' not in args.input:
                    inputs = tokenizer(handler.process(lines)[0], return_tensors='pt', max_length=512, padding=True, truncation=True)
                elif 'gsm8k' in args.input:
                    inputs = tokenizer(lines, return_tensors='pt', max_length=200, padding=True, truncation=True)
                input_ids = inputs['input_ids'].to(device)


                # different max-sentences may have different max_length, and result in different evaluation results    
                max_length = args.max_length

                # print(args.beam_size)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if 'gsm8k' not in args.input:
                        outputs = model.generate(
                            input_ids,
                            attention_mask=inputs['attention_mask'].to(device),
                            do_sample=False,
                            num_beams=args.beam_size,
                            num_return_sequences=1,
                            max_length=max_length,
                        )
                    else:
                        outputs = model.generate(
                            input_ids,
                            attention_mask=inputs['attention_mask'].to(device),
                            do_sample=False,
                            num_beams=args.beam_size,
                            num_return_sequences=1,
                            max_length=max_length,
                            logits_processor=logits_proc,
                            return_dict_in_generate=False,
                            **gen_kwargs,
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
        model_name,
        cache_dir='/home/gluo').to(device)
    
    m = CustomLogitsProcessor(tokenizer=tokenizer, tool=tool_fn)
    logits_proc = [m]
    
    decoding(args, model, tokenizer, device, logits_proc)