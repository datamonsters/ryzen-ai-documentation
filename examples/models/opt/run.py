import torch
import logging 
import time 
import argparse 
from transformers import set_seed
from transformers import AutoTokenizer, OPTForCausalLM
import os 

import qlinear 

from utils import Utils
from model_utils import (
    decode_prompts,
    benchmark,
    get_wikitext2,
    perplexity
)

import gc 
import smooth
import numpy as np 

set_seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Different OPT model sizes", type=str, default="opt-1.3b", choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b"])
    parser.add_argument("--target", help="cpu, aie", type=str, default="aie", choices=["cpu", "aie"])
    parser.add_argument('--quant_mode', help="Quantization mode - none, dynamic", type=str, default="ptdq", choices=["none", "ptdq"])
    parser.add_argument('--smoothquant', help="Enable smoothquant", action='store_true')
    parser.add_argument('--perplexity', help="Calculate perplexity on wikitext2 dataset", action='store_true')
    parser.add_argument('--benchmark', help="measure token-time using wikitext2 dataset", action='store_true')
    parser.add_argument('--dataset', help="wikitext2-raw-v1, wikitext2-v1", type=str, default="non-raw", choices=["non-raw", "raw"])
    parser.add_argument('--load', help="Load quantized weights from checkpoint. Currently only supported for accelerated target=aie smoothquant enabled", action='store_true')
    args = parser.parse_args()
    print(f"{args}")

    log_dir = "./logs_%s"%args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s_%s_%s.log"%(args.model_name, args.target, args.quant_mode)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/" +  args.model_name)
    
    if args.load:
        if (args.quant_mode == "ptdq"):
            model = torch.load("./quantized_%s.pth"%args.model_name)
            model.eval()
            print("SmoothQuant is always enabled in this mode ...")
            if (args.target == "aie") :
                node_args = ()
                node_kwargs = {}
                Utils.replace_node( model, 
                                    torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                                    qlinear.QLinear, 
                                    node_args, node_kwargs 
                            )
            else: #(args.target == "cpu"):
                pass
        else:
            print("Mode not supported: if target=cpu, use without --load")
            raise SystemExit
    else:                
        if (args.quant_mode == "none"):
            if (args.target == "aie"):    
                print("Mode not supported - only quantized models can run on AIE")
                raise SystemExit
            else:
                model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)
                if (args.smoothquant == True):
                    act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%args.model_name)
                    smooth.smooth_lm(model, act_scales, 0.5)
                    print(f"SmoothQuant enabled ...")
                model.eval()       
                
        else:
            model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)
            print(model)
            if (args.smoothquant == True):
                act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%args.model_name)
                smooth.smooth_lm(model, act_scales, 0.5)
                print(f"SmoothQuant enabled ...")
            model.eval()
            torch.ao.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True )
            
            collected = gc.collect()

            if (args.target == "aie"):    
                Utils.replace_node( model, 
                                    torch.ao.nn.quantized.dynamic.modules.linear.Linear, 
                                    qlinear.QLinear, 
                                    (), {} )
                                
            else: #(args.target == "cpu"):
                pass 

    print(model)

    collected = gc.collect()

    if (args.perplexity == True): 
        start = time.time()
        perplexity(model, tokenizer, dataset=args.dataset)
        print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")
    
    if (args.benchmark ==True):
        trainloader, testenc = get_wikitext2(tokenizer, nsamples=128, seqlen=2048)
        seqlens =  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        for seqlen in seqlens:
            input_ids = next(iter(trainloader))[0][:, :seqlen]
            tpt, latency = benchmark(model, input_ids)
            print(f"Model:{args.model_name}  Input-Prompt-Tength:{seqlen}  Benchmark(Time/token):{tpt*1000.0} ms")
    
    if (args.benchmark==False) and (args.perplexity==False):
        decode_prompts(model, tokenizer)
        