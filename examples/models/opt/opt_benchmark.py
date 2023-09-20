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
    benchmark,
    get_wikitext2,
)

import gc 
import numpy as np 
import matplotlib.pyplot as plt 

set_seed(123)

if __name__ == "__main__":

    model_names = ["opt-1.3b"]
    plot_markers = ["*", "o", ">", "v", "8", "s", "+"]
    seqlens =  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    generate_times = {}

    log_file = "log_opt_benchmark.log"
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    
    for model_name in model_names: 
        print("*"*20, model_name)
        model = model = torch.load("./quantized_%s.pth"%model_name)
        model.eval()
        Utils.replace_node( model, 
                            torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                            qlinear.QLinear, 
                            (), {} )
        tokenizer = AutoTokenizer.from_pretrained("facebook/" + model_name)
        trainloader, testenc = get_wikitext2(tokenizer, nsamples=128, seqlen=2048)
        
        gt_seqlen = {} 
        
        for seqlen in seqlens:
            
            input_ids = next(iter(trainloader))[0][:, :seqlen]
            timepertoken, latency = benchmark(model, input_ids)

            gt_seqlen[seqlen] = {'latency':latency, 'timepertoken':timepertoken*1000.}
            #print(f"gt_seqlen: {gt_seqlen}")
            
            print(f"***** {model_name} ***** prompt length: {str(seqlen)} time per token: {timepertoken*1000.}")
            logging.info(f"***** {model_name} ***** prompt length: {str(seqlen)} time per token: {timepertoken*1000.}")
            
                
        generate_times[model_name] = gt_seqlen
        del model 
        del tokenizer 
        gc.collect()
        

    logging.info(f"generate_time: {generate_times}")
    print(f"generate_time: {generate_times}")

    plt.figure()
    i = 0
    for key in generate_times.keys():
        x, y = [], []
        for seqlen in seqlens:
            x.append(seqlen)
            timepertoken = generate_times[key][seqlen]['timepertoken']
            y.append(timepertoken)
        plt.plot(x, y, plot_markers[i]+":", label="%s"%(key))
        i += 1
    plt.xlabel("Input prompt length")
    plt.ylabel("Time/token [ms]")
    plt.ylim(bottom=0)
    plt.grid(which='both', axis="y")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=len(model_names)) 
    plt.title("Benchmark OPT models: input prompt length vs time/token (RyzenAI AIE)")
    plt.savefig("opt_benchmark_aie.png")
    plt.close()
