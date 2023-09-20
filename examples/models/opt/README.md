# OPT 

OPT (Open Pre-trained Transformer) class of language models are released by Meta and are available through [Huggingface](https://huggingface.co/facebook/opt-1.3b)

## Model Variants
OPT has several variants: **opt-125m, opt-350m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b, etc.** User can select any model variant for this example but the upper limit is dependent on the system memory capacity. **For Ryzen-AI, limit to opt1.3b**. 

## Eager Mode
The **eager mode** execution model consists of the sequence of steps enumerated below.

1. Load FP32 model
2. Perform state-of-the-art [SmoothQuant](https://arxiv.org/pdf/2211.10438.pdf) to condition weights before quantization.
3. Quantize the model using PyTorch's dynamic quantization.
   1. Optionally load from checkpoint. 
4. Identify Linear nodes in the model
5. Replace each Linear node with a custom Linear node (QLinear)
6. Initialize weights through Torch's PackedParams
7. Pad, tile and cache weights as initialization step
8. Run inference


## Steps to run the OPT Model with Dynamic Quantization
### Step 1 - Prerequisites and installation
1. Ensure that the project is configured and dependencies are installed. Follow the [prerequisites](../../README.md#prerequisites).
2. Additional installation of driver:
    1. Download the [IPU driver](https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ipu_stack_rel_silicon_2308.zip).
    2. Ensure that your laptop/device can install custom driver by following the steps listed in this [guide](docs/first_time_laptop_setup.md).
    3. Follow installation of the drivers using steps listed in [driver installation readme](docs/manual_install_firmware.md).
    

### Step 2 - Quantize and save model weights
```
python save_weights.py --action save --model_name opt-1.3b
```
This saves 2 sets of weights. A `quantized_<model_name>.pth` and `weights_<model_name>` directory. The `quantized_<model_name>.pth` is loaded into model 

### Step 3 - Option 1 or 2: Using run.py
This script gives option to do the following:
* Benchmark code - measure time/token latency
* Calculate perplexity score
* Decode a set of prompts to show model liveliness

```python
python run.py --help
usage: run.py [-h] [--model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}] [--target {cpu,aie}] [--quant_mode {none,ptdq}] [--smoothquant] [--perplexity] [--benchmark]
              [--dataset {non-raw,raw}] [--load]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}
                        Different OPT model sizes
  --target {cpu,aie}    cpu, aie
  --quant_mode {none,ptdq}
                        Quantization mode - none, dynamic
  --smoothquant         Enable smoothquant
  --perplexity          Calculate perplexity on wikitext2 dataset
  --benchmark           measure token-time using wikitext2 dataset
  --dataset {non-raw,raw}
                        wikitext2-raw-v1, wikitext2-v1
  --load                Load quantized weights from checkpoint. Currently only supported for accelerated target=aie smoothquant enabled
```
Each run generates a log directory `log_<model_name>` and all logs are within this directory. 

#### Decode prompts
```
python run.py --model_name opt-1.3b --load --smoothquant --quant_mode ptdq --target aie 
python run.py --model_name opt-1.3b --load --smoothquant --quant_mode ptdq --target cpu 
python run.py --model_name opt-1.3b --smoothquant --quant_mode none --target cpu
```
These are few samples of responses from OPT1.3B for 30 tokens
```
********************
Prompt: What is the meaning of life?
Response:
The meaning of life is a question that has been asked by many people throughout history. The answer to

********************
Prompt: What does Xilinx do?
Response: 

Xilinx is a global technology company that designs and delivers advanced semiconductor solutions. Our products are
```

#### Perplexity on wikitext2-raw dataset
```
python run.py --model_name opt-1.3b --load --target aie --quant_mode ptdq --perplexity
python run.py --model_name opt-1.3b --load --target cpu --quant_mode ptdq --perplexity
python run.py --model_name opt-1.3b --target cpu --quant_mode none --perplexity
```


#### Profiling - Benchmark
benchmark is done same as GPTQ. 
```
python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --benchmark --load
python run.py --model_name opt-1.3b --target cpu --quant_mode ptdq --benchmark --load
python run.py --model_name opt-1.3b --target cpu --quant_mode none --benchmark
```


### Step 3 - Option 2 or 2: Using opt_demo.py
This script gives user option to run the model on any set or prompts with 3 search strategies
```
python opt_demo.py --help
usage: opt_demo.py [-h] [--model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}] [--target {cpu,aie}] [--quant_mode {none,ptdq}] [--load]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}
                        Different OPT model sizes
  --target {cpu,aie}    cpu, aie
  --quant_mode {none,ptdq}
                        Quantization mode - none, or smoothquant+pytorch dynamic-quant
  --load                Load quantized weights from checkpoint
```

**The script asks for search strategy, to get deterministic answers, always use greedy search.**

The demo gives user flexibility to provide any prompts, with different search options and output token lengths.
Three search options are provided: **greedy, stochastic and contrastive**. These search options provide different level of quality to text generation process. User can modify the strengths of parameters in this file. 

**This feature is available without--load option only.**

This is optional, to see individual tokens as they print to the screen in greedy search mode, open the 

```installationfolder\anaconda3\envs\ryzenai-transformers\lib\site-packages\transformers\generation\utils.py```

In ```def greedy_search(...)``` function,  

after ```next_tokens = torch.argmax(next_tokens_scores, dim=-1)```, 

add this new line: ```print(self.tokenizer.decode(next_tokens)) ```

This prints each new token to the screen as the text-generation process unfolds. 

```
python opt_demo.py  --quant_mode ptdq --target aie --load
python opt_demo.py  --quant_mode ptdq --target cpu --load
python opt_demo.py  --quant_mode none --target cpu
```

Here are examples for 3 search options for the same prompt and token length on AIE with SmoothQuant + Pytorch Dynamic Quantization:

```
   ********************
   Enter prompt or 'exit': San Francisco is a city of
   Enter response length (1-1000): 30
   Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 0
   Setting search to:  Greedy search
   San Francisco is a city of contrasts. It’s a city of the arts, of the food, of the people, of the history
   ********************
   Enter prompt or 'exit': San Francisco is a city of
   Enter response length (1-1000): 30
   Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 1
   Setting search to:  Stochastic search
   San Francisco is a city of incredible contrasts. It has the highest concentration of Jews of any city in the world, and it is known as a
   ********************
   Enter prompt or 'exit': San Francisco is a city of
   Enter response length (1-1000): 30
   Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 2
   Setting search to:  Contrastive search
   San Francisco is a city of many cultures.
   
   The city has a long history of immigration and is home to the largest number of immigrants in
   ********************
```

### Benchmark
A benchmarking script is provided to analyze input prompt length vs time/token of OPT models.
```
python opt_benchmark.py
``` 
