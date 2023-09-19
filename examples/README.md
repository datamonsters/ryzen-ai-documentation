<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Ryzen AI Software Platform </h1>
    </td>
 </table>

## Ryzen-AI examples structure

```
ryzen-ai-documentation/examples
├── models                          # test models  cards with code and all details 
│ ├── resnet50                      # model name
│ │ ├── README.md                   # instructions on how to run the example
│ │ ├── ...                         # other files needed for launching example on Ryzen AI
│ ...  
├── scripts                         # helper scripts folder
│ ├── config.env                    # env-variables configuration of all needed paths for configuring the project 
│ └── install.bat                   # single script for completed installation of the project
└── README.md                       # examples descriptions
```

## Prerequisites

Before running any example, you need to install and configure all necessary project dependencies. 
1. Download the following files from the Ryzen AI Software Platform:
   - [Vitis AI Quantizer](https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=vai_q_onnx-1.15.0-py2.py3-none-any.whl)
   - [Vitis AI Execution Provider](https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=voe-4.0-win-amd64.zip)
2. Install and set up the dependencies. Choose one of the following options: 
   - Follow the installation guide in the [documentation](https://ryzenai.docs.amd.com/en/latest/inst.html#installation-steps). 
   - Set necessary paths in `config.env` and then run the `install.bat` script.
    > **Note**: <br>
    You must specify all paths of downloaded installation files in `config.env` 

## Examples

- [Resnet50](./models/resnet50/README.md#resnet50-example)