<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Ryzen AI Software Platform </h1>
    </td>
 </table>

## Resnet50 example

### Step 1: Prerequisites
1. Ensure that the project is configured and dependencies are installed. Follow the [prerequisites](../../README.md#prerequisites)
2. The example requires a couple of additional packages. Run the following command to install them: <br>
```python -m pip install -r requirements.txt```

### Step 2: Prepare the Model and the Data
In this example, the ResNet-50 model from PyTorch Hub is utilized and trained using the CIFAR-10 dataset.

The `prepare_model_data.py` script downloads the ResNet-50 model from the PyTorch Hub. 
The script also downloads the CIFAR10 dataset and uses it to retrain the model using the transfer learning technique. 
The training process runs over 500 images for each epoch up to five epochs. 
The training process takes approximately 30 minutes to complete. At the end of the training, 
the trained model is used for the subsequent steps.

Run the following command to start the training: <br>
```python prepare_model_data.py --num_epochs 5```
### Step 3: Quantize the Model

Quantizing AI models from floating-point to 8-bit integers reduces computational power and the memory footprint required for inference. 
For model quantization, you can either use Vitis AI quantizer or Microsoft Olive. 
This example utilizes the Vitis AI ONNX quantizer workflow. 
Quantization tool takes the pre-trained float32 model from the previous step (`resnet_trained_for_cifar10.onnx`) and produces a quantized model. <br>
```python resnet_quantize.py```

### Step 4: Deploy the Model
The `predict.py` script is used to deploy the model. It extracts the first ten images from the CIFAR-10 test dataset and converts them to the `.png` format. 
The script then reads all those ten images and classifies them by running the quantized ResNet-50 model on CPU or IPU.

- Deploy the Model on the CPU: <br>
```python predict.py```
- Deploy the Model on the Ryzen AI IPU: <br>
```python predict.py --ep ipu```
    > **Note**: <br>
    Ensure that the `XLNX_VART_FIRMWARE` environment variable is correctly pointing to the XCLBIN file included in the ONNX Vitis AI Execution Provider package. For more information, see the [installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html#runtime-ipu-binary-selection). <br> 
    Copy the `vaip_config.json` runtime configuration file from the Vitis AI Execution Provider package to the current directory. 
      For more information, see the [installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html#install-vitis-ai-execution-provider). The `vaip_config.json` is used by the `predict.py` script to configure the Vitis AI Execution Provider.