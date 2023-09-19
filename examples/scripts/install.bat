REM Define the path to the config.env file in the scripts directory
set "config_file=scripts\config.env"

REM Check if the config.env file exists
if not exist "%config_file%" (
    echo "config.env" file not found in the 'scripts' directory. Please make sure it exists.
    exit /b 1
)

REM Read and process the config.env file
for /f "tokens=1,* delims=:" %%a in ('type "%config_file%"') do (
    set "config_%%a=%%b"
)

REM Display the variables
echo "Variables read from config.env:"
for /f "tokens=2 delims==" %%v in ('set config_') do (
    echo "%%v"
)

conda create --name ryzen-ai-examples python=3.9
conda install -n ryzen-ai-examples -c conda-forge nodejs zlib re2
conda activate ryzen-ai-examples
pip install %VAI_Q_ONNX_PATH%
pip install olive-ai[cpu]
pip install pydantic==1.10.9
pip install onnxruntime
pip install %VOE_PATH%