# Llama3 Experiment
1. You can fetch original model weights from [hugging face:https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) or [modelscope:https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct).
 ```bash
git lfs install
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3-8B-Instruct.git
 ```
3. code_train.xlsx for classification trainning, eval.xlsx for classification evalation, valid.xlsx for classification validation.
4. You can fetch training source codes from [Github OpenChatKit Project:https://github.com/togethercomputer/OpenChatKit](https://github.com/togethercomputer/OpenChatKit)
5. Please refer to the training script to adapt it to your hardware configuration. "start_phd_train.sh" is my training script. This script has been adapted for use with the llama model and the GPT-NeoX model.
6. use data_loader to load xls files for training datasets.
7. The trained model can be fetched from [https://huggingface.co/Misery-HaHa/SkyLlama38Code](https://huggingface.co/Misery-HaHa/SkyLlama38Code)






# GPT-NeoX Experiment
1. You can fetch original model weights from [GPT-NeoX:https://huggingface.co/EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) or [OpenChatKit-GPT-NeoX-20B:https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B)
2. You can fetch training source codes from [Github OpenChatKit Project:https://github.com/togethercomputer/OpenChatKit](https://github.com/togethercomputer/OpenChatKit)
3. Please refer to the training script to adapt it to your hardware configuration. "start_phd_train.sh" is my training script. This script has been adapted for use with the llama model and the GPT-NeoX model.
4. C_wasm_source_code_52000.rar includes 52,000 C language samples. It is from ojclone datasets. You can fetch it from [https://github.com/clonebench/BigCloneBench](https://github.com/clonebench/BigCloneBench)
5. use run_c2wasm.sh to compile C source codes
```shell
#!/bin/bash
success=0
total=0
err_num=0
for file in *.c; do
	echo "------------compile wasm from c/c++ file: $file-----------------"
	emcc -Oz -ferror-limit=1 -s WASM=1 -s SIDE_MODULE=1 -s USE_BOOST_HEADERS=0 -s ASSERTIONS=0 -g0 -Wmain-return-type -Wreturn-type -Werror,-Wimplicit-function-declaration,-Wunknown-warning-option,-Wimplicit-function-declaration,-Wdeprecated -o $(basename "$file").wasm $file
	#wasm2wat -o $(basename "$file").wast $(basename "$file").wasm
	#echo --------compiler wasm---------------------
	#echo wasmtime $(basename "$file").wasm
	ret=$?
	total=$(expr $total + 1 )
	echo "compile result: $ret"
	if [ $ret -eq 0 ]; then
		#echo "success:$success,$total. fail:$err_num"
		success=$(expr $success + 1)
		echo "success:$success,$total. fail:$err_num"
	else
		err_num=$(expr $err_num + 1)
		echo "error files count : $err_num"
	fi
done
echo "error=$err_num, success=$success, total=$total" > log.txt
ls -l | grep .c.wasm | wc -l
```
6. use data_loader to load wasm files for training datasets byte by byte as token index. don't token the instructions.
7. Adjust the batch size and the number of layers of the model network loaded by each GPU, and then start the training. The startup script can be referred to as start_phd_train.sh

