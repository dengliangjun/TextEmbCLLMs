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
