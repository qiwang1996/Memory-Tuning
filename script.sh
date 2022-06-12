source activate qw

cd  Sentence-Level

#an example for  runing on CB dataset  
python3 main.py  --mode 'ft'  --train-file './data/CB/train.jsonl' --val-file './data/CB/val.jsonl'  --test-file './data/CB/test.jsonl'   --saved-file  './predict/ft/CB.jsonl'   --task 'CB'  --lr 2e-5   --batch-size 16   --max-length 300   
python3 main.py  --mode 'bitfit'  --train-file './data/CB/train.jsonl' --val-file './data/CB/val.jsonl'  --test-file './data/CB/test.jsonl'   --saved-file  './predict/bitfit/CB.jsonl'   --task 'CB'    --lr 1e-4   --batch-size 16     --max-length 300  
python3 main.py  --mode 'prefix'  --train-file './data/CB/train.jsonl' --val-file './data/CB/val.jsonl'  --test-file './data/CB/test.jsonl'   --saved-file  './predict/prefix/CB.jsonl'   --task 'CB'  --preseqlen 32    --lr 1e-4   --batch-size 16  --max-length 300   
python3 main.py  --mode 'adapter'  --train-file './data/CB/train.jsonl' --val-file './data/CB/val.jsonl'  --test-file './data/CB/test.jsonl'   --saved-file  './predict/adapter/CB.jsonl'   --task 'CB'    --random-seed 62  --lr 1e-4   --batch-size 16     --max-length 300  --ffn-module-size 32   
python3 main.py  --mode 'memory-ffn'  --train-file './data/CB/train.jsonl' --val-file './data/CB/val.jsonl'  --test-file './data/CB/test.jsonl'   --saved-file  './predict/memory-ffn/CB.jsonl'   --task 'CB'   --lr 1e-4   --batch-size 16     --max-length 300  --ffn-module-size 32  --memory-num 50    
python3 main.py  --mode 'memory'  --train-file './data/CB/train.jsonl' --val-file './data/CB/val.jsonl'  --test-file './data/CB/test.jsonl'   --saved-file  './predict/memory/CB.jsonl'   --task 'CB'  --lr 1e-4   --batch-size 16     --max-length 300  --ffn-module-size 16  --memory-num 50  --preseqlen 16
python3 main.py  --mode 'mam_adapter'  --train-file './data/CB/train.jsonl' --val-file './data/CB/val.jsonl'  --test-file './data/CB/test.jsonl'   --saved-file  './predict/mam_adapter/CB.jsonl'   --task 'CB'   --lr 1e-4  --max-length 300   --batch-size 16   --preseqlen 16  --ffn-module-size 16   


#an example for  runing on  MRPC dataset 
python3 main.py  --mode 'ft'  --train-file './data/MRPC/train.tsv' --val-file './data/MRPC/dev.tsv'  --test-file './data/MRPC/test.tsv'   --saved-file  './predict/ft/MRPC.tsv'   --task 'MRPC'  --lr 2e-5   --batch-size 16      
python3 main.py  --mode 'bitfit'  --train-file './data/MRPC/train.tsv' --val-file './data/MRPC/dev.tsv'  --test-file './data/MRPC/test.tsv'   --saved-file  './predict/ft/MRPC.tsv'   --task 'MRPC'    --lr 1e-4   --batch-size 16        
python3 main.py  --mode 'prefix'  --train-file './data/MRPC/train.tsv' --val-file './data/MRPC/dev.tsv'  --test-file './data/MRPC/test.tsv'   --saved-file  './predict/ft/MRPC.tsv'   --task 'MRPC'  --preseqlen 32    --lr 1e-4   --batch-size 16  
python3 main.py  --mode 'adapter'   --train-file './data/MRPC/train.tsv' --val-file './data/MRPC/dev.tsv'  --test-file './data/MRPC/test.tsv'   --saved-file  './predict/ft/MRPC.tsv'   --task 'MRPC'    --random-seed 62  --lr 1e-4   --batch-size 16   --ffn-module-size 32   
python3 main.py  --mode 'memory-ffn'    --train-file './data/MRPC/train.tsv' --val-file './data/MRPC/dev.tsv'  --test-file './data/MRPC/test.tsv'   --saved-file  './predict/ft/MRPC.tsv'   --task 'MRPC'    --lr 1e-4   --batch-size 16    --ffn-module-size 32  --memory-num 50    
python3 main.py  --mode 'memory'    --train-file './data/MRPC/train.tsv' --val-file './data/MRPC/dev.tsv'  --test-file './data/MRPC/test.tsv'   --saved-file  './predict/ft/MRPC.tsv'   --task 'MRPC'    --lr 1e-4   --batch-size 16     --ffn-module-size 16  --memory-num 50  --preseqlen 16
python3 main.py  --mode 'mam_adapter'    --train-file './data/MRPC/train.tsv' --val-file './data/MRPC/dev.tsv'  --test-file './data/MRPC/test.tsv'   --saved-file  './predict/ft/MRPC.tsv'   --task 'MRPC'   --lr 1e-4    --batch-size 16   --preseqlen 16  --ffn-module-size 16   






cd Token-Level

python3 main.py --mode 'prefix' --train-file './data/msra/train.txt' --val-file './data/msra/valid.txt'  --test-file './data/msra/test.txt'   --task 'msra'  --preseqlen 32    --lr 2e-4    
python3 main.py --mode 'ft' --train-file './data/msra/train.txt' --val-file './data/msra/valid.txt'  --test-file './data/msra/test.txt'   --task 'msra'  --lr 2e-5   
python3 main.py --mode 'memory' --train-file './data/msra/train.txt' --val-file './data/msra/valid.txt'  --test-file './data/msra/test.txt'    --task 'msra'  --preseqlen 16  --ffn-module-size 16   --lr 1e-4   --memory-num 50
python3 main.py  --mode 'adapter'  --train-file './data/msra/train.txt' --val-file './data/msra/valid.txt'  --test-file './data/msra/test.txt'     --task 'msra'  --ffn-module-size 32  --lr 1e-4   
python3 main.py  --mode 'memory-ffn'  --train-file './data/msra/train.txt' --val-file './data/msra/valid.txt'  --test-file './data/msra/test.txt'   --task 'msra'  --ffn-module-size 32  --lr 1e-4    --memory-num 50
python3 main.py  --mode 'mam_adapter'  --train-file './data/msra/train.txt' --val-file './data/msra/valid.txt'  --test-file './data/msra/test.txt'   --task 'msra'  --ffn-module-size 16  --lr 1e-4   --preseqlen 16
python3 main.py  --mode 'bitfit'  --train-file './data/msra/train.txt' --val-file './data/msra/valid.txt'  --test-file './data/msra/test.txt'   --task 'msra'   --lr 1e-4   