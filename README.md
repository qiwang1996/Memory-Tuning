# Memory-Tuning
Code for NeuralPS 2022 submission titled "Memory-Tuning: A Unified Parameter-Efficient Tuning Method for Pre-trained Language Models "


## Note  
Our released code is setted for using BERT-Large. If you want to validate results conducted on  BERT-base,  Please revise code as the following intruction:  
Step1: In ffn_trainabl e_module/bert.py  line 120-121,   set  n_layer=12 n_head=12,  n_embd=768, mid_dim=1024   
Step2: In main.py line 77, set hidden_size=768  


## Main experiment and Robustness Analysis  

script.sh supports 7 tuning methods  for training and  inferring on 8 dataset used in paper.
There are  line command examples running on 3  datasets  in given script.sh. You can add the remaining
line commands for other datasets according to those given ones and please note different setup for different dataset in Table 1 of our paper.
 

## Visualization for attention vector  of  memory slots  

Our released code is setted for using BERT-Large but our visualization experiment is conducted  by using BERT-base. So you need to revise several
places in code for visualization experiment as the Note part illustrated.  

Step1: revise code as  the Note part  illustrated   
Step2: Train  BERT-base  on  SST-2 and MRPC  for  MT-M, MT-F, MT-MF.      
Step3:   
Step4: use  predict.py  for inferring  
Step5:  set   --test-file './data/SST-2/dev.tsv'    --test-file './data/MRPC/dev.tsv'  in line command for SST-2 and MRPC respectively.  (Since we experiment on  the dev file)  
Step6: copy the inferred attention data and  pred.txt  into Visualize\MSRA\memory-ffn,  Run compare.txt and then Run t-sne.py  

 
If you have any question,  feel free to connect me by sending an email into 1290220814@qq.com.  :)


