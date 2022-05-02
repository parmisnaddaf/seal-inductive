





echo "this code is used to run LL model with diff arch"


## declare an array variable
declare -a dataset_list=( "LLGF_cora"  "LLGF_ACM"  "LLGF_DBLP" "LLGF_IMDB")
declare -a labling=("DRNL")
for i in "${dataset_list[@]}"
do
	for la in "${labling[@]}"
	do

                           	echo "$i"
                           	echo "$la"

                           	# or do whatever with individual element of the array
                           	#nohup python -u VGAE_FrameWork.2.1.py  -dataset "$i" -decoder_type "$k" -encoder_type "$en" -NofRels $j >  res/"${i} ${en} ${k} ${j}"
				nohup python -u seal_link_pred.py  --num_hops 1 --use_feature  --runs 1 --model DGCNN --node_label "$la" --dataset "$i">  ReportedResultOnLLGFdatasets/"DGGCN ${i} ${la}-1hop-32"
          			rm -r  datasets_LLGF_r/
				rm -r results/
	done
done
