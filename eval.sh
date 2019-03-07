python3 main.py -te save/model/my_s2s_sent/1-1_300/100000_backup_bidir_model.tar -tf data/acl_multi_testfile.txt -of outfile.txt -sif SIF/acl_multi_test_sif > OUT/acl_100000

python3 bleu.py > OUT/acl_100000_bleu.txt


python3 main.py -te save/model/my_s2s_sent/1-1_300/100000_backup_bidir_model.tar -tf data/my_multi_testfile.txt -of outfile.txt -sif SIF/my_multi_test_sif > OUT/my_100000

python3 bleu.py > OUT/my_100000_bleu.txt

