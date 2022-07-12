 #!/bin/bash
python step5_attack_target.py -d Electricity -s 2210 -m RF -a BRNV -e 0.1 -n 1 -t Positive --demo 15

python step5_attack_target.py -d Electricity -s 2210 -m RF -a BRS -e 0.1 -n 1 -t Negative --demo 15

python step5_attack_target.py -d Electricity -s 2210 -m RF -a FULLVITA -e 0.1 -n 1 -t Positive --demo 10

python step5_attack_target.py -d Electricity -s 2210 -m CNN -a NVITA -e 0.1 -n 1 -t Positive --demo 10

python step5_attack_target.py -d Electricity -s 2210 -m LSTM -a NVITA -e 0.1 -n 3 -t Negative --demo 10

python step5_attack_target.py -d Electricity -s 2210 -m GRU -a FGSM -e 0.1 -n 1 -t Positive --demo 10

python step5_attack_target.py -d Electricity -s 2210 -m LSTM -a BIM -e 0.1 -n 3 -t Negative --demo 10

 # a for loop for each dataset