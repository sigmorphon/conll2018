from sys import argv

from baseline_task2 import eval
from data import read_dataset

if __name__=="__main__":
    sysdata,_,_,_,_ = read_dataset(argv[1])
    golddata,_,_,_,_ = read_dataset(argv[2])
    acc,lev = eval([sysdata,golddata],id2char={},generating=0)
    print("Accuracy: %.2f" % (100*acc))
    print("Avg. Levenshtein distance: %.2f" % (lev))
