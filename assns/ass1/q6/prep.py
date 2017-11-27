import numpy as np
import sys
from modshogun import LMNN, RealFeatures, MulticlassLabels
from sklearn.datasets import load_svmlight_file
from matplotlib import pyplot
def main():
    # Get training file name from the command line
    traindatafile = sys.argv[1]

        # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile);
    print("loaded data")
    Xtr = tr_data[0].toarray(); # Converts sparse matrices to dense
    Ytr = tr_data[1]; # The trainig labels
    # Cast data to Shogun format to work with LMNN
    dists = np.ndarray(len(Xtr))
    meanpoint = np.average(Xtr,axis=0)
    print("meanpoint =",  meanpoint)
    sub = Xtr - meanpoint
    for j in range(len(sub)):
                dists[j] = np.dot(sub[j],sub[j])
    sort  = np.argsort(dists)
    Xtrain = np.ndarray(shape=(3000,np.shape(Xtr)[1]))
    Ytrain = np.ndarray(3000)
    for j in range(3000):
                Xtrain[j] = Xtr[sort[j]]
                Ytrain[j] = Ytr[sort[j]]
    Xtr = None
    Ytr = None
    sort = None
    sub = None
    print("free extra data...")
    print("got the points, casting to shogun format...")
    np.save("Xtrain.npy", Xtrain)
    np.save("Ytrain.npy",Ytrain)
if __name__ == '__main__':
    main()

