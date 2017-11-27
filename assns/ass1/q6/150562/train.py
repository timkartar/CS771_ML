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
    init_transform = np.eye(tr_data[0].toarray().shape[1])
    print(init_transform)
    Xtr = tr_data[0][:6000].toarray(); # Converts sparse matrices to dense
    Ytr = tr_data[1][:6000]; # The trainig labels
    # Cast data to Shogun format to work with LMNN
    features = RealFeatures(Xtr.T)
    labels = MulticlassLabels(Ytr.astype(np.float64))

    ### Do magic stuff here to learn the best metric you can ###

    # Number of target neighbours per example - tune this using validation
    k = 21

    # Initialize the LMNN package
    print("starting lmnn train....")
    lmnn = LMNN(features, labels, k)


    # Choose an appropriate timeout
    lmnn.set_maxiter(3000)
    lmnn.train(init_transform)
    # Let LMNN do its magic and return a linear transformation
    # corresponding to the Mahalanobis metric it has learnt
    L = lmnn.get_linear_transform()
    M = np.matrix(np.dot(L.T, L))
    print(M)
    # Save the model for use in testing phase
	# Warning: do not change this file name
    statistics = lmnn.get_statistics()
    pyplot.plot(statistics.obj.get())
    pyplot.grid(True)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('LMNN objective')
    pyplot.show()
    np.save("model.npy", M) 

if __name__ == '__main__':
    main()
