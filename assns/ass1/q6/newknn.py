import sys
import numpy as np
from sklearn.datasets import load_svmlight_file
def main():
	m = np.load('model.npy')
	# Get training file name from the command line
	traindatafile = sys.argv[1]
	testdata = sys.argv[2]# The training file is in libSVM format
	tr_data = load_svmlight_file(traindatafile);
	test_data = load_svmlight_file(testdata);
	Xtr = tr_data[0].toarray(); # Converts sparse matrices to dense
	Ytr = tr_data[1]; # The trainig labels
	Xtest = test_data[0].toarray();
	Ytest_act = test_data[1];
	#print(Xtr[0])

	# Cast data to Shogun format to work with LMNN
	#features = RealFeatures(Xtr.T)
	#labels = MulticlassLabels(Ytr.astype(np.float64))

	### Do magic stuff here to learn the best metric you can ###

	# Number of target neighbours per example - tune this using validation
	klist = [11] #[1,2,3,5,10]
	Ytest = np.ndarray(shape = (len(Xtest),len(klist)))
	for i in range(len(Xtest)):
		neighbours = []
		#data = np.vstack(i)
		#print(i)
		# construct a kd-tree
		sub = Xtr - Xtest[i]
		dists = np.ndarray(len(Xtr))
		for j in range(len(sub)):
			dists[j] = np.dot(sub[j],sub[j])
		sort  = np.argsort(dists)
		for k in range(len(klist)):
			print("processing " + str(i) + "th  test point for k = " + str(klist[k]))
			closest = sort[:klist[k]]
			neigh_labels = [Ytr[j] for j in closest]
			Ytest[i][k] = max(set(neigh_labels), key=neigh_labels.count)
	# find k nearest neighbors for each element of data, squeezing out the zero result (the first nearest neighbor is always itself)
			# apply an index filter on data to get the nearest neighbor elements
			#neighbours.append(closest)
	Ycheck = np.ndarray(shape = (len(klist),len(Ytest)))
	accuracy = []
	for i in range(len(klist)):
		Ycheck[i] = (Ytest.T[i] == Ytest_act)
		#print(Ycheck.size,Ytest.T.size,Ytest_act.size)
		accuracy.append(sum(Ycheck[i])/len(Ycheck[i]))
	print(accuracy)
		
			
	
if __name__ == '__main__':
	main()
 
