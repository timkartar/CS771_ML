import sys
import numpy as np
from sklearn.datasets import load_svmlight_file
def main():

	# Get training file name from the command line
	traindatafile = sys.argv[1]
	data = load_svmlight_file(traindatafile);
	Xdata = data[0].toarray(); # Converts sparse matrices to dense
	ydata = data[1]; # The trainig labels
	X_train = Xdata[:48000]
	X_val = Xdata[48000:]
	y_train = ydata[:48000]
	y_val = ydata[48000:]
	

	
	klist = range(15,25) 
	y_predicted = np.ndarray(shape = (len(X_val),len(klist)))
	for i in range(len(X_val)):
		neighbours = []
		#data = np.vstack(i)
		#print(i)
		# construct a kd-tree
		#print("processing " + str(i) + "th  test point")
		sub = X_train - X_val[i]
		dists = np.ndarray(len(X_train))
		for j in range(len(sub)):
			dists[j] = np.dot(sub[j],sub[j])
		sort  = np.argsort(dists)
		for k in range(len(klist)):
			print("processing " + str(i) + "th  test point for k = " + str(klist[k]))
			closest = sort[:klist[k]]
			neigh_labels = [y_train[j] for j in closest]
			y_predicted[i][k] = max(set(neigh_labels), key=neigh_labels.count)
	# find k nearest neighbors for each element of data, squeezing out the zero result (the first nearest neighbor is always itself)
			# apply an index filter on data to get the nearest neighbor elements
			#neighbours.append(closest)
	Ycheck = np.ndarray(shape = (len(klist),len(y_predicted)))
	accuracy = []
	for i in range(len(klist)):
		Ycheck[i] = (y_predicted.T[i] == y_val)
		#print(Ycheck.size,y_predicted.T.size,y_predicted_act.size)
		accuracy.append(sum(Ycheck[i])/len(Ycheck[i]))
	print(accuracy)
		
			
	
if __name__ == '__main__':
	main()
 
