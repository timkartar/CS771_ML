import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math

def main():
	# Get training file name from the command line
	traindatafile = sys.argv[1];
	# For how many iterations do we wish to execute GD?
	n_iter = int(sys.argv[2]);
	# After how many iterations do we want to timestamp?
	spacing = int(sys.argv[3]);

	# The training file is in libSVM format
	tr_data = load_svmlight_file(traindatafile);
	Xtr = tr_data[0]; # Training features in sparse format
	Ytr = tr_data[1]; # Training labels
	# We have n data points each in d-dimensions
	n, d = Xtr.get_shape();
	#n = 100
	# The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
	Ytr = 2*(Ytr - 1.5);
	Ytr = Ytr.astype(int);

	# Optional: densify the features matrix.
	# Warning: will slow down computations
	#Xtr = Xtr.toarray();

	# Initialize model
	# For primal GD, you only need to maintain w
	# Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
	#w = csr_matrix((1, d));
	#print(w[0][0])
	w = np.random.rand(d)
	#w=np.ones(d)
	#print(w)
	# We will take a timestamp after every "spacing" iterations
	time_elapsed = np.zeros(math.ceil(n_iter/spacing));
	tick_vals = np.zeros(math.ceil(n_iter/spacing));
	obj_val = np.zeros(math.ceil(n_iter/spacing));

	tick = 0;

	ttot = 0.0;
	t_start = datetime.now();
	#Ytr = Ytr.reshape(n,1);
	# print(Xtr.shape, Ytr.shape, w.shape, (w.T).shape);
	for t in range(n_iter):
		### Doing primal GD ###
		# Compute gradient
		# yWXtr = np.multiply(Ytr.T,Xtr.dot(w.T));
		# losses = 1 - yWXtr;
		#print(yWXtr.shape,yWXtr,losses)
		# ytemp = np.multiply(Ytr,yWXtr < 1);
		losses = 1 - np.multiply(Ytr.T, Xtr.dot(w.T));
		A = np.maximum(0,losses);
		A = np.sign(A);
		Z = np.multiply(Ytr,A)*Xtr;
		# break
		#print(ytemp)	
		# Z = Xtr.copy();
		# Z.data *= ytemp.repeat(np.diff(Z.indptr))
		#print(Z.sum(axis=0).T.shape)
		#g = np.ndarray((54,1))
		g = np.subtract(w.T,Z)
		# g = np.squeeze(np.asarray(g));
		#g.reshape(1,d); # Reshaping since model is a row vector

		# Calculate step lenght. Step length may depend on n and t
		# eta = (n/10000) * 1/(math.sqrt(t) * 10 + 1);
		eta = 50/n * 1/(math.sqrt(t**0.9) + 1);

		# Update the model
		#print(g)
		#print(w)
		#print(g.shape,w.shape,eta)
		
		#w = np.subtract(w,(eta * g));
		#w = w.T;
		
		#w.reshape((54,));
		#print(w)
		#print(g.shape,w.shape,eta)
		#print("w",w);
		# Use the averaged model if that works better (see [\textbf{SSBD}] section 14.3)
		### Calculating wbar
		wbar = (w*tick + np.subtract(w,eta*g))/(tick+1); 
		w = np.subtract(w,eta*g)
		w = np.squeeze(np.asarray(w))
		wbar = np.squeeze(np.asarray(wbar))
		# Take a snapshot after every few iterations
		# Take snapshots after every spacing = 5 or 10 GD iterations since they are slow

		#if t%spacing == 0:
		# Stop the timer - we want to take a snapshot
		#	t_now = datetime.now();
		#	delta = t_now - t_start;
		#	time_elapsed[tick] = ttot + delta.total_seconds();
		#	ttot = time_elapsed[tick];
		#	tick_vals[tick] = tick;
		#	losses = -np.sum(np.add(-1,yWXtr)[yWXtr < 1]);
		#	obj_val[tick] = 0.5 * wbar.T.dot(wbar) + np.sum(np.maximum(0,losses));# + np.sum(1 - np.min(1,Ytr.dot(Xtr.dot(w.T)))); # Calculate the objective value f(w) for the current model w^t or the current averaged model \bar{w}^t
		#	print(time_elapsed[tick],obj_val[tick]);
		#	tick = tick+1;
			# Start the timer again - training time!
		#	t_start = datetime.now();
			#break;
		# Choose one of the two based on whichever works better for you
	w_final = wbar;#.toarray();
	np.save("model_GD.npy", w_final);
	#plt.plot(time_elapsed,obj_val);
	#plt.show();
if __name__ == '__main__':
	main()
