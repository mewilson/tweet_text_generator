import  matplotlib.pyplot as plt
import pickle
import sys
import time

def unpickle_file(pickle_path):
	"""unpickle the file, wtf you think this do??
	
	Arguments:
		pickle_path {string} -- describes which pickle to unpickle
	
	Returns:
		data -- unpickled object
	"""
	try:
		with open(pickle_path, "rb") as f:
			data = pickle.load(f)

	except Exception as e:
		print("ERROR: ENCOUNTERED A PROBLEM TRYING TO UNPICKLE YOUR PICKLE\n\
		 ALSO, PICKLE RIIIIIIICK!!!")

	return data


def data_to_png(raw_data, save_as_fname, metric):
	"""Void Function Creates a PNG file from 
		Pickled Keras Model History
	
	Arguments:
		raw_data {unpickle} -- data obj from unpickle operation
		save_as_fname {string} -- filename to save PNG as
	"""
	plt.plot(raw_data.history[str(metric)])
	plt.title("Model " + str(metric))
	plt.ylabel(str(metric))
	plt.xlabel("Epoch")
	plt.legend("TRAINING & TESING ", loc="upper left")
	plt.savefig("../figs/" + str(save_as_fname) + ".png")
	plt.show()
	print("CHECK figs/ DIR FOR NEW PNG FILE")


if __name__ == "__main__":

	start = time.time()
	if (len(sys.argv) <= 1):
		print("ERROR: PATH OF INPUT PICKLE IS MISSING OR UNSPECIFIED.\nTRY AGAIN.")
		exit()

	#UNPICKLE
	raw_data = unpickle_file(sys.argv[1])

	print("[*] TESTING IF DATA MADE IT OUT ALIVE AND IN ONE PIECE")
	print(raw_data.history.keys())

	data_to_png(raw_data, sys.argv[2], sys.argv[3])

	fin = time.time()
	print("OPERATION : " + str((fin - start)/60) + " minutes to complete")