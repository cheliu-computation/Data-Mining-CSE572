import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from scipy.fftpack import fft,ifft

from sklearn import svm
import pickle
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.signal import argrelmax
def find_tau(array):
	count = 0
	for i in  array:
		i = i[5:]
		max_v = max(i)
		index = max(np.where(i == max_v)[0])

		index = int(index)
		tau = index

		tau_seq = tau if count == 0 else np.vstack((tau_seq, tau))
		count += 1
	return tau_seq

def find_mean_norm(array):
	count = 0
	for i in array:
		max_v = max(i)
		mean = i.mean()
		min_v = min(i)
		norm_v = (max_v-mean)/(max_v-min_v)

		norm_seq = norm_v if count == 0 else np.vstack((norm_seq, norm_v))
		count += 1
	return norm_seq

def fft_data(array):
	count = 0
	for i in array:
		y = abs(fft(i)[range(int(len(i)/2))].real)
		new_y = np.copy(y)
		local_max = argrelmax(new_y)[0]
		if len(local_max) > 1:
			local_max = np.delete(local_max, 0)
			peak,freq = [], []
			if len(local_max) == 0:
				peak = [0, 0]
			elif len(local_max) == 1:
				peak.append(new_y[local_max[0]])
				peak.append(0)
			else:
				peak = [new_y[local_max[0]], new_y[local_max[1]]]

			freq1 = np.fft.fftfreq(len(new_y), d=(1/len(new_y)))
			if len(local_max) == 0:
				freq = [0, 0]
			elif len(local_max) == 1:
				freq.append(new_y[local_max[0]])
				freq.append(0)
			else:
				freq = [new_y[local_max[0]], new_y[local_max[1]]]
			
			p_f = [peak[0], freq[0], peak[1], freq[1]]
		else:
			p_f = [0, 0, 0, 0]
		fft_seq = p_f if count == 0 else np.vstack((fft_seq, p_f))
		count += 1
	return fft_seq

def differentiate(array):
	count = 0
	for i in array:
		if len(i) >24:
			temp_i = i[6:]
			max_v = max(temp_i)
			index = int(max(np.where(temp_i==max_v)[0]))
			index = index + 6
			i = i[5:index] # tm to peak

			diff_1 = np.diff(i)
			if len(diff_1) >= 1:
				diff_2 = np.diff(diff_1)
			else:
				diff_2 = 0
			if len(diff_1) < 1:
				diff_1 = 0

			if isinstance(diff_1, int):
				diff_1 = diff_1
			else:
				diff_1 = diff_1.mean()

			if isinstance(diff_2, int):
				diff_2 = diff_2
			elif len(diff_2) < 1:
				diff_2 = 0
			else:
				diff_2 = diff_2.mean()
			diff = [diff_1, diff_2]
			
			slope = diff if count == 0 else np.vstack((slope, diff))
			count += 1
		else:
			temp_i = i
			max_v = max(temp_i)
			index = int(max(np.where(temp_i==max_v)[0]))
			
			i = i[0:index] # tm to peak

			diff_1 = np.diff(i)
			if len(diff_1) >= 1:
				diff_2 = np.diff(diff_1)
			else:
				diff_2 = 0
			if len(diff_1) < 1:
				diff_1 = 0

			if isinstance(diff_1, int):
				diff_1 = diff_1
			else:
				diff_1 = diff_1.mean()

			if isinstance(diff_2, int):
				diff_2 = diff_2
			elif len(diff_2) < 1:
				diff_2 = 0
			else:
				diff_2 = diff_2.mean()
			diff = [diff_1, diff_2]
			
			slope = diff if count == 0 else np.vstack((slope, diff))
			count += 1
	return slope

test = pd.read_csv('test.csv', low_memory=False, header=None)
test = test.to_numpy()

# tau_test = find_tau(test)
# mean_norm_test = find_mean_norm(test)
# fft_data_test= fft_data(test)
# slope_test = differentiate(test)

# test_feature = np.hstack((tau_test, mean_norm_test, fft_data_test, slope_test))

clf = pickle.load(open('model', 'rb'))
result = clf.predict(test)

df = pd.DataFrame(result)
df.to_csv(r'Result.csv', index=False, header=None)