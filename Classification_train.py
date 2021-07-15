import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from scipy.fftpack import fft,ifft

from sklearn import svm
import pickle
from scipy.signal import argrelmax
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#BWZ Carb Input (grams)

# RawInsulinData = pd.read_csv('InsulinData.csv', low_memory=False)
# RawCGMData = pd.read_csv('CGMData.csv', low_memory=False)
# RawPatientInsulin = pd.read_csv('Insulin_patient2.csv', low_memory=False)
# RawPatientCGM = pd.read_csv('CGM_patient2.csv', low_memory=False)

# def format_insulin(df):

#     df['Date and Time'] = df['Date'] + "-" + df['Time']
#     df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%m/%d/%Y-%H:%M:%S')
#     df = df[df['Date and Time'].notnull()]
#     InsulinData = df[df['BWZ Carb Input (grams)'].notnull() & (df['BWZ Carb Input (grams)'] != 0)]
#     a =InsulinData.copy()
#     return a

# def format_CGM(df):

#     b = df[df['Sensor Glucose (mg/dL)'].notnull()].copy()
#     b['Date and Time'] = b['Date'] + "-" + b['Time']
#     b['Date and Time'] = pd.to_datetime(b['Date and Time'],format='%m/%d/%Y-%H:%M:%S')
#     b = b[b['Date and Time'].notnull()]
#     return b

# def format_patient_insulin(df):
#     df['Date'] = df['Date'].str.replace(' 00:00:00', '')
#     df['Date and Time'] = df['Date'] + "-" + df['Time']
#     df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%Y-%m-%d-%H:%M:%S')
#     InsulinData = df[df['BWZ Carb Input (grams)'].notnull() & (df['BWZ Carb Input (grams)'] != 0)]
#     a = InsulinData.copy()
#     return a

# def format_patient_CGM(df):
#     df['Date'] = df['Date'].str.replace(' 00:00:00', '')
#     b = df[df['Sensor Glucose (mg/dL)'].notnull()].copy()
#     b['Date and Time'] = b['Date'] + "-" + b['Time']
#     b['Date and Time'] = pd.to_datetime(b['Date and Time'], format='%Y-%m-%d-%H:%M:%S')
#     b = b[b['Date and Time'].notnull()]
#     return b

# Insulin_data = format_insulin(RawInsulinData)
# CGM_data = format_CGM(RawCGMData)
# PatientInsulin_data = format_patient_insulin(RawPatientInsulin)
# PatientCGM_data = format_patient_CGM(RawPatientCGM)

# #find_meal_data
# def find_meal_data(Insulin_df, CGM_df):
#     two_hour = pd.to_timedelta('2:00:00', unit='h')
#     half_hour = pd.to_timedelta('00:30:00', unit='h')

#     mealDate = []
#     mealDataSet = []

#     for i in range(len(Insulin_df)-1, -1,-1):
#         if i == 0:
#             mealDate.append(Insulin_df['Date and Time'].iloc[0])
#         elif Insulin_df['Date and Time'].iloc[i] + two_hour < Insulin_df['Date and Time'].iloc[i-1]:
#             mealDate.append(Insulin_df['Date and Time'].iloc[i])

#     for k in mealDate:
#         CGMData = []
#         CGMDataSet = CGM_df.loc[(CGM_df['Date and Time'] > k-half_hour) & (CGM_df['Date and Time'] < k+two_hour)]
#         for l in CGMDataSet['Sensor Glucose (mg/dL)']:
#             CGMData.append(l)

#         mealDataSet.append(CGMData)

#     return mealDataSet

# ## find no meal data

# def find_nomeal_data(RawInsulin_df,CGM_df):
#     two_hour = pd.to_timedelta('2:00:00', unit='h')
#     noMealDataSet = []

#     j = RawInsulin_df['Date and Time'].iloc[-1]

#     while j <= RawInsulin_df['Date and Time'].iloc[0]:
#         noMealData = []
#         dataSet = RawInsulin_df[(RawInsulin_df['Date and Time']>=j) & (RawInsulin_df['Date and Time']<j+two_hour)]
#         if len(dataSet[(dataSet['BWZ Carb Input (grams)'].notnull()) & (dataSet['BWZ Carb Input (grams)'] != 0)]) == 0:
#             start_time = j
#             end_time = j+two_hour
#             CGMDataSet = CGM_df[(CGM_df['Date and Time'] >= start_time) & (CGM_df['Date and Time'] <= end_time)]

#             for x in CGMDataSet['Sensor Glucose (mg/dL)']:
#                 noMealData.append(x)

#             noMealDataSet.append(noMealData)
#             j = end_time

#         else:
#             idx = dataSet[dataSet['BWZ Carb Input (grams)'].notnull() & (dataSet['BWZ Carb Input (grams)'] != 0)].index[0]
#             j = RawInsulin_df['Date and Time'].iloc[idx] + two_hour

#     return noMealDataSet

# meal_mat = find_meal_data(Insulin_data,CGM_data)
# nomeal_mat = find_nomeal_data(RawInsulinData,CGM_data)
# patient_meal_mat = find_meal_data(PatientInsulin_data, PatientCGM_data)
# patient_nomeal_mat = find_nomeal_data(RawPatientInsulin, PatientCGM_data)

# def format_Data(n,m):

#     n = [a[:30] for a in n if len(a) >= 30]
#     m = [b[:24] for b in m if len(b) >= 24]

#     n2 = [x[::-1] for x in n]
#     m2 = [y[::-1] for y in m]
#     return n2,m2

# format_meal_mat,format_nomeal_mat = format_Data(meal_mat,nomeal_mat)
# patient_format_meal_mat, patient_format_nomeal_mat = format_Data(patient_meal_mat,patient_nomeal_mat)
def extract_meal_nomeal(filename1, filename2):
	InsulinData_1 = pd.read_csv(filename1, low_memory=False)
	CGM_1 = pd.read_csv(filename2, low_memory=False)
	CGM_1 = CGM_1[['Time', 'Date', 'Sensor Glucose (mg/dL)']].dropna(how='any')
	# print(CGM.shape)
	new_CGM_1 = InsulinData_1[(InsulinData_1['BWZ Carb Input (grams)'].notnull()) & (InsulinData_1['BWZ Carb Input (grams)'] != 0)]
	new_CGM = new_CGM_1.copy()
	CGM = CGM_1.copy()
	new_CGM['Date'] = new_CGM['Date'].str.replace(' 00:00:00', '')
	CGM['Date'] = CGM['Date'].str.replace(' 00:00:00', '')
	date_time = pd.to_datetime(new_CGM['Date'].apply(str)+' '+new_CGM['Time']) # no drop timestamp_meal
	date_time_copy = date_time.copy().to_frame()

	for i in range(int(new_CGM.shape[0])-1):
		curr_time = date_time.iloc[i]
		last_time = date_time.iloc[i+1]
		diff = (curr_time - last_time).seconds/3600 # time difference--unit: hours
		if diff <= 2:
			date_time_copy.drop(date_time_copy[date_time_copy.iloc[:,0] == last_time].index, inplace=True) # after drop meal timestamp

	twohours = timedelta(hours = 2)
	halfhour = timedelta(hours = 0.5)

	time_start_meal = date_time_copy.iloc[:,0]-halfhour
	time_end_meal = date_time_copy.iloc[:,0]+twohours

	date_time_copy['meal_start'] = time_start_meal
	date_time_copy['meal_end'] = time_end_meal
	date_time_copy = date_time_copy.iloc[:,1:] # meal time interval 
	# print(date_time_copy.shape)
	# add timestamp into CGM
	CGM['Date_Time'] = pd.to_datetime(CGM['Date'].apply(str)+' ' + CGM['Time'])
	# extract meal data (order increasing by time)
	k = 0
	date_time_copy = date_time_copy.iloc[::-1]
	for i in date_time_copy.index:
		time_stamp = date_time_copy.loc[i,:].tolist()
		
		temp_df = CGM[(CGM['Date_Time'] >= time_stamp[0]) & (CGM['Date_Time'] <= time_stamp[1])]
		if temp_df.shape[0] != 30:
			# print('666')
			continue
		sensor = temp_df['Sensor Glucose (mg/dL)'].to_numpy()
		meal_data = sensor if k == 0 else np.vstack((meal_data, sensor))
		k += 1 

	# extract no meal data
	p = 0 
	date_time = date_time.to_frame()
	date_time = date_time.iloc[::-1]
	for q in range(date_time.shape[0]-1):
		curr_ts = date_time.iloc[q].tolist()
		next_ts = date_time.iloc[q+1].tolist()

		if (next_ts[0]-curr_ts[0]) < twohours:
			continue
		j = 0
		while True:
			h = (j+1)*2
			h_end = h + 2
			j += 1
			hour_start = timedelta(hours=h)
			hour_end = timedelta(hours=h_end)

			no_meal_start = curr_ts[0]+ hour_start # one stretch start
			no_meal_end = curr_ts[0] + hour_end # one stretch end
			if no_meal_start > next_ts[0]:
				break # break while loop
		
			temp_no_meal = CGM[(CGM['Date_Time'] >= no_meal_start) & (CGM['Date_Time'] <= no_meal_end)]
			if temp_no_meal.shape[0] != 24:		
				continue
			sensor_no = temp_no_meal['Sensor Glucose (mg/dL)'].to_numpy()
			no_meal_in = sensor_no if j == 1 else np.vstack((no_meal_in, sensor_no))
			
		no_meal_data = no_meal_in if p == 0 else np.vstack((no_meal_data, no_meal_in))
		p += 1
		# output = no_meal_data (order by increasing time)
	return meal_data, no_meal_data
meal_data1, no_meal_data1 = extract_meal_nomeal('InsulinData.csv', 'CGMData.csv')
meal_data2, no_meal_data2 = extract_meal_nomeal('Insulin_patient2.csv', 'CGM_patient2.csv')

meal_data = np.vstack((meal_data1, meal_data2))
meal_data = meal_data[:,6:]
no_meal_data = np.vstack((no_meal_data1, no_meal_data2))
data = np.vstack((meal_data, no_meal_data))

label_train_p = np.ones((meal_data.shape[0], 1))
label_train_n = np.zeros((no_meal_data.shape[0], 1))
label = np.vstack((label_train_p, label_train_n))
# np.save('meal_data.npy', meal_data)
# np.save('no_meal_data.npy', no_meal_data)

# meal_data = np.load('meal_data.npy')
# no_meal_data = np.load('no_meal_data.npy')

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

# tau_meal = find_tau(meal_data)
# mean_norm_meal = find_mean_norm(meal_data)
# fft_data_meal= fft_data(meal_data)
# slope_meal = differentiate(meal_data)

# meal_feature = np.hstack((tau_meal, mean_norm_meal, fft_data_meal, slope_meal))

# tau_no_meal = find_tau(no_meal_data)
# mean_norm_no_meal = find_mean_norm(no_meal_data)
# fft_data_no_meal= fft_data(no_meal_data)
# slope_no_meal = differentiate(no_meal_data)

# no_meal_feature = np.hstack((tau_no_meal, mean_norm_no_meal, fft_data_no_meal, slope_no_meal))
# # print(np.isnan(meal_feature), np.isnan(no_meal_feature))
# ratio = 0.5
# meal_train = meal_feature[:int(ratio * len(meal_feature)), :]
# meal_test = meal_feature[int(ratio * len(meal_feature)):, :]

# no_meal_train = no_meal_feature[:int(ratio * len(no_meal_feature)), :]
# no_meal_test = no_meal_feature[int(ratio * len(no_meal_feature)):, :]

# train_feature = np.vstack((meal_train, no_meal_train))
# test_feature = np.vstack((meal_test, no_meal_test))

# # df = pd.DataFrame(test_feature)
# # df.to_csv(r'test.csv', index=False, header=False)

# p_label_train = np.ones((meal_train.shape[0], 1))
# n_label_train = np.zeros((no_meal_train.shape[0], 1))
# p_label_test = np.ones((meal_test.shape[0], 1))
# n_label_test = np.zeros((no_meal_test.shape[0], 1))

# label_train = np.vstack((p_label_train, n_label_train))
# label_test = np.vstack((p_label_test, n_label_test))

#train part
clf = svm.SVC(kernel='rbf', gamma='auto')
clf.fit(data, label.ravel())
pickle.dump(clf, open('model', 'wb'))

