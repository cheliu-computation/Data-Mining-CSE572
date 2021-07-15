import pandas as pd
import numpy as np
import os, shutil
# from sklearn.datasets import load_iris
# from sklearn import tree
# import matplotlib.pyplot as plt
# os.chdir('/media/sf_Project_1_Student_Files/backup')
# filename = os.listdir()
CGMData = pd.read_csv('CGMData.csv', low_memory=False)
InsulinData = pd.read_csv('InsulinData.csv', low_memory=False)

# seperating manual and auto
# find the auto mode time stamp
row_index = InsulinData.Alarm[InsulinData.Alarm == 'AUTO MODE ACTIVE PLGM OFF'].index.tolist()
slice_start = row_index[1]

slice_time = InsulinData.iloc[[slice_start]].Time.tolist()
slice_date = InsulinData.iloc[[slice_start]].Date.tolist()

# find the row of automode start in CGM
time_seq = CGMData.Time[CGMData.Date == slice_date[0]].tolist()

#convert str to int for comparing
split_time = slice_time[0].split(':')
for i in range(len(split_time)):
	split_time[i] = int(split_time[i])

def find_slice_time(time_seq):
	last_time = time_seq[0]
	for k in time_seq:
		temp_time = k.split(':')
		for i in range(len(temp_time)):
			temp_time[i] = int(temp_time[i])
		if split_time[0] == temp_time[0] and split_time[1] > temp_time[1]:
		
			return(k)	
			break
		last_time = k

slice_time_CGM = find_slice_time(time_seq)
# print(slice_time_CGM)
row_CGM = CGMData.Time[CGMData.Time == slice_time_CGM].index.tolist()
row_CGM = int(row_CGM[0])

def slicing(row_CGM):
	manual_mode_Data = CGMData.iloc[row_CGM::]
	manual_mode_Data = manual_mode_Data.loc[: ,['Date', 'Time', 'Sensor Glucose (mg/dL)']]
	
	auto_mode_Data = CGMData.iloc[0:row_CGM]
	auto_mode_Data = auto_mode_Data.loc[:, ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
	return manual_mode_Data, auto_mode_Data

manual_mode_Data = slicing(row_CGM)[0]
auto_mode_Data = slicing(row_CGM)[1]


# dayligh: 6am -- 00am overnigh: 00am -- 6am whole day: 12am-12am
new_manual_mode = manual_mode_Data.interpolate(method='akima')
new_auto_mode = auto_mode_Data.interpolate(method='akima')


def day_index(mode):
	count = [0]
	for i in range(mode.shape[0]-1):
		if mode.Date.iloc[i] != mode.Date.iloc[i+1]:
			count.append(i+1) # index_i+1 is the start on next day
	count.append((mode.shape[0]))
	return count

manual_count = np.array(day_index(new_manual_mode))
auto_count = np.array(day_index(new_auto_mode))
def drop_df (count, df):
	newdf = df.head()
	for i in range(len(count)-1):
		if (count[i+1]-count[i])/288 > 0.5:
			temp_df = df.iloc[count[i]: count[i+1]]
			newdf = temp_df if i == 0 else pd.concat([newdf, temp_df])
	return newdf
new_manual_mode_drop = drop_df(manual_count, new_manual_mode)
new_auto_mode_drop = drop_df(auto_count, new_auto_mode)

manual_count_drop = day_index(new_manual_mode)
auto_count_drop = day_index(new_auto_mode)
def set_each_timerange(mode, count):
	metric = []
	for i in range(len(count)-1): # iterating by day
	# for i in range(1):
		set_a =[] # >180
		set_b =[] # > 250
		set_c =[] # >=70 and <=180
		set_d =[] # >=70 <= 150
		set_e =[] # <70
		set_f =[] # <54


		end_time = [int(i) for i in mode.Time.iloc[count[i]].split(':')]
		start_time = [int(i) for i in mode.Time.iloc[count[i+1]-1].split(':')]
		# print(start_time, end_time)

		time_interval = (end_time[0]*3600+end_time[1]*60+end_time[2]) - (start_time[0]*3600+start_time[1]*60+start_time[2])
		# print(time_interval)
		for k in range(count[i], count[i+1]): # by time
			temp_metric =[]
			data = mode['Sensor Glucose (mg/dL)'].iloc[k]
			data = int(data)
			if data > 180:
				set_a.append(data)
			if data > 250:
				set_b.append(data)
			if data >= 70 and data <= 180:
				set_c.append(data)
			if data >=70 and data <= 150:
				set_d.append(data)
			if data < 70:
				set_e.append(data)
			if data < 54:
				set_f.append(data)
		# temp_metric.append((len(set_a))/(count[i+1] - count[i])*100)
		# temp_metric.append((len(set_b))/(count[i+1] - count[i])*100)
		# temp_metric.append((len(set_c))/(count[i+1] - count[i])*100)
		# temp_metric.append((len(set_d))/(count[i+1] - count[i])*100)
		# temp_metric.append((len(set_e))/(count[i+1] - count[i])*100)
		# temp_metric.append((len(set_f))/(count[i+1] - count[i])*100)
		temp_metric.append((len(set_a))/288*100)
		temp_metric.append((len(set_b))/288*100)
		temp_metric.append((len(set_c))/288*100)
		temp_metric.append((len(set_d))/288*100)
		temp_metric.append((len(set_e))/288*100)
		temp_metric.append((len(set_f))/288*100)
		metric = temp_metric if i == 0 else np.vstack((metric, temp_metric))
	# n*6 matrix created, n = days
	
	# metric = metric.mean(axis=0)
	return metric # for whole day

all_day_manual = (set_each_timerange(new_manual_mode_drop, manual_count_drop)).sum(axis=0)/len(manual_count_drop) #  manual mean for whole day
all_day_auto = (set_each_timerange(new_auto_mode_drop, auto_count_drop)).sum(axis=0)/len(auto_count_drop) # auto mean for whole day

def set_day_night_timerange(mode, count):
	day_end_list = []
	for i in range(len(count)-1): # iterating by day
		new_df = mode.iloc[count[i]: count[i+1]-1] # one day dataframe
		num_row = 0
		for t in range(new_df.shape[0]):
			if int(new_df.Time.iloc[t].split(':')[0])>5: 
				num_row += 1
		day_end_list.append(num_row+count[i]) # endpoint of daytime
	return	day_end_list

seg_day_manual = set_day_night_timerange(new_manual_mode_drop, manual_count_drop)
seg_day_auto = set_day_night_timerange(new_auto_mode_drop, auto_count_drop)


def mean_day(mode, count, day_count):
	metric = []
	for i in range(len(count)-1): # iterating by day
		set_a =[] # >180
		set_b =[] # > 250
		set_c =[] # >=70 and <=180
		set_d =[] # >=70 <= 150
		set_e =[] # <70
		set_f =[] # <54
		
		# print(start_time, end_time)
		
		# print(time_interval)
		for k in range(count[i], day_count[i]): # by time
			temp_metric =[]
			data = mode['Sensor Glucose (mg/dL)'].iloc[k]
			data = int(data)
			if data > 180:
				set_a.append(data)
			if data > 250:
				set_b.append(data)
			if data >= 70 and data <= 180:
				set_c.append(data)
			if data >=70 and data <= 150:
				set_d.append(data)
			if data < 70:
				set_e.append(data)
			if data < 54:
				set_f.append(data)
		temp_metric.append(len(set_a)/(day_count[i]-count[i]))
		temp_metric.append(len(set_b)/(day_count[i]-count[i]))
		temp_metric.append(len(set_c)/(day_count[i]-count[i]))
		temp_metric.append(len(set_d)/(day_count[i]-count[i]))
		temp_metric.append(len(set_e)/(day_count[i]-count[i]))
		temp_metric.append(len(set_f)/(day_count[i]-count[i]))
		
		metric = temp_metric if i == 0 else np.vstack((metric, temp_metric))
	# n*6 matrix created, n = days
	
	# metric = metric.mean(axis=0)
	return metric # for whole day
daytime_manual = mean_day(new_manual_mode_drop, manual_count_drop, seg_day_manual).sum(axis=0)/len(manual_count_drop)
# print('-----------------------')
daytime_auto = mean_day(new_auto_mode_drop, auto_count_drop, seg_day_auto).sum(axis=0)/len(auto_count_drop)

def set_night_timerange(mode, count):
	night_end_list = []
	new_count = count.copy()
	for i in range(len(count)-1): # iterating by day
		new_df = mode.iloc[count[i]: count[i+1]-2] # one day dataframe
		num_row = 0
		for t in range(new_df.shape[0]):
			if int(new_df.Time.iloc[t].split(':')[0])>5: 
				num_row += 1
		# print(num_row, count[i+1]-1-count[i])
			if num_row+1 == count[i+1]-1-count[i]:
				del new_count[i]
		night_end_list.append(num_row+count[i]) # endpoint of daytime
	return	night_end_list, new_count

new_manual_count = set_night_timerange(new_manual_mode, manual_count)[1]
seg_night_manual = set_night_timerange(new_manual_mode, manual_count)[0]
new_auto_count = set_night_timerange(new_auto_mode, auto_count)[1]
seg_night_auto = set_night_timerange(new_auto_mode, auto_count)[0]
def mean_night(mode, count, day_count):
	metric = []
	for i in range(len(count)-1): # iterating by day
		set_a =[] # >180
		set_b =[] # > 250
		set_c =[] # >=70 and <=180
		set_d =[] # >=70 <= 150
		set_e =[] # <70
		set_f =[] # <54
		
		for k in range(day_count[i]+1, count[i+1]): # by time
			temp_metric =[]
			data = mode['Sensor Glucose (mg/dL)'].iloc[k]
			data = int(data)
			if data > 180:
				set_a.append(data)
			if data > 250:
				set_b.append(data)
			if data >= 70 and data <= 180:
				set_c.append(data)
			if data >=70 and data <= 150:
				set_d.append(data)
			if data < 70:
				set_e.append(data)
			if data < 54:
				set_f.append(data)
		temp_metric.append((len(set_a))/(count[i+1] - count[i])*100)
		temp_metric.append((len(set_b))/(count[i+1] - count[i])*100)
		temp_metric.append((len(set_c))/(count[i+1] - count[i])*100)
		temp_metric.append((len(set_d))/(count[i+1] - count[i])*100)
		temp_metric.append((len(set_e))/(count[i+1] - count[i])*100)
		temp_metric.append((len(set_f))/(count[i+1] - count[i])*100)

		
		metric = temp_metric if i == 0 else np.vstack((metric, temp_metric))
	# n*6 matrix created, n = days
	
	# metric = metric.mean(axis=0)
	return metric # for whole da
overnight_manual = mean_night(new_manual_mode, new_manual_count, seg_night_manual).sum(axis=0)/len(new_manual_count)

overnight_auto = mean_night(new_auto_mode, new_auto_count, seg_night_auto).sum(axis=0)/len(new_auto_count)


manual = np.concatenate((daytime_manual,overnight_manual, all_day_manual))
auto = np.concatenate((daytime_auto,overnight_auto, all_day_auto))

dataf = np.vstack((manual, auto))


dataf = pd.DataFrame(dataf)

dataf.to_csv('Results.csv', index = False, header=False)