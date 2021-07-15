import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
def extract_meal_nomeal(filename1, filename2):
	InsulinData_1 = pd.read_csv(filename1, low_memory=False)
	CGM_1 = pd.read_csv(filename2, low_memory=False)
	CGM_1 = CGM_1[['Time', 'Date', 'Sensor Glucose (mg/dL)']].dropna(how='any')
	
	new_CGM_1 = InsulinData_1[(InsulinData_1['BWZ Carb Input (grams)'].notnull()) & (InsulinData_1['BWZ Carb Input (grams)'] != 0)]
	new_CGM = new_CGM_1.copy()
	
	CGM = CGM_1.copy()
	new_CGM['Date'] = new_CGM['Date'].str.replace(' 00:00:00', '')
	new_CGM['Timestamp'] = pd.to_datetime(new_CGM['Date'].apply(str)+' '+new_CGM['Time'])
	# print(new_CGM['Timestamp'])
	CGM['Date'] = CGM['Date'].str.replace(' 00:00:00', '')
	date_time = pd.to_datetime(new_CGM['Date'].apply(str)+' '+new_CGM['Time']) # insulin timestamp
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
	# datetime copy[timestamp, start, end]
	# print(date_time_copy.head())
	# add timestamp into CGM
	CGM['Date_Time'] = pd.to_datetime(CGM['Date'].apply(str)+' ' + CGM['Time'])
	# extract meal data (order increasing by time)
	k = 0
	date_time_copy = date_time_copy.iloc[::-1]
	for i in date_time_copy.index:
		time_stamp = date_time_copy.loc[i,:].tolist()
		
		temp_df = CGM[(CGM['Date_Time'] >= time_stamp[1]) & (CGM['Date_Time'] <= time_stamp[2])]
		if temp_df.shape[0] != 30:
			# print('666')
			continue
		sensor = temp_df['Sensor Glucose (mg/dL)'].to_numpy()
		
		meal_amount = int(new_CGM[new_CGM['Timestamp']==time_stamp[0]]['BWZ Carb Input (grams)'])
		meal_amount = np.array([meal_amount])

		sensor = np.concatenate((sensor, meal_amount))
		meal_data = sensor if k == 0 else np.vstack((meal_data, sensor))
		k += 1 

	# extract no meal data
	
		# output = no_meal_data (order by increasing time)
	return meal_data
meal_data = extract_meal_nomeal('InsulinData.csv', 'CGMData.csv')
# print(meal_data.shape)


def ground_truth(input):
	zero_twenty = np.array([])
	twenty_fourty = np.array([])
	forty_sixty =np.array([])
	sixty_eighty =np.array([])
	eighty_hun =np.array([])
	hun_end=np.array([])
	for k in range(input.shape[0]):
		i = input[k,:]
		if i[-1] <= 20:
			zero_twenty = i[0:31] if zero_twenty.size == 0 else np.vstack((zero_twenty, i[0:31]))
		elif i[-1]>20 and i[-1] <= 40:
			twenty_fourty = i[0:31] if twenty_fourty.size == 0 else np.vstack((twenty_fourty, i[0:31]))
		elif i[-1]>40 and i[-1] <= 60:
			forty_sixty = i[0:31] if forty_sixty.size == 0 else np.vstack((forty_sixty, i[0:31]))
		elif i[-1]>60 and i[-1] <= 80:
			sixty_eighty = i[0:31] if sixty_eighty.size == 0 else np.vstack((sixty_eighty, i[0:31]))
		elif i[-1]>80 and i[-1] <= 100:
			eighty_hun = i[0:31] if eighty_hun.size == 0 else np.vstack((eighty_hun, i[0:31]))
		else:
			hun_end = i[0:31] if hun_end.size == 0 else np.vstack((hun_end, i[0:31]))
	return (zero_twenty.shape[0]), (twenty_fourty.shape[0]), (forty_sixty.shape[0]), (sixty_eighty.shape[0]), (eighty_hun.shape[0]), (hun_end.shape[0])

def log(a):
	if a == 0:
		an = 0
	else:
		an = -a*np.log10(a)
	return an
# K-means
def kmeans():
	transformer = Normalizer(norm='l2').fit(meal_data)
	meal_data_norm=transformer.transform(meal_data)
	kmeans = KMeans(n_clusters=6, random_state=0).fit(meal_data_norm)
	label_kmean = kmeans.labels_
	entro = []
	purity = []
	for i in range(5):
		kmean_dict = np.where(label_kmean == i)[0]
		j = 0 
		for k in kmean_dict:
			temp_data = meal_data[k,:]
			temp_clus = temp_data if j == 0 else np.vstack((temp_clus, temp_data))
			j += 1
		a,b,c,d,e,f = ground_truth(temp_clus)
		a,b,c,d,e,f = a/(temp_clus.shape[0]),b/(temp_clus.shape[0]),c/(temp_clus.shape[0]),d/(temp_clus.shape[0]),e/(temp_clus.shape[0]),f/(temp_clus.shape[0])
		en = log(a) + log(b) + log(c) + log(d) + log(e) + log(f)
		wei_en = temp_clus.shape[0]/(meal_data_norm.shape[0])*en
		entro.append(wei_en)
		pur = max(np.array([a,b,c,d,e,f]))
		wei_pur = temp_clus.shape[0]/(meal_data_norm.shape[0])*pur
		purity.append(wei_pur)
	return (sum(entro)), (sum(purity)*3), (kmeans.inertia_*2)



#DBSCAN
def dbscan():
	pca = PCA(n_components=3)
	mealT=meal_data.T
	meal_pca = pca.fit(mealT)
	meal_decom = meal_pca.components_.T



	m1 = DBSCAN(eps=0.018, min_samples=2)
	m1.fit(meal_decom)
	indic = (m1.core_sample_indices_)
	label = (m1.labels_)
	num_cluster = max(label)

	final_sse = 0
	for i in range(num_cluster):
		dic_indi = np.where(label == i)[0]
		q = 0 
		for k in dic_indi:
			temp = meal_decom[k] if q == 0 else temp+meal_decom[k]
			q += 1
		center = temp.mean(axis=0)
		error = temp-center
		temp_sse = np.sum(error**2, axis=0)/3
		sse = np.sum(temp_sse)
		
		final_sse += sse
	
	entro = []
	purity = []
	for i in range(num_cluster):
		kmean_dict = np.where(label == i)[0]
		j = 0 
		for k in kmean_dict:
			temp_data = meal_data[k,:]
			temp_clus = temp_data if j == 0 else np.vstack((temp_clus, temp_data))
			j += 1
		a,b,c,d,e,f = ground_truth(temp_clus)
		nd = [a,b,c,d,e,f]
		for p in range(len(nd)):
			if nd[p] == 31:
				nd[p] =1
		a,b,c,d,e,f = nd[0],nd[1],nd[2],nd[3],nd[4],nd[5]
		a,b,c,d,e,f = a/(temp_clus.shape[0]),b/(temp_clus.shape[0]),c/(temp_clus.shape[0]),d/(temp_clus.shape[0]),e/(temp_clus.shape[0]),f/(temp_clus.shape[0])
		en = log(a) + log(b) + log(c) + log(d) + log(e) + log(f)
		
		wei_en = temp_clus.shape[0]/(meal_decom.shape[0])*en
		
		entro.append(wei_en)
		pur = max(np.array([a,b,c,d,e,f]))
		wei_pur = temp_clus.shape[0]/(meal_decom.shape[0])*pur
		purity.append(wei_pur)
	
	return (sum(entro)), (sum(purity)*3), (final_sse)
en1, pur1, sse1 = (kmeans())
en2, pur2, sse2 = (dbscan())
result = np.array([sse1, sse2, en1, en2, pur1, pur2])
df1 = pd.DataFrame(result[None])
df1.to_csv("Result.csv", index=False, header=False)