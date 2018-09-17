#preprocessing


from sklearn import preprocessing
import numpy as np

data= np.array([[2.2,5.9,-1.8],[5.4,-3.2,-5.1],[-1.9,4.2, 3.2]])
data
bindata=preprocessing.Binarizer(threshold=1.5).transform(data)
bindata


#Mean removal

data.mean(axis=0)#array([ 1.9       ,  2.3       , -1.23333333])
data.std(axis=0)#highly variable array([2.98775278, 3.95052739, 3.41207008])

#so,
scaled_data=preprocessing.scale(data)
scaled_data.mean(axis=0)#array([0.00000000e+00, 0.00000000e+00, 7.40148683e-17])
scaled_data.std(axis=0)#array([1., 1., 1.])

#scaling
#work with same data
data

minmax_scaler=preprocessing.MinMaxScaler(feature_range=(0, 1))
data_minmax=minmax_scaler.fit_transform(data)
data_minmax


#Normalization
#bringing the values of each feature vector on a common scale

#L1 - Least Absolute Deviations - sum of absolute values (on each row) = 1; it is insensitive to outliers
#L2 - Least Squares - sum of squares (on each row) = 1; takes outliers in consideration during training

data

data_l1=preprocessing.normalize(data,norm="l1")
data_l1
#sum of the first row (absolute value) is browght to one
 0.22222222+0.5959596+0.18181818


data_l2=preprocessing.normalize(data,norm="l2")
data_l2
#sum of squares of the first row (absolute value) is browght to one

 0.3359268**2+0.9008946**2+(-0.2748492)**2
