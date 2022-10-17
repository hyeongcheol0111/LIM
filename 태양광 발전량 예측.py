#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#데이터 전처리


# In[3]:


import numpy as np


# In[4]:


Inputdata2=pd.read_csv("2번인버터.csv")


# In[5]:


Inputdata1=pd.read_csv("1번인버터.csv")


# In[6]:


Inputdata3=pd.read_csv("3번인버터.csv")


# In[7]:


Inputdata0=pd.read_csv("이게 진짜 테스트.csv")


# In[8]:


Inputdata0


# In[9]:


Inputdata1


# In[10]:


Inputdata2


# In[11]:


Inputdata3 


# In[12]:


#Outputdata=Inputdata.pop("Power(kW)") 


# In[13]:


#Outputdata


# In[14]:


#Train, Test set


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X=Inputdata1[['Temp','Rain','WindS','Pressure','MidBotCloud','WorstCloud','Insulation(MJ/m2)']]


# In[17]:


X


# In[18]:


Y1=Inputdata1['Power(kW)']


# In[19]:


Y1


# In[20]:


Y2=Inputdata2['Power(kW)']


# In[21]:


Y2


# In[22]:


Y3=Inputdata3['Power(kW)']


# In[23]:


Y3


# In[24]:


A=Inputdata0[['Temp','Rain','WindS','Pressure','MidBotCloud','WorstCloud','Insulation(MJ/m2)']]


# In[25]:


B=Inputdata0['Power(kW)']


# In[26]:


Z=Inputdata1[['Temp','Rain','WindS','Pressure','MidBotCloud','WorstCloud','Insulation(MJ/m2)','Power(kW)']]


# In[27]:


X_train, X_test, Y1_train, Y1_test=train_test_split(X,Y1,test_size=0.4,shuffle=False, random_state=1004)


# In[28]:


X_train, X_test, Y2_train, Y2_test=train_test_split(X,Y2,test_size=0.4,shuffle=False, random_state=1004)


# In[29]:


X_train, X_test, Y3_train, Y3_test=train_test_split(X,Y3,test_size=0.4,shuffle=False, random_state=1004)


# In[30]:


X_train


# In[31]:


X_test


# In[32]:


Y1_train


# In[33]:


Y1_test


# In[34]:


Y2_train


# In[35]:


Y2_test


# In[36]:


Y3_train


# In[37]:


Y3_test


# In[38]:


#정규화


# In[39]:


#from sklearn.preprocessing import StandardScaler


# In[40]:


#scaler=StandardScaler()


# In[41]:


#scaler.fit(X)


# In[42]:


#Normalized_Values=scaler.transform(X)


# In[43]:


#NVdata=pd.DataFrame(Normalized_Values,columns=X.columns)


# In[44]:


#NVdata


# In[45]:


from sklearn.preprocessing import MinMaxScaler


# In[46]:


scaler=MinMaxScaler()


# In[47]:


scaler.fit(X_train)


# In[48]:


Normalized_Values_train=scaler.transform(X_train)


# In[49]:


Normal_X_train=pd.DataFrame(Normalized_Values_train,columns=X_train.columns)


# In[50]:


Normal_X_train


# In[51]:


scaler.fit(X_test)


# In[52]:


Normalized_Values_test=scaler.transform(X_test)


# In[53]:


Normal_X_test=pd.DataFrame(Normalized_Values_test,columns=X_test.columns)


# In[54]:


Normal_X_test


# In[55]:


#데이터 시각화


# In[56]:


import matplotlib.pyplot as plt


# In[57]:


plt.plot(Normalized_Values_train)


# In[58]:


#모델설계


# In[59]:


from tensorflow.keras.models import Sequential


# In[60]:


from tensorflow.keras.layers import Dense


# In[61]:


import tensorflow as tf


# In[62]:


from tensorflow.keras.callbacks import EarlyStopping


# In[63]:


np.random.seed(3)


# In[64]:


tf.random.set_seed(3)


# In[65]:


model=Sequential()


# In[66]:


model.add(Dense(30,input_dim=7, activation='relu'))


# In[67]:


model.add(Dense(25, activation='relu'))


# In[68]:


model.add(Dense(15, activation='relu'))


# In[69]:


model.add(Dense(1, activation='relu'))


# In[70]:


model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])


# In[71]:


earlystopper=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=50, verbose=1)


# In[72]:


history1= model.fit(Normal_X_train, Y1_train, epochs=100, batch_size=100, callbacks=[earlystopper], validation_data=(Normal_X_test, Y1_test))


# In[73]:


history2= model.fit(Normal_X_train, Y2_train, epochs=100, batch_size=100, callbacks=[earlystopper], validation_data=(Normal_X_test, Y2_test))


# In[74]:


history3= model.fit(Normal_X_train, Y3_train, epochs=100, batch_size=100, callbacks=[earlystopper], validation_data=(Normal_X_test, Y3_test))


# In[75]:


plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.ylim(0,2)
plt.plot(history1.history['mse'])
plt.plot(history2.history['mse'])
plt.plot(history3.history['mse'])


# In[76]:


plt.xlabel('Epoch')
plt.ylim(2,5)
plt.ylabel('MSE')
plt.plot(history1.history['val_mse'])
plt.plot(history2.history['val_mse'])
plt.plot(history3.history['val_mse'])


# In[77]:


#상관도분석


# In[78]:


import seaborn as sns


# In[79]:


plt.figure(figsize=(100,100))


# In[80]:


sns.heatmap(Z.corr(),linewidths=0.3,
annot=True,annot_kws={'size' : 6},
cmap=plt.cm.gist_heat,
vmin=-1,vmax=1,)


# In[81]:


#예측값과 실제값 비교


# In[82]:


#Id=pd.read_csv("기상자료 시험.csv")


# In[83]:


#Id


# In[84]:


#tX=Id[['Temp','WindS','WindD','Hum','VaporPressure','dewpointTemp','Pressure','SnowL','TotCloud','MidBotCloud','WorstCloud','Insulation(MJ/m2)']]


# In[85]:


#tY=Id['Power(kWh/m2)']


# In[86]:


#정규화
scaler.fit(A)
Normalized_Values_A=scaler.transform(A)
Normal_A=pd.DataFrame(Normalized_Values_A,columns=A.columns)


# In[87]:


Normal_A


# In[88]:


tY_prediction=model.predict(Normal_A).flatten() #예측값


# In[89]:


plt.figure(figsize=(10,5))
plt.plot(tY_prediction)
plt.plot(B)
plt.xlim(144,264)


# In[90]:


#예측 파랑 , 실제 주황


# In[91]:


tY_prediction=(0.8)*model.predict(Normal_A).flatten() #예측값의 파라미터값을 0.8으로조정


# In[92]:


plt.figure(figsize=(10,5))
plt.plot(tY_prediction)
plt.plot(B)
plt.xlim(144,264)


# In[93]:


#예측 파랑 , 실제 주황

