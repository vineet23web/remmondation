import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from google.cloud import storage
import gcsfs
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


# Change json_path
json_path="C:/Users/JENILPATEL/Desktop/project-e3a05-88282a7bc99f.json"

fs = gcsfs.GCSFileSystem(token= json_path, project='project')
with fs.open('gs://project-e3a05.appspot.com/tiles_data_full.csv') as f:
    df = pd.read_csv(f)

df = df.drop(['NAME','URL','LENGTH', 'WIDTH', 'COLOR', 'SQ.FT.'] ,axis=1)

df2=df.iloc[:,[0,1,7,8,9,2,3,4,5,6]]
y=df2.iloc[:,0]
x=df2.iloc[:,2:]

#firebase setup
cred = credentials.Certificate(json_path)
firebase_admin.initialize_app(cred, name = 'database')

firebase_admin.initialize_app(cred, {'databaseURL' : 'https://project-e3a05-default-rtdb.asia-southeast1.firebasedatabase.app/'})
ref = db.reference('recommendation/')
button_ref = ref.child('buttonValue')
recom_ref = ref.child('values')

start=int(random.randrange(len(df)+1))

print(start)
button_ref.set({
    'Input' : 2,
    'flag': False
    })

history=[]
temp=[]
repeat=0

while True:
    inp_temp= ref.child("buttonValue").get()
    inp = (list(inp_temp.values())[0])
    flag = (list(inp_temp.values())[1])
    present=0
    prev=start
    recom_ref.update({'initial': int(start)})
    if start not in history:
        print("adding new element: ", )
        history.append(start)
        print(history)
    
    
    
    if(inp==5 and flag == False):
        del history[:]
        
        break
    elif(inp==0 and flag == True): #inp == 0 and flag == True
        print('\n \n start value: ', start)
        history.pop()
        start=int(random.randrange(len(df)+1))
        recom_ref.update({'initial': int(start)})
        button_ref.update({'flag': False})
        print('\n \n start value at the end: ', start)
        
    elif(inp==1 and flag == True): #inp == 1 and flag == True
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=int(random.random()*100))
        
        try:
            if ( x_train.loc[start].any()):
                x_train=x_train.drop(start,axis=0)
                y_train=y_train.drop(start,axis=0)
        except:
            present=0

        y_train=y_train.values
        
        
        input_pred=df2.iloc[start][2:]
        input_pred=np.array(input_pred)
        input_pred=input_pred.reshape(1,-1)

        
        knn_model = KNeighborsClassifier(n_neighbors =3)
        knn_model.fit(x_train,y_train)
        
        
        output=knn_model.predict(input_pred)
        output2=[]
        if(output[0] not in history and start != output[0]):
            start=output[0]   
        else: 
            repeat+=1
            input_pred=df2.iloc[start][2:]
            input_pred=np.array(input_pred)
            input_pred=input_pred.reshape(1,-1)
            output2=knn_model.predict(input_pred)
            
            start=output2[0]
            print('This is value  of output2', start)
            
        if(repeat>=1 and output2[0]==output[0]):
            
            temp=temp+history
            history=[]
            repeat=0
            start=int(random.randrange(len(df)+1))
            print('\nThis is value  of random start value: ', start)

        #recom_ref.update({'initial': int(prev)})
        recom_ref.update({'recommendation': int(start)})
        button_ref.update({
            'flag': False
            })

        if(flag == True):
            print(start)
        print("\n\n Start value: ", start)
        print(history)

print(repeat)
