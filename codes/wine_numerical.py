#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle



class Bagging:
    
    def __init__(self,file_path,k):
        
        self.file_path = file_path 
        self.k = k
        
    def readfile(self):

        df = pd.read_csv(self.file_path,delimiter='\t')
        
        # unique classes
        classes = np.unique(df.loc[:,'# class'])
        
        # spilt dataset according to classes
        D1 = df.loc[df['# class'] == classes[0]]
        D2 = df.loc[df['# class'] == classes[1]]
        D3 = df.loc[df['# class'] == classes[2]]

        # shuffle the subset
        D1 = shuffle(D1,random_state=0)
        D2 = shuffle(D2,random_state=0)
        D3 = shuffle(D3,random_state=0)
        
        #drop the original index and reset index
        D1 = D1.reset_index(drop=True)
        D2 = D2.reset_index(drop=True)
        D3 = D3.reset_index(drop=True)
        
        return df,D1,D2,D3
 
    def get_index(self,D1,D2,D3,i):
        # the index to split test from train
        index1= np.rint(np.linspace(0, (D1.shape[0]-1), num=self.k+1))
        index2= np.rint(np.linspace(0, (D2.shape[0]-1), num=self.k+1))
        index3= np.rint(np.linspace(0, (D3.shape[0]-1), num=self.k+1))

        
        i11 = int(index1[i])
        i12 = int(index1[i+1])

        i21 = int(index2[i])
        i22 = int(index2[i+1])
        
        i31 = int(index3[i])
        i32 = int(index3[i+1])
        
        return i11,i12,i21,i22,i31,i32
        
    def stratified(self,D1,D2,D3,i11,i12,i21,i22,i31,i32):

        D1_test = D1.loc[i11:i12]
        D1_train1 = D1.loc[0:i11-1]
        D1_train2 = D1.loc[i12+1:]
        

        D2_test = D2.loc[i21:i22]
        D2_train1 = D2.loc[0:i21-1]
        D2_train2 = D2.loc[i22+1:]
 
            
        D3_test = D3.loc[i31:i32]
        D3_train1 = D3.loc[0:i31-1]
        D3_train2 = D3.loc[i32+1:]


        frame_train = [D1_train1, D1_train2, D2_train1, D2_train2, D3_train1, D3_train2]
        train = pd.concat(frame_train)
        #train = shuffle(train)
        
        frame_test = [D1_test,D2_test, D3_test]
        test = pd.concat(frame_test)   
        #test = shuffle(test)
        
        return test,train
   
    def bootstrap(self,train):
        
        index = np.random.choice(train.shape[0], size=train.shape[0], replace=True)
        
        train = pd.DataFrame(train)
      
        newdf = train.iloc[index]
       

        return newdf    




class Decision_tree:
    def __init__(self, value): 
        self.value = value
        self.split_node = {}
        self.threshold = None
def entropy(numerator,denominator):
    p = numerator/denominator
    I = -p*np.log2(p)
    return I

# select the unique values in dataset
def unique_values(train_set,attr):
    values = np.unique(train_set[attr])
    return values

# select the majority label of original dataset
# value_num: the number of unique classes left
def majority_label(train_set):
    
    value_num = train_set['# class'].value_counts() 
    
    most = value_num.idxmax()
   
    return value_num, most

#selcet n attributes with random
#n = np.sqrt(n_attrs)
def select_attr(train_set):
    attr_list = np.random.choice(train_set.columns.values.tolist()[1:], 
                                 size=int(np.sqrt(train_set.shape[1]-1)), replace=False)
    return attr_list

# spilt node according to the mean value of this node
def split_threshold(attr_data):
    
    thresholds = np.mean(attr_data)
    
    return thresholds
    
# calculate the entropy of each available attribute
def cal_info_num(train_set,attr): 
    
    I_avg = 0
   
    thresholds = split_threshold(train_set[attr])
    
    n_classes = unique_values(train_set,'# class')
    
    
    for i in range(2):
      
        # when the value of this attr is bigger than mean value
        if (i==0):
            attr_set = train_set[train_set[attr] >= thresholds]
            
        # when the value of this attr is smaller than mean value    
        else:
            attr_set = train_set[train_set[attr] < thresholds]
            
        # the weigh of the number of att_set[where(att==value)] and the number of att_set
        w = len(attr_set)/(len(train_set))
        
        info = 0
        for label in n_classes:
            if (i==0):
                # the length of (attr ==value)&&('target'==label)
                num = len(train_set[(train_set[attr] >= thresholds) & (train_set['# class'] == label)])
            else:
                num = len(train_set[(train_set[attr] < thresholds) & (train_set['# class'] == label)])
            # add a small num 10**(-20) in case of the denominator==0
            p = num/(len(attr_set)+10**(-20))
            info = info + (-(p)*np.log2(p+10**(-20)))
            
        I_avg = I_avg + w*info

    return I_avg

def choose_node_num(train_set):
    
    attr_list = select_attr(train_set)
    value_num, most = majority_label(train_set)#most means the majority of classes in the current set.
    if len(attr_list)==0:
        decision_tree = Decision_tree(most)
    elif len(value_num)==1:
        decision_tree = Decision_tree(most)  
    else:    
        I_ori = 0

        n_classes = unique_values(train_set,'# class')
        #calculate the entropy before spliting
        for label in n_classes:
            I = entropy(len(train_set[train_set['# class'] == label]),len(train_set))
            I_ori = I_ori + I

        I_set = {}

        for attr in attr_list:

            info_attr = cal_info_num(train_set,attr)
            I_set[attr] = I_ori - info_attr

        # sort the information gain     
        I_set = sorted(I_set.items(), key=lambda x:x[1])

        #select the biggest gain
        node_name = I_set[-1][0]

        decision_tree = Decision_tree(node_name)
        thresholds = split_threshold(train_set[node_name])
        decision_tree.threshold = thresholds
        
        new_df_big = train_set[train_set[node_name] >= thresholds]
        new_df_small = train_set[train_set[node_name] < thresholds]
        
        
        if len(new_df_big) == 0:
            decision_tree.split_node['bigger'] = Decision_tree(most)
        else:
            decision_tree.split_node['bigger'] = choose_node_num(new_df_big)
            
        if len(new_df_small) == 0:
            decision_tree.split_node['smaller'] = Decision_tree(most)
        else:
            decision_tree.split_node['smaller'] = choose_node_num(new_df_small)
            
    return decision_tree

#function to traverse the tree predicting label for is_numeric=true
def label(ins,decision_tree):
    
    if decision_tree.threshold is None:
        
        return decision_tree.value
    
    else: 
        
        if ins[decision_tree.value] >= decision_tree.threshold:
            return label(ins,decision_tree.split_node['bigger'])
        
        if ins[decision_tree.value] < decision_tree.threshold:
            return label(ins,decision_tree.split_node['smaller'])
        
def prediction(test_set,decision_tree):
    predict_label = np.zeros(len(test_set))
    for j in range(len(test_set)):
        predict_label[j] = label(test_set.iloc[j][1:], decision_tree)
    return predict_label



if __name__ == '__main__':
    file_path = 'datasets/hw3_wine.csv'
    k = 10 #k: the number of folders using in cross-validation
    
    dataset = Bagging(file_path,k)
    df,D1,D2,D3 = dataset.readfile()
    ntrees_list = [1, 5, 10, 20, 30, 40,50]
    accuracy = np.zeros((k,len(ntrees_list)))
    precision = np.zeros((k,len(ntrees_list)))
    recall = np.zeros((k,len(ntrees_list)))
    F1_score = np.zeros((k,len(ntrees_list)))
    
    for index in range(k):
        print('This is the '+str(index)+'-th loop.')
        #stratified
        i11,i12,i21,i22,i31,i32 = dataset.get_index(D1,D2,D3,index)

        test,train = dataset.stratified(D1,D2,D3,i11,i12,i21,i22,i31,i32)
        column = list(train.columns.values)
        
        
        for j in range(len(ntrees_list)):
            
            ntrees = ntrees_list[j]
            
            print('The number of trees are: ',ntrees)
            predict = np.zeros((len(test),ntrees))
            for n in range(ntrees):
                
                new_train = dataset.bootstrap(train)
                
               
                decision_tree = choose_node_num(new_train)
                predict_label = prediction(test,decision_tree)
              
                print('predict_label',predict_label)
                predict[:,n] = predict_label

            predict = pd.DataFrame(predict)  
            pre = np.zeros(3)
            rec = np.zeros(3)
            confusion_matrix = np.zeros((3,3))
            for i in range(len(test)):
                majority = 0
                value_num = predict.iloc[i].value_counts()
                majority = value_num.idxmax()


                
                if ((test.iloc[i,0] == 1)&(majority == 1)):
                    confusion_matrix[0,0] = confusion_matrix[0,0] + 1
                elif ((test.iloc[i,0] == 1)&(majority == 2)):
                    confusion_matrix[0,1] = confusion_matrix[0,1] + 1
                elif ((test.iloc[i,0] == 1)&(majority == 3)):
                    confusion_matrix[0,2] = confusion_matrix[0,2] + 1
                    
                    
                elif ((test.iloc[i,0] == 2)&(majority == 1)):
                    confusion_matrix[1,0] = confusion_matrix[1,0] + 1
                elif ((test.iloc[i,0] == 2)&(majority == 2)):
                    confusion_matrix[1,1] = confusion_matrix[1,1] + 1
                elif ((test.iloc[i,0] == 2)&(majority == 3)):
                    confusion_matrix[1,2] = confusion_matrix[1,2] + 1
                    
                    
                elif ((test.iloc[i,0] == 3)&(majority == 1)):
                    confusion_matrix[2,0] = confusion_matrix[2,0] + 1    
                elif ((test.iloc[i,0] == 3)&(majority == 2)):
                    confusion_matrix[2,1] = confusion_matrix[2,1] + 1
                elif ((test.iloc[i,0] == 3)&(majority == 3)):
                    confusion_matrix[2,2] = confusion_matrix[2,2] + 1
               
            accuracy[index,j] =  (confusion_matrix[0,0] + confusion_matrix[1,1] + confusion_matrix[2,2])/len(test)
            pre[0] =  (confusion_matrix[0,0])/ (confusion_matrix[0,0] + 
                                                confusion_matrix[1,0] + confusion_matrix[2,0])
            pre[1] =  (confusion_matrix[1,1])/ (confusion_matrix[1,1] + 
                                                confusion_matrix[0,1] + confusion_matrix[2,1])
            pre[2] =  (confusion_matrix[2,2])/ (confusion_matrix[2,2] + 
                                                confusion_matrix[0,2] + confusion_matrix[1,2])
            precision[index,j] = (pre[0] + pre[1] + pre[2])/3
                
            rec[0] = (confusion_matrix[0,0])/ (confusion_matrix[0,0] + 
                                               confusion_matrix[0,1] + confusion_matrix[0,2])
            rec[1] = (confusion_matrix[1,1])/ (confusion_matrix[1,1] + 
                                               confusion_matrix[1,0] + confusion_matrix[1,2])   
            rec[2] = (confusion_matrix[2,2])/ (confusion_matrix[2,2] + 
                                               confusion_matrix[2,0] + confusion_matrix[2,1])
            recall[index,j] = (rec[0] + rec[1] + rec[2])/3
    
            F1_score[index,j] = (2*precision[index,j]*recall[index,j])/(precision[index,j] + recall[index,j])
            print(accuracy[index,j],precision[index,j],recall[index,j],F1_score[index,j])
        print('#####################################################################')
    print('The accuracy is: ',np.average(accuracy, axis=0))
    print('The precision is: ',np.average(precision, axis=0))
    print('The recall is: ',np.average(recall, axis=0))
    print('The F1_score is: ',np.average(F1_score, axis=0))




import matplotlib.pyplot as plt
y = [0.86333333, 0.95119048 ,0.96071429 ,0.96142857 ,0.98095238, 0.98071429,
 0.97547619]
x = [1, 5, 10, 20, 30, 40,50]
plt.plot(x,y)
plt.scatter(x,y)
plt.xlabel('the number of trees in thr random forest')
plt.ylabel('Accuracy')
plt.title('Accuracy of the random forest as a function of ntree for the wine dataset')
#plt.savefig('/home/jingran/CS589_hw3/figures/wine_Accuracy.png')
plt.show()






