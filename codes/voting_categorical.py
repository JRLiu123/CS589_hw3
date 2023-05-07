#!/usr/bin/env python
# coding: utf-8



#just run codes in jupyter notebook

import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle


class Bagging:
    
    def __init__(self,file_path,k):
        
        self.file_path = file_path 
        self.k = k
        
    def readfile(self):

        df = pd.read_csv(self.file_path)
        
        # unique classes
        classes = np.unique(df.loc[:,'class'])
        
        # spilt dataset according to classes
        D1 = df.loc[df['class'] == classes[0]]
        D2 = df.loc[df['class'] == classes[1]]
       

        # shuffle the subset
        D1 = shuffle(D1,random_state=0)
        D2 = shuffle(D2,random_state=0)
       
        
        #drop the original index and reset index
        D1 = D1.reset_index(drop=True)
        D2 = D2.reset_index(drop=True)
       
        
        return df,D1,D2
 
    def get_index(self,D1,D2,i):
        # the index to split test from train
        index1= np.rint(np.linspace(0, (D1.shape[0]-1), num=self.k+1))
        index2= np.rint(np.linspace(0, (D2.shape[0]-1), num=self.k+1))
        
        i11 = int(index1[i])
        i12 = int(index1[i+1])

        i21 = int(index2[i])
        i22 = int(index2[i+1])
        
        return i11,i12,i21,i22
        
    def stratified(self,D1,D2,i11,i12,i21,i22):

        D1_test = D1.loc[i11:i12]
        D1_train1 = D1.loc[0:i11-1]
        D1_train2 = D1.loc[i12+1:]
        
        D2_test = D2.loc[i21:i22]
        D2_train1 = D2.loc[0:i21-1]
        D2_train2 = D2.loc[i22+1:]

        frame_train = [D1_train1, D1_train2, D2_train1, D2_train2]
        train = pd.concat(frame_train)
        train = shuffle(train)
        
        frame_test = [D1_test,D2_test]
        test = pd.concat(frame_test)  
        test = shuffle(test)
        
        return test,train
   
    def bootstrap(self,train):
        
        index = np.random.choice(train.shape[0], size=train.shape[0], replace=True)
        
        train = pd.DataFrame(train)
      
        newdf = train.iloc[index]
        
        return newdf  



#calculate the entropy
def entropy(numerator,denominator):
    p = numerator/denominator
    I = -p*np.log2(p)
    return I
# select the unique values in dataset
def unique_values(train_set,attr):
    values = np.unique(train_set[attr])
    return values
def select_attr(train_set):
    attr_list = np.random.choice(train_set.columns.values.tolist()[0:-1], 
                                 size=int(np.sqrt(train_set.shape[1]-1)), replace=False)
    return attr_list

#calculate the average entropy of branches
#attr: the parent of branches
def cal_info(train_set,attr): 
    
    I_avg = 0
    
    attr_values = unique_values(train_set,attr)
    
    for value in attr_values:
        info = 0
        attr_set = train_set[train_set[attr] == value]
        # the weigh of the number of att_set[where(att==value)] and the number of att_set
        w = len(attr_set)/(len(train_set))
        
        for label in [0,1]:
            # the length of (attr ==value)&&('target'==label)
            num = len(train_set[(train_set[attr] == value) & (train_set['class'] == label)])
            # add a small num 10**(-20) in case of the denominator==0
            p = num/(len(attr_set)+10**(-20))
            info = info + (-(p)*np.log2(p+10**(-20)))
            
        I_avg = I_avg + w*info

    return I_avg

# select the majority label of original dataset
def majority_label(train_set):
    value_num = train_set.loc[:,'class'].value_counts() 
    most = value_num.idxmax()
    return most

# choose node at this moment
# most:the majority label of whole dataset
# uni_data: the unique values in this node attribute
# return decision_tree
def choose_node(train_set):
    most = majority_label(train_set)
    I_ori = 0
    for label in [0,1]:
        I = entropy(len(train_set[train_set['class'] == label]),len(train_set))
        I_ori = I_ori + I
        
    I_set = {}
    attr_list = select_attr(train_set)
    
    for attr in attr_list:

        info_attr = cal_info(train_set,attr)
        I_set[attr] = I_ori - info_attr
        
    # sort the information gain     
    I_set = sorted(I_set.items(), key=lambda x:x[1])
  
    #select the biggest gain
    node_name = I_set[-1][0]
    

    decision_tree = {}
    decision_tree[node_name] = {}
    
    # create a list with values in [0,1,2] but not in current attr_set
    left_value = []
    standard_value = np.array([0,1,2])

    node_value = unique_values(train_set,node_name)
    
    for i in standard_value:
        if i not in node_value and i not in left_value:
            left_value.append(i)
            
    #If this instance is not suitable for this decision tree        
    #Let left_value[0] be a leaf node labeled with the majority class in train_set        
    if left_value !=[]:
        for i in range(len(left_value)):
            
            decision_tree[node_name][left_value[i]] = most
        
    for value in node_value:
        # drop the column of current node from the current dataset
        new_set = train_set[train_set[node_name] == value]
        #new_set = new_set.drop([node_name], axis=1)
        
        uni_data = unique_values(new_set,'class')
        # if there is only one value in this set, we define this attributs as new node
        # if not, recursion
        if len(uni_data) == 1: 
            decision_tree[node_name][value] = uni_data[0]
        else:
            decision_tree[node_name][value] = choose_node(new_set)          
        
    return decision_tree  
# predict the label of new instance
# new_ins: single instance in train_set or test_set
def label(new_ins, tree):
    
    for i in tree.keys():
        #find the value of the root node of new instance
        value = new_ins[i]
        # determine whether the node exsits or not
        # if exists, go to next node
        # else, assign the majority labels of whole data_set to this new instance
        #print(tree[i][value]dict.has_key)
        '''
        if (value in tree[i]):
            tree = tree[i][value]
        else:
            tree = most
        '''
        tree = tree[i][value]
        predict_label = None
        # if this is not the end node, recursion will be used.
        if type(tree) is dict:
            predict_label = label(new_ins, tree)
        else:
            predict_label = tree
            break

    return predict_label

# calculate the number of correct prediction
def predict_list(test_set, decision_tree):
    

    predict_label = np.zeros(len(test_set))
    for j in range(len(test_set)):
 
        predict_label[j] = label(test_set.iloc[j][:-1], decision_tree) 
    
    return predict_label
def plot_acc(y,label = 'Accuracy'):
    x = [1, 5, 10, 20, 30, 40, 50]
    plt.plot(x,y)
    plt.scatter(x,y)
    plt.xlabel('ntrees')
    plt.ylabel(label)
    plt.title(label + ' of the random forest as a function of ntree for the voting dataset')
    #plt.savefig('/home/jingran/CS589_hw3/figures/voting_'+label+'.png')
    plt.show()



if __name__ == '__main__':
    file_path = 'datasets/hw3_house_votes_84.csv'
    k = 10 #k: the number of folders using in cross-validation
    
    dataset = Bagging(file_path)
    df,D1,D2 = dataset.readfile()
    ntrees_list = [1, 5, 10, 20, 30, 40,50]
    accuracy = np.zeros((k,len(ntrees_list)))
    precision = np.zeros((k,len(ntrees_list)))
    recall = np.zeros((k,len(ntrees_list)))
    F1_score = np.zeros((k,len(ntrees_list)))
    
    for index in range(k):
        print('This is the '+str(index)+'-th loop.')
        #stratified
        i11,i12,i21,i22 = dataset.get_index(D1,D2,index)
        test,train = dataset.stratified(D1,D2,i11,i12,i21,i22) 
        
        
        for j in range(len(ntrees_list)):
            
            ntrees = ntrees_list[j]
            
            print('The number of trees are: ',ntrees)
            predict = np.zeros((len(test),ntrees))
            for n in range(ntrees):
                
                new_train = dataset.bootstrap(train)
                decision_tree = choose_node(new_train) 

                predict_label= predict_list(test, decision_tree)

                predict[:,n] = predict_label

            predict = pd.DataFrame(predict)        
            confusion_matrix = np.zeros((2,2))
            for i in range(len(test)):
                majority = 0
                value_num = predict.iloc[i].value_counts()
                majority = value_num.idxmax()


                if ((test.iloc[i,-1] == 0)&(majority == 0)):
                    confusion_matrix[0,0] = confusion_matrix[0,0] + 1

                elif ((test.iloc[i,-1] == 1)&(majority == 1)):
                    confusion_matrix[1,1] = confusion_matrix[1,1] + 1

                elif ((test.iloc[i,-1] == 0)&(majority == 1)):
                    confusion_matrix[0,1] = confusion_matrix[0,1] + 1

                elif ((test.iloc[i,-1] == 1)&(majority == 0)):
                    confusion_matrix[1,0] = confusion_matrix[1,0] + 1
                    
            accuracy[index,j] =  (confusion_matrix[0,0] + confusion_matrix[1,1])/len(test)
            precision[index,j] = (confusion_matrix[0,0])/(confusion_matrix[0,0] + confusion_matrix[1,0])
            recall[index,j] = (confusion_matrix[0,0])/(confusion_matrix[0,0] + confusion_matrix[0,1])
            F1_score[index,j] = (2*precision[index,j]*recall[index,j])/(precision[index,j] + recall[index,j])
            print(accuracy[index,j],precision[index,j],recall[index,j],F1_score[index,j])
        print('#####################################################################')
    print('The accuracy is: ',np.average(accuracy, axis=0))
    print('The precision is: ',np.average(precision, axis=0))
    print('The recall is: ',np.average(recall, axis=0))
    print('The F1_score is: ',np.average(F1_score, axis=0))





import matplotlib.pyplot as plt
y = [0.89867589, 0.93849583, 0.96487484, 0.9472881,  0.949361,   0.95380764,
 0.94496487]
x = [1, 5, 10, 20, 30, 40, 50]
plt.plot(x,y)
plt.scatter(x,y)
plt.xlabel('ntrees')
plt.ylabel('F1_score')
plt.title('F1_score of the random forest as a function of ntree for the voting dataset')
#plt.savefig('/home/jingran/CS589_hw3/figures/voting_F1_score.png')
plt.show()





