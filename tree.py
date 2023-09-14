# importing pandas
import pandas as pd
import numpy as np
#x: data matrix (atributes of type numeric)
#row = attributes of one training sample
#y: vector with classes, 0/1
#nmin: number of observations to be a split otherwise leafnode
#minleaf: min observations to be a leafnode when splitted, consider when splitting if no split-> leafnode
#use gini index for quality of split
#nfeat: number of features per split -> random draw nfeat
#return tree object -> predict new cases
def tree_grow(x,y,nmin,minleaf,nfeat):
    pass
#data matrix
#tr: tree object 
#return y_predict
def tree_pred(x,tr):
    pass

#function for real data
def load_data(path):
    # read text file into pandas DataFrame
    df = pd.read_csv(path, sep=",")
    #first 8 columns: numerical data
    x = df.iloc[:,:8]
    #last column = label(binary value) 
    y = df.iloc[:,8]
    return x,y

class Tree():
    def __init__(self, root) -> None:
        self.root = root

class Node():
    def __init__(self, indices) -> None:
        self.indices = indices

        self.pure = False
        
        self.split_value = None
        self.split_feature = None
        
        self.left = None
        self.right = None
    
    def calculate_gini(self,part):
        total = len(part)
        p_zeros = len(np.where(part == 0)[0])/total
        p_ones = 1-p_zeros
        return p_zeros * p_ones
    
    def split(self):
        #min gini per feature
        min_gini_for_all_features = []
        #get x_data and y_data from df
        x_data = x.iloc[self.indices]
        y_data = y.iloc[self.indices]
        #get 1 feature and decide best splitpoint
        for col_index in range(len(x_data.columns)):
            #get 1 feature
            f = x_data.iloc[:, col_index]
            #sort numeric feature
            sorted_indices = np.argsort(f)
            #sort y at the same way
            f_s = f[sorted_indices]
            y_s = y_data[sorted_indices]
            
            #calculate gini for different splits of feature
            gini_splits = []
            #start at index one of the feature
            for index in range(1,len(f_s)):
                y_s_part_one = y_s[0:index]
                gini_part_one = self.calculate_gini(y_s_part_one)

                y_s_part_two = y_s[index:]
                gini_part_two = self.calculate_gini(y_s_part_two)
                
                total_gini = len(y_s_part_one)/len(f_s) * gini_part_one + len(y_s_part_two)/len(f_s) * gini_part_two
                gini_splits.append((f_s[index],total_gini,gini_part_one,gini_part_two))
            
            f_s_min, min_gini_total,min_gini_p1,min_gini_p2 = min(gini_splits,key=lambda x:x[1])
            min_gini_for_all_features.append((col_index,f_s_min,min_gini_total,min_gini_p1,min_gini_p2))
            
        #best feature to split with the split point
        col_index,best_split_point,min_gini_total,min_gini_p1,min_gini_p2 = min(min_gini_for_all_features, key=lambda x: x[2])
        print(col_index,best_split_point,min_gini_total,min_gini_p1,min_gini_p2)
        self.split_feature = col_index
        self.split_value = best_split_point

        #set indices of data for left and right of split
        left_indices = x_data.loc[x_data.iloc[:, col_index]<best_split_point].index
        right_indices = x_data.loc[x_data.iloc[:, col_index]>=best_split_point].index
        print(left_indices)
        print(right_indices)
       
        #create left and right node
        left_node = Node(left_indices)
        right_node = Node(right_indices)

        #check wheter left or right are pure
        if min_gini_p1 == 0:
            left_node.pure = True
        if min_gini_p2 == 0:
            right_node.pure = True

        return left_node,right_node


    
#sample data
data = {'X': [1, 2, 3,4,5],
        #'Z': [1, 3, 8, 10,20,30],
        'Y': [0,1,0,0,1]}
data = pd.DataFrame(data)

x = data.iloc[:,:1]
y = data['Y']

root = Node(x.index)
tree = Tree(root)
nodes_to_visit = [root]
while True:
    print(nodes_to_visit)
    if len(nodes_to_visit) == 0:
        print(1)
        break
    current_node = nodes_to_visit[0]
    left_node,right_node = current_node.split()
    if left_node.pure == False:
        nodes_to_visit.append(left_node)
    if right_node.pure == False:
        nodes_to_visit.append(right_node)
    nodes_to_visit.pop(0)
