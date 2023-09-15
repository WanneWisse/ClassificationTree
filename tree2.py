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
    df = pd.read_csv(path, sep=",",header=None)
    #first 8 columns: numerical data
    x = df.iloc[:,:5]
    #last column = label(binary value) 
    y = df.iloc[:,5]
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
    
    def split(self,min_leaf):
        #min gini per feature
        min_gini_for_all_features = []
        #get x_data and y_data from df
        x_data = x.iloc[self.indices]
        y_data = y.iloc[self.indices]
        
        #get 1 feature and decide best splitpoint
        for col_index in range(len(x_data.columns)):
            #get 1 feature
            f = x_data.iloc[:, col_index]
            f_sorted = np.sort(np.unique(f)) # Sort e.g. income vals low to high
            possible_splits = (f_sorted[0:len(f_sorted)-1] + f_sorted[1:len(f_sorted)]) /2
            gini_splits = []
            #start at index one of the feature
            for split in possible_splits:
                indices_s_part_one = x_data.loc[x_data.iloc[:, col_index]<split].index
                indices_s_part_two = x_data.loc[x_data.iloc[:, col_index]>=split].index
                
                y_s_part_one =  y.iloc[indices_s_part_one]
                y_s_part_two =  y.iloc[indices_s_part_two]
                
                gini_part_two = self.calculate_gini(y_s_part_two)
                gini_part_one = self.calculate_gini(y_s_part_one)
                total_gini = len(y_s_part_one)/len(f) * gini_part_one + len(y_s_part_two)/len(f) * gini_part_two
                
                gini_splits.append((split,total_gini,gini_part_one,gini_part_two,indices_s_part_one,indices_s_part_two))

            min_split, min_gini_total,min_gini_p1,min_gini_p2, indices_s_part_one,indices_s_part_two = min(gini_splits,key=lambda x:x[1])
        min_gini_for_all_features.append((col_index,min_split,min_gini_total,min_gini_p1,min_gini_p2, indices_s_part_one,indices_s_part_two))
            
        #best feature to split with the split point
        col_index,best_split_point,min_gini_total,min_gini_p1,min_gini_p2, indices_s_part_one,indices_s_part_two = min(min_gini_for_all_features, key=lambda x: x[2])
        print(f"best feature column: {col_index}, best split value(smaller then): {best_split_point}, minimal gini total: {min_gini_total}, minimal gini part 1: {min_gini_p1}, minimal gini part 2: {min_gini_p2}")
        self.split_feature = col_index
        self.split_value = best_split_point

        #set indices of data for left and right of split
        left_indices = indices_s_part_one
        right_indices = indices_s_part_two
        print("indices to smaller: ", left_indices)
        print("indices to larger or equal: ",right_indices)
       
        #create left and right node
        left_node = Node(left_indices)
        right_node = Node(right_indices)

        #check wheter left or right are pure

        if min_gini_p1 == 0 or len(left_node.indices) <= min_leaf:
            left_node.pure = True
        if min_gini_p2 == 0 or len(right_node.indices) <= min_leaf:
            right_node.pure = True

        return left_node,right_node


    
#sample data
data = {'X': [1, 2, 3,4,5],
        #'Z': [1, 3, 8, 10,20],
        'Y': [0,1,0,0,1]}
data = pd.DataFrame(data)

# x = data.iloc[:,:1]
# y = data['Y']
x,y = load_data("credit.txt")
x = x.iloc[:,[3]]
# x = x.iloc[:,[1]]
print(x)
print(y)

root = Node(x.index)
tree = Tree(root)
nodes_to_visit = [root]
while True:
    print("------------------")
    print("nodes in queue:", nodes_to_visit)
    if len(nodes_to_visit) == 0:
        break
    current_node = nodes_to_visit[0]
    left_node,right_node = current_node.split(2)
    if left_node.pure == False:
        nodes_to_visit.append(left_node)
    if right_node.pure == False:
        nodes_to_visit.append(right_node)
    nodes_to_visit.pop(0)
