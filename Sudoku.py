import pandas as pd
import numpy as np
import math

#df_initial = pd.DataFrame([[5, np.NaN, np.NaN, np.NaN, 2, np.NaN, np.NaN, np.NaN, np.NaN],
#                            [8, 3, np.NaN, np.NaN, 4, 9, np.NaN, np.NaN, 7],
#                            [6, np.NaN, np.NaN, 5, np.NaN, 1, 9, 8, np.NaN],
#                            [np.NaN, 5, np.NaN, 4, np.NaN, 6, 8, 3, 2],
#                            [np.NaN, np.NaN, 4, np.NaN, 5, 3, np.NaN, np.NaN, np.NaN],
#                            [np.NaN, np.NaN, 6, np.NaN, np.NaN, np.NaN, 7, np.NaN, 5],
#                            [4, np.NaN, 5, np.NaN, 6, np.NaN, 3, np.NaN, np.NaN],
#                            [7, np.NaN, np.NaN, 1, np.NaN, np.NaN, np.NaN, 2, 9],
#                            [9, 2, 8, np.NaN, 7, 4, np.NaN, 6, np.NaN]], 
#                            index=list(np.arange(1, 10)), columns=list(np.arange(1, 10))) 
#
#df_final = pd.DataFrame([[5, 4, 9, 8, 2, 7, 6, 1, 3],
#                        [8, 3, 1, 6, 4, 9, 2, 5, 7],
#                        [6, 7, 2, 5, 3, 1, 9, 8, 4],
#                        [1, 5, 7, 4, 9, 6, 8, 3, 2],
#                        [2, 8, 4, 7, 5, 3, 1, 9, 6],
#                        [3, 9, 6, 2, 1, 8, 7, 4, 5],
#                        [4, 1, 5, 9, 6, 2, 3, 7, 8],
#                        [7, 6, 3, 1, 8, 5, 4, 2, 9],
#                        [9, 2, 8, 3, 7, 4, 5, 6, 1]],
#                        index=list(np.arange(1, 10)), columns=list(np.arange(1, 10)))

#df_initial = pd.DataFrame([[np.NaN, 9, np.NaN, np.NaN, np.NaN, np.NaN, 8, np.NaN, 1],
#                            [1, 8, np.NaN, np.NaN, 2, np.NaN, np.NaN, np.NaN, np.NaN],
#                            [3, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
#                            [np.NaN, np.NaN, np.NaN, 5, 3, np.NaN, np.NaN, np.NaN, 4],
#                            [np.NaN, 7, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
#                            [5, 6, 8, 4, np.NaN, 9, np.NaN, np.NaN, np.NaN],
#                            [np.NaN, np.NaN, np.NaN, 6, np.NaN, np.NaN, 9, np.NaN, 5],
#                            [8, np.NaN, 6, 9, np.NaN, np.NaN, 4, np.NaN, np.NaN],
#                            [np.NaN, 1, np.NaN, np.NaN, 5, np.NaN, np.NaN, np.NaN, np.NaN]], 
#                            index=list(np.arange(1, 10)), columns=list(np.arange(1, 10))) 
#
#
#df_final = pd.DataFrame([[6, 9, 2, 3, 4, 5, 8, 7, 1],
#                        [1, 8, 5, 7, 2, 6, 3, 4, 9],
#                        [3, 4, 7, 8, 9, 1, 2, 5, 6],
#                        [9, 2, 1, 5, 3, 8, 7, 6, 4],
#                        [4, 7, 3, 1, 6, 2, 5, 9, 8],
#                        [5, 6, 8, 4, 7, 9, 1, 3, 2],
#                        [2, 3, 4, 6, 8, 7, 9, 1, 5],
#                        [8, 5, 6, 9, 1, 3, 4, 2, 7],
#                        [7, 1, 9, 2, 5, 4, 6, 8, 3]],
#                        index=list(np.arange(1, 10)), columns=list(np.arange(1, 10)))

#df_initial = pd.DataFrame([[2, 5, np.NaN, np.NaN, np.NaN, 3, np.NaN, np.NaN, np.NaN],
#                            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 2, 7, np.NaN],
#                            [8, 7, np.NaN, np.NaN, np.NaN, 6, 4, np.NaN, np.NaN],
#                            [np.NaN, 2, np.NaN, np.NaN, np.NaN, 8, 1, 9, 3],
#                            [np.NaN, 1, 5, np.NaN, 4, np.NaN, 8, np.NaN, np.NaN],
#                            [np.NaN, np.NaN, np.NaN, 1, np.NaN, np.NaN, np.NaN, np.NaN, 4],
#                            [np.NaN, np.NaN, np.NaN, 7, 3, 4, np.NaN, np.NaN, np.NaN],
#                            [np.NaN, np.NaN, np.NaN, 6, np.NaN, np.NaN, np.NaN, np.NaN, 9],
#                            [np.NaN, 6, 4, np.NaN, np.NaN, 9, np.NaN, 5, 8]],
#                            index=list(np.arange(1, 10)), columns=list(np.arange(1, 10)))

df_initial = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, 7, 3, 9, 8, 1],
                        [3, 4, 9, 8, 5, 1, 2, 7, 6],
                        [8, 7, np.NaN, np.NaN, 2, 6, 4, 3, 5],
                        [4, np.NaN, 7, 5, 6, 8, 1, 9, 3],
                        [9, 1, 5, 3, 4, 2, 8, 6, 7],
                        [6, 8, 3, 1, 9, 7, 5, 2, 4],
                        [5, 9, 8, 7, np.NaN, 4, 6, 1, 2],
                        [1, 3, 2, 6, 8, 5, 7, 4, 9],
                        [7, 6, 4, 2, 1, 9, 3, 5, 8]],
                        index=list(np.arange(1, 10)), columns=list(np.arange(1, 10)))

df_final= pd.DataFrame([[2, 5, 6, 4, 7, 3, 9, 8, 1],
                        [3, 4, 9, 8, 5, 1, 2, 7, 6],
                        [8, 7, 1, 9, 2, 6, 4, 3, 5],
                        [4, 2, 7, 5, 6, 8, 1, 9, 3],
                        [9, 1, 5, 3, 4, 2, 8, 6, 7],
                        [6, 8, 3, 1, 9, 7, 5, 2, 4],
                        [5, 9, 8, 7, 3, 4, 6, 1, 2],
                        [1, 3, 2, 6, 8, 5, 7, 4, 9],
                        [7, 6, 4, 2, 1, 9, 3, 5, 8]],
                        index=list(np.arange(1, 10)), columns=list(np.arange(1, 10)))

results_comparison = pd.DataFrame(columns=['strategy', 'steps_no', 'is_matrix_completed_correctly'])

def check_corectness():
    identical_column = []
    for i in df.columns:
        identical_column.append((df[i] == df_final[i]).all())
    is_df_correct = set(identical_column) == {True}
    return is_df_correct

def find_square_side(no):
    if no in [1, 2, 3]:
        return [1, 2, 3]
    if no in [4, 5, 6]:
        return [4, 5, 6]
    if no in [7, 8, 9]:
        return [7, 8, 9]
    else:
        raise Exception('Unexpected no in df')
        
def check_column(df, remaining_nums, row_idx, steps_no):
    for col_idx in remaining_nums.copy().keys():
        steps_no += 1 
        col_temp = df.loc[:, col_idx]
        # Remove numbers (from remaining_nums) which occur in the column
        remaining_nums[col_idx] = [i for i in remaining_nums[col_idx] if i not in col_temp.dropna().values]
        
        # If only one num left to be added
        if len(remaining_nums[col_idx]) == 1:
            # Add new number to our df
            df.loc[row_idx, col_idx] = remaining_nums[col_idx][0]
            # Delete this no from remaining_nums
            del remaining_nums[col_idx]
    return steps_no

def check_square(df, remaining_nums, axis1_idx, axis1_idx_name, steps_no):
    for axis2_idx in remaining_nums.copy().keys():
        steps_no += 1 
        square_axis1 = find_square_side(axis1_idx)
        square_axis2 = find_square_side(axis2_idx)
        if axis1_idx_name == 'row_idx':
            square_temp_values = pd.Series([df.loc[k, l] for k in square_axis1 for l in square_axis2])
        if axis1_idx_name == 'col_idx':
            square_temp_values = pd.Series([df.loc[k, l] for k in square_axis2 for l in square_axis1])
        remaining_nums[axis2_idx] = [i for i in remaining_nums[axis2_idx] if i not in square_temp_values.dropna().values]
        
        # If only one num left to be added
        if len(remaining_nums[axis2_idx]) == 1:
        # Add new number to our df
            if axis1_idx_name == 'row_idx':
                df.loc[axis1_idx, axis2_idx] = remaining_nums[axis2_idx][0]
            if axis1_idx_name == 'col_idx':
                df.loc[axis2_idx, axis1_idx] = remaining_nums[axis2_idx][0]
        # Delete this no from remaining_nums
            del remaining_nums[axis2_idx]    
    return steps_no
    

def check_row(df, remaining_nums, col_idx, steps_no):
    for row_idx in remaining_nums.copy().keys():
        steps_no += 1 
        row_temp = df.loc[row_idx, :]
        # Remove numbers (from remaining_nums) which occur in the row
        remaining_nums[row_idx] = [i for i in remaining_nums[row_idx] if i not in row_temp.dropna().values]
        
        # If only one num left to be added
        if len(remaining_nums[row_idx]) == 1:
            # Add new number to our df
            df.loc[row_idx, col_idx] = remaining_nums[row_idx][0]
            # Delete this no from remaining_nums
            del remaining_nums[row_idx]
    return steps_no
   

def check_column_for_square(df, remaining_nums, steps_no):
    col_idxs = [i[1] for i in list(remaining_nums.copy().keys())]
    for col_idx in set(col_idxs):
        steps_no += 1 
        col_temp = df.loc[:, col_idx]
        for key_temp in remaining_nums.copy().keys():
             if key_temp[1] == col_idx:
                remaining_nums[key_temp] = [i for i in remaining_nums[key_temp] if i not in col_temp.dropna().values]
    
        # If only one num left to be added in any cell of our square
        for key_temp, val_temp in remaining_nums.copy().items():
            if(len(val_temp)) == 1:
            # Add new number to our df
                df.loc[key_temp[0], key_temp[1]] = remaining_nums[key_temp][0]
            # Delete this no from remaining_nums
                del remaining_nums[key_temp]
    return steps_no                

def check_row_for_square(df, remaining_nums, steps_no):
    row_idxs = [i[0] for i in list(remaining_nums.copy().keys())]
    for row_idx in set(row_idxs):
        steps_no += 1 
        row_temp = df.loc[row_idx, :]
        for key_temp in remaining_nums.copy().keys():
             if key_temp[0] == row_idx:
                remaining_nums[key_temp] = [i for i in remaining_nums[key_temp] if i not in row_temp.dropna().values]
            
        # If only one num left to be added in any cell of our square
        for key_temp, val_temp in remaining_nums.copy().items():
            if(len(val_temp)) == 1:
            # Add new number to our df
                df.loc[key_temp[0], key_temp[1]] = remaining_nums[key_temp][0]
            # Delete this no from remaining_nums
                del remaining_nums[key_temp]
    return steps_no

def get_remaining_nums(axis_temp, blank_idx):
    blank_idx = list(axis_temp[axis_temp.isna()].index)
    remaining_nums_values = [i for i in numbers if i not in (axis_temp.dropna().values)]
    remaining_nums_values = [remaining_nums_values for i in range(len(blank_idx))]
    remaining_nums = dict(zip(blank_idx, remaining_nums_values))
    return remaining_nums

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
steps_no = 0

  


# Strategy 1) 'row - column - square'

df = df_initial.copy()
 
while df.isna().any().any():
    
    # Check a row
    for row_idx in df.index.copy():
        row_temp = df.loc[row_idx, :]
        blank_idx = list(row_temp[row_temp.isna()].index)
        remaining_nums = get_remaining_nums(row_temp, blank_idx)

        # If all numbers in the row are completed
        if len(remaining_nums) == 0:
            continue
        
        steps_no += 1 

        # If only one num left to be added
        if len(remaining_nums) == 1:
            # Add new number to our df
            df.loc[row_idx, blank_idx] = remaining_nums[blank_idx[0]][0]
            # Go to next row
            continue
        
        else:
            # Check a column
            steps_no = check_column(df, remaining_nums, row_idx, steps_no)
     
            # Check a square
            steps_no = check_square(df, remaining_nums, row_idx, 'row_idx', steps_no)

results_temp = {'strategy':'row_column_square', 'steps_no':steps_no, 'is_matrix_completed_correctly':check_corectness()}
results_comparison = results_comparison.append(results_temp, ignore_index=True)


# Strategy 2) 'row - square - column'

df = df_initial.copy()

while df.isna().any().any():
    
    # Check a row
    for row_idx in df.index.copy():
        row_temp = df.loc[row_idx, :]
        blank_idx = list(row_temp[row_temp.isna()].index)
        remaining_nums = get_remaining_nums(row_temp, blank_idx)

        # If all numbers in the row are completed
        if len(remaining_nums) == 0:
            continue
        
        steps_no += 1 

        # If only one num left to be added
        if len(remaining_nums) == 1:
            # Add new number to our df
            df.loc[row_idx, blank_idx] = remaining_nums[blank_idx[0]][0]
            # Go to next row
            continue
        
        else:
            # Check a square
            steps_no = check_square(df, remaining_nums, row_idx, 'row_idx', steps_no)

            # Check a column
            steps_no = check_column(df, remaining_nums, row_idx, steps_no)

results_temp = {'strategy':'row_square_column', 'steps_no':steps_no, 'is_matrix_completed_correctly':check_corectness()}
results_comparison = results_comparison.append(results_temp, ignore_index=True)


# Strategy 3) 'column - row - square'

df = df_initial.copy()

while df.isna().any().any():
    
    # Check a column
    for col_idx in df.index.copy():
        col_temp = df.loc[:, col_idx]
        blank_idx = list(col_temp[col_temp.isna()].index)
        remaining_nums = get_remaining_nums(col_temp, blank_idx)
    
        # If all numbers in the row are completed
        if len(remaining_nums) == 0:
            continue
        
        steps_no += 1 

        # If only one num left to be added
        if len(remaining_nums) == 1:
            # Add new number to our df
            df.loc[blank_idx, col_idx] = remaining_nums[blank_idx[0]][0]
            # Go to next row
            continue
        
        else:
            # Check a row
            steps_no = check_row(df, remaining_nums, col_idx, steps_no)
     
            # Check a square
            steps_no = check_square(df, remaining_nums, col_idx, 'col_idx', steps_no)

results_temp = {'strategy':'column_row_square', 'steps_no':steps_no, 'is_matrix_completed_correctly':check_corectness()}
results_comparison = results_comparison.append(results_temp, ignore_index=True)


# Strategy 4) 'column - square - row'

df = df_initial.copy()

while df.isna().any().any():
    
    # Check a column
    for col_idx in df.index.copy():
        col_temp = df.loc[:, col_idx]
        blank_idx = list(col_temp[col_temp.isna()].index)
        remaining_nums = get_remaining_nums(col_temp, blank_idx)

        steps_no += 1 

        # If all numbers in the row are completed
        if len(remaining_nums) == 0:
            continue
        
        # If only one num left to be added
        if len(remaining_nums) == 1:
            # Add new number to our df
            df.loc[blank_idx, col_idx] = remaining_nums[blank_idx[0]][0]
            # Go to next row
            continue
        
        else:    
            # Check a square
            steps_no = check_square(df, remaining_nums, col_idx, 'col_idx', steps_no)

            # Check a row
            steps_no = check_row(df, remaining_nums, col_idx, steps_no)

results_temp = {'strategy':'column_square_row', 'steps_no':steps_no, 'is_matrix_completed_correctly':check_corectness()}
results_comparison = results_comparison.append(results_temp, ignore_index=True)


# Strategy 5) 'square - row - column'

df = df_initial.copy()
square_start_row_idxs = [1, 4, 7]
square_start_col_idxs = [1, 4, 7]

while df.isna().any().any():

    # Check a square:
    for square_start_row_idx in square_start_row_idxs:
        for square_start_col_idx in square_start_col_idxs:
            blank_idxs = [(i, j) for i in range(square_start_row_idx, square_start_row_idx + 3) 
                        for j in range(square_start_col_idx, square_start_col_idx + 3) if math.isnan(df.loc[i, j])]
            square_temp_values = pd.Series([df.loc[i, j] for i in range(square_start_row_idx, square_start_row_idx + 3) 
                                for j in range(square_start_col_idx, square_start_col_idx + 3)])
            remaining_nums_values = [i for i in numbers if i not in (square_temp_values.dropna().values)]
            remaining_nums = {i: remaining_nums_values for i in blank_idxs}

            # If all numbers in the row are completed
            if len(remaining_nums) == 0:
                continue
            
            steps_no += 1 

            # If only one num left to be added
            if len(remaining_nums) == 1:
                # Add new number to our df
                df.loc[list(remaining_nums.keys())[0][0], list(remaining_nums.keys())[0][1]] = list(remaining_nums.values())[0][0] # brzydkie rozwiazanie !!!
                # Go to next square
                continue
    
            else:

                # Check rows in a square
                steps_no = check_row_for_square(df, remaining_nums, steps_no)

                # Check columns in a square
                steps_no = check_column_for_square(df, remaining_nums, steps_no)

results_temp = {'strategy':'square_row_column', 'steps_no':steps_no, 'is_matrix_completed_correctly':check_corectness()}
results_comparison = results_comparison.append(results_temp, ignore_index=True)
         

# Strategy 6) 'square - column - row'

df = df_initial.copy()
square_start_row_idxs = [1, 4, 7]
square_start_col_idxs = [1, 4, 7]

while df.isna().any().any():

    # Check a square:
    for square_start_row_idx in square_start_row_idxs:
        for square_start_col_idx in square_start_col_idxs:
            blank_idxs = [(i, j) for i in range(square_start_row_idx, square_start_row_idx + 3) 
                        for j in range(square_start_col_idx, square_start_col_idx + 3) if math.isnan(df.loc[i, j])]
            square_temp_values = pd.Series([df.loc[i, j] for i in range(square_start_row_idx, square_start_row_idx + 3) 
                                for j in range(square_start_col_idx, square_start_col_idx + 3)])
            remaining_nums_values = [i for i in numbers if i not in (square_temp_values.dropna().values)]
            remaining_nums = {i: remaining_nums_values for i in blank_idxs}

            # If all numbers in the row are completed
            if len(remaining_nums) == 0:
                continue
            
            steps_no += 1 

            # If only one num left to be added
            if len(remaining_nums) == 1:
                # Add new number to our df
                df.loc[list(remaining_nums.keys())[0][0], list(remaining_nums.keys())[0][1]] = list(remaining_nums.values())[0][0] # brzydkie rozwiazanie !!!
                # Go to next square
                continue
    
            else:
                
                # Check columns in a square
                steps_no = check_column_for_square(df, remaining_nums, steps_no)

                # Check rows in a square
                steps_no = check_row_for_square(df, remaining_nums, steps_no)
                
results_temp = {'strategy':'square_column_row', 'steps_no':steps_no, 'is_matrix_completed_correctly':check_corectness()}
results_comparison = results_comparison.append(results_temp, ignore_index=True)


print(results_comparison)

