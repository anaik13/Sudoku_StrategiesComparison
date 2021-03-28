import pandas as pd
import numpy as np

df_initial = pd.DataFrame([[5, np.NaN, np.NaN, np.NaN, 2, np.NaN, np.NaN, np.NaN, np.NaN],
                            [8, 3, np.NaN, np.NaN, 4, 9, np.NaN, np.NaN, 7],
                            [6, np.NaN, np.NaN, 5, np.NaN, 1, 9, 8, np.NaN],
                            [np.NaN, 5, np.NaN, 4, np.NaN, 6, 8, 3, 2],
                            [np.NaN, np.NaN, 4, np.NaN, 5, 3, np.NaN, np.NaN, np.NaN],
                            [np.NaN, np.NaN, 6, np.NaN, np.NaN, np.NaN, 7, np.NaN, 5],
                            [4, np.NaN, 5, np.NaN, 6, np.NaN, 3, np.NaN, np.NaN],
                            [7, np.NaN, np.NaN, 1, np.NaN, np.NaN, np.NaN, 2, 9],
                            [9, 2, 8, np.NaN, 7, 4, np.NaN, 6, np.NaN]], 
                            index=list(np.arange(1, 10)), columns=list(np.arange(1, 10))) 

#df_initial = pd.DataFrame([[5, 4, 9, 8, 2, 7, np.NaN, 1, 3],
#                            [8, 3, np.NaN, np.NaN, 4, 9, np.NaN, np.NaN, 7],
#                            [6, np.NaN, np.NaN, 5, np.NaN, 1, 9, 8, np.NaN],
#                            [np.NaN, 5, np.NaN, 4, np.NaN, 6, 8, 3, 2],
#                            [np.NaN, np.NaN, 4, np.NaN, 5, 3, np.NaN, np.NaN, np.NaN],
#                            [np.NaN, np.NaN, 6, np.NaN, np.NaN, np.NaN, 7, np.NaN, 5],
#                            [4, np.NaN, 5, np.NaN, 6, np.NaN, 3, np.NaN, np.NaN],
#                            [7, np.NaN, np.NaN, 1, np.NaN, np.NaN, np.NaN, 2, 9],
#                            [9, 2, 8, np.NaN, 7, 4, np.NaN, 6, np.NaN]], 
#                            index=list(np.arange(1, 10)), columns=list(np.arange(1, 10))) 

df_final = pd.DataFrame([[5, 4, 9, 8, 2, 7, 6, 1, 3],
                        [8, 3, 1, 6, 4, 9, 2, 5, 7],
                        [6, 7, 2, 5, 3, 1, 9, 8, 4],
                        [1, 5, 7, 4, 9, 6, 8, 3, 2],
                        [2, 8, 4, 7, 5, 3, 1, 9, 6],
                        [3, 9, 6, 2, 1, 8, 7, 4, 5],
                        [4, 1, 5, 9, 6, 2, 3, 7, 8],
                        [7, 6, 3, 1, 8, 5, 4, 2, 9],
                        [9, 2, 8, 3, 7, 4, 5, 6, 1]],
                        index=list(np.arange(1, 10)), columns=list(np.arange(1, 10)))

df = df_initial.copy()

def find_square_side(no):
    if no in [1, 2, 3]:
        return [1, 2, 3]
    if no in [4, 5, 6]:
        return [4, 5, 6]
    if no in [7, 8, 9]:
        return [7, 8, 9]
    else:
        raise Exception('Unexpected no in df')
        
def check_corectness():
    identical_column = []
    for i in df.columns:
        identical_column.append((df[i] == df_final[i]).all())
    is_df_correct = set(identical_column) == {True}
    return is_df_correct


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

  
# Strategy 1) 'row - column - square'

steps_no = 0

while df.isna().any().any():
    
    # Check a row
    for row_idx in df.index.copy():
        steps_no += 1 
        row_temp = df.loc[row_idx, :]
        blank_idx = list(row_temp[row_temp.isna()].index)
        remaining_nums_values = [i for i in numbers if i not in (row_temp.dropna().values)]
        remaining_nums_values = [remaining_nums_values for i in range(len(blank_idx))]
        remaining_nums = dict(zip(blank_idx, remaining_nums_values))

        # If all numbers in the row are completed
        if len(remaining_nums) == 0:
            continue
        
        # If only one num left to be added
        elif len(remaining_nums) == 1:
            # Add new number to our df
            df.loc[row_idx, blank_idx] = remaining_nums[blank_idx[0]][0]
            # Go to next row
            continue
        
        else:    
        # Check a column
            for col_idx in remaining_nums.copy().keys():
                steps_no += 1 
                col_temp_values = df.loc[:, col_idx]
                # Remove numbers (from remaining_nums) which occur in the column
                remaining_nums[col_idx] = [i for i in remaining_nums[col_idx] if i not in col_temp_values.dropna().values]
                
                # If only one num left to be added
                if len(remaining_nums[col_idx]) == 1:
                    # Add new number to our df
                    df.loc[row_idx, col_idx] = remaining_nums[col_idx][0]
                    # Delete this no from remaining_nums
                    del remaining_nums[col_idx]
     
            # Check a square
            for col_idx in remaining_nums.copy().keys():
                steps_no += 1 
                square_rows = find_square_side(row_idx)
                sqaure_cols  = find_square_side(col_idx)
                square_temp_values = pd.Series([df.loc[k, l] for k in square_rows for l in sqaure_cols])
                remaining_nums[col_idx] = [i for i in remaining_nums[col_idx] if i not in square_temp_values.dropna().values]
                
                # If only one num left to be added
                if len(remaining_nums[col_idx]) == 1:
                # Add new number to our df
                    df.loc[row_idx, col_idx] = remaining_nums[col_idx][0]
                # Delete this no from remaining_nums
                    del remaining_nums[col_idx]

print('Is Sudoku matrix completed correctly?', check_corectness())
print('Number of steps needed for finishing Sudoku game:', steps_no)


