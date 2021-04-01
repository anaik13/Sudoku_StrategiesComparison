The goal of the project is to make a comparison between below described strategies of playing Sudoku game.

Tested Strategies:
1) row-column-square <br/>
(A Sudoku matrix is completed at first by looking at rows. If there is only one missing number in a row, we add it to the Sudoku matrix and go to the next row.  <br/> 
If there are more missing numbers in a row, then we look at numbers in columns. If information about numbers gathered from columns is enough to complete a row, we do it and go to the next row. <br/>
If not, we look at numbers in squares.)
2) row-square-column (analogously to 1st strategy)
3) column-row-square (analogously to 1st strategy)
4) column-square-row (analogously to 1st strategy)
5) square-row-column (analogously to 1st strategy)
6) square-column-row (analogously to 1st strategy)

The performance of strategies is based on number of steps which needs to be taken to complete a Sudoku matrix. <br/>
Comparison of strategies is presented in a dataframe 'results_comparison':

|strategy|steps_no|
|---|---|
|row_column_square|551|
|row_square_column|1233|
|column_row_square|1667|
|column_square_row|2192|
|square_row_column|2599|
|square_column_row|2950|

 <br/> <br/>
TODO:
- test strategies on n Sudoku matrixes, not on 1 Sudoku matrix
