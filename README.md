# easyID3
Simple and easy to use ID3 tree build and print

requires pandas, math, numpy and colorama.
Easy textual print of ID3 for any number of possible answers. 
Supports dataframes with only categorical data (strings only, case-sensitive).
Will also clean data of perfectly unique columns (such as an index column).
As of now there are four rotating colors for middle nodes (cyan magenta blue yellow) and two (red and green) for the final nodes. Future release will include functions for controlling the print color scheme.

example code:

tree = ID3(df, "Purchased?")
print(tree)



