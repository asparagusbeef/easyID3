# easyID3
Simple and easy to use ID3 tree build and print

Utilizes pandas, math, numpy and colorama. 
Easy textual print of the categorical-only ID3 adjusted for any number of answers and will clean data of perfectly unique columns (such as an index column).
example code:

tree = ID3(df, "Purchased?")
print(tree)

As of now there are four rotating colors for middle nodes (cyan magenta blue yellow) and two (red and green) for the final nodes. future release will include functions for controlling the print color scheme.
