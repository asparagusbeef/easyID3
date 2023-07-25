# easyID3
Python implementation of the original ID3 algorithm by Ross Quinlan. Works only on categorical data, and outputs a leaf only at perfect homogenouity. 

```
import pandas as pd
from easyID3 import ID3DecisionTreeClassifier

df = pd.read_csv('my_categorical_data.csv')
tree = ID3DecisionTreeClassifier()
tree.fit(df)
tree.predict(df)
tree.print_tree()
```


