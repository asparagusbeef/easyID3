import pandas as pd
import math
import numpy as np
from node import Node

class ID3DecisionTreeClassifier:
    def __init__(self):
        self.root = None

    def _preprocess_data(self, X, y):
        """
        Order the data after removing columns with unique values.

        Args:
            X (DataFrame): input DataFrame.
            y (Series): the column to be excluded from feature set.

        Returns:
            Tuple[DataFrame, List[str]]: The ordered DataFrame and a list of features.
        """
        data = pd.concat([X, y], axis=1)
        data = data.astype('str')
        for col in X.columns:
            if len(data[col]) == len(pd.unique(data[col])):
                data = data.drop(col, axis=1)
        features = [feat for feat in data.columns if feat != y.name]
        return data, features

    def _calculate_entropy(self, data, target):
        """
        Calculate the entropy of the given data based on the answer column.

        Args:
            data (DataFrame): The input DataFrame.
            target (str): The column in the DataFrame to calculate entropy for.

        Returns:
            float: The calculated entropy.
        """
        answers = data[target].unique()
        probabilities = [
            sum(data[target] == answer) / len(data[target]) for answer in answers
        ]

        entropy = -sum(p * math.log(p, 2) for p in probabilities)
        return entropy


    def _info_gain(self, data, feature, target):
        """
        Calculate the information gain based on entropy.

        Args:
            data (DataFrame): The input DataFrame.
            feature (str): The feature to calculate information gain for.
            target (str): The column in the DataFrame to calculate entropy for.

        Returns:
            float: The calculated information gain.
        """
        unique_values = np.unique(data[feature])
        gain = self._calculate_entropy(data, target)
        
        for value in unique_values:
            sub_data = data[data[feature] == value]
            sub_entropy = self._calculate_entropy(sub_data, target)
            gain -= (len(sub_data) / len(data)) * sub_entropy

        return gain


    def _id3(self, data, features, target):
        """
        Implement the ID3 algorithm for decision tree learning.

        Args:
            data (DataFrame): The input DataFrame.
            features (List[str]): List of features in the DataFrame.
            target (str): The target variable column.

        Returns:
            Node: Root of the constructed decision tree.
        """
        # Get unique answers
        answers = data[target].unique()

        # Create root node
        root = Node()

        # Identify feature with max information gain
        max_gain = 0
        max_feature = ''
        for feature in features:
            gain = self._info_gain(data, feature, target)
            if gain > max_gain:
                max_gain = gain
                max_feature = feature
        root.value = max_feature
        root.is_category = True

        # If no information gain, create an uncertain leaf node
        if max_gain == 0:
            root.is_leaf = True
            root.target = target
            for answer in answers:
                count = sum(data[target]==answer)
                root.answers_count[answer] = count
            return root

        # If a feature was identified, split data on unique values of that feature
        if max_feature:
            unique_values = np.unique(data[max_feature])
            for value in unique_values:
                sub_data = data[data[max_feature] == value]
                
                # If entropy is zero, create a leaf node
                if self._calculate_entropy(sub_data, target) == 0.0:
                    new_node = Node()
                    new_node.is_leaf = True
                    new_node.value = value
                    new_node.prediction = np.unique(sub_data[target])
                    new_node.target = target
                    new_node.prediction_id = np.where(np.sort(answers)==new_node.prediction[0])[0][0]
                    root.children.append(new_node)
                else:
                    # If entropy is not zero, recurse on the subset of data
                    dummy_node = Node()
                    dummy_node.value = value
                    new_features = features.copy()
                    new_features.remove(max_feature)
                    child = self._id3(sub_data, new_features, target)
                    dummy_node.children.append(child)
                    dummy_node.target = target
                    answers = data[target].unique()
                    for answer in answers:
                        count = sum(sub_data[target]==answer)
                        dummy_node.answers_count[answer] = count
                    root.children.append(dummy_node)
            
        return root

    def fit(self, X, y):
        """
        Train the decision tree model.

        Args:
            X (DataFrame): The DataFrame of features to train on.
            y (Series): The target variable.

        Returns:
            self: Returns an instance of the fitted model.
        """
        data, features = self._preprocess_data(X, y)
        self.root = self._id3(data, features, y.name)
        return self


    def predict_instance(self, instance, node):
        """
        Predict the target variable of a given instance.

        Args:
            instance (Series): A pandas Series representing a single data instance.
            node (Node): The decision node from which to start the prediction.

        Returns:
            str: The predicted target variable value.
            None: If no match is found in the decision tree for the instance.
        """
        if node.is_leaf:
            if node.answers_count:
                return f"Uncertain: {node.answers_count}"
            return node.prediction[0]
        
        attribute = node.value
        for child in node.children:
            # If the child node is a category, predict using the grandchild node
            if child.is_category:
                for grandchild in node.children:
                    return self.predict_instance(instance, grandchild)

            # If the child node's value matches the instance's attribute, use this child node for prediction
            if child.value == instance[attribute]:
                return self.predict_instance(instance, child)
        
        # Return None if no match found in the decision tree for the instance
        return ["No Match!"]


    def predict(self, data):
        """
        Predict the target variable for each instance in the given data.

        Args:
            data (DataFrame): A DataFrame containing instances to predict.

        Returns:
            Series: A pandas Series of predicted target variable values.
        """
        data = data.astype("str")
        predictions = data.apply(lambda instance: self.predict_instance(instance, self.root), axis=1)
        return predictions



    def print_tree(self, root=None, depth=0):
        """
        Prints the decision tree in a readable format with color-coding.

        Args:
            root (Node, optional): The root node of the tree. Defaults to the root of the tree.
            depth (int, optional): The depth level of the current node. Defaults to 0.
        """
        if root is None:
            root = self.root

        tree_colors = [('\x1b[30m', '\x1b[46m'), ('\x1b[37m', '\x1b[45m'),
                    ('\x1b[37m', '\x1b[44m'), ('\x1b[30m', '\x1b[43m')]
        ans_colors = [('\x1b[37m', '\x1b[41m'), ('\x1b[30m', '\x1b[42m')]

        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        FORE_WHITE = '\x1b[37m'
        BACK_BLACK = '\x1b[40m'
        RESET_ALL = '\x1b[0m'

        for _ in range(depth):
            print("\t", end="")

        # Use color-cycling according to the depth of the tree
        tree_color = tree_colors[depth % len(tree_colors)]
        print(f'{tree_color[0]}{tree_color[1]}{root.value}{RESET_ALL}', end="")

        if root.is_leaf:
            if root.prediction:
                ans_color = ans_colors[root.prediction_id % len(ans_colors)]
                print(" -> ", f'{BOLD}{UNDERLINE}{ans_color[0]}{ans_color[1]}'
                    f'[{root.target}: {root.prediction[0]}]{RESET_ALL}')
            else:  # Uncertain leaf
                print(" -> ", f'{BOLD}{UNDERLINE}{FORE_WHITE}{BACK_BLACK}Uncertain '
                    f'{dict(sorted(root.answers_count.items()))}{RESET_ALL}')

        print()

        for child in root.children:
            self.print_tree(child, depth + 1)


