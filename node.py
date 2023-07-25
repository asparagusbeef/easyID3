class Node:
    def __init__(self):
        self.children = []
        self.value = ''
        self.is_leaf = False
        self.prediction = ''
        self.target = ''
        self.prediction_id = 999999
        self.answers_count = {}
        self.is_category = False