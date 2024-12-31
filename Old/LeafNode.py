class LeafNode:
    def __init__(self, classification, count, elements):
        self.classification = classification
        self.count = count
        self.elements = elements

    def __str__(self):

        return(f"Classification: {self.classification}\nCount: {self.count}\nLabel Counts: {self.elements.tolist()}")
