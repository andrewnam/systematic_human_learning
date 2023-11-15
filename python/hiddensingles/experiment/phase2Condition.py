class Phase2Condition:
    def __init__(self, house_type, house_index, cell_index, digit_set):
        self.house_type = house_type
        self.house_index = house_index
        self.cell_index = cell_index
        self.digit_set = digit_set

    def __repr__(self):
        s = "ht" if self.house_type else ""
        s += "hi" if self.house_index else ""
        s += "ci" if self.cell_index else ""
        s += "ds" if self.digit_set else ""
        return s