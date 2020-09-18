

class DataAugmentation(object):
    
    def __init__(self):
        self.tags = []
        self.additonal_patterns = {}       
        
    def load_tags(self, tags):
        self.tags = tags
        
    def get_similar_patterns(self, pattern):
        pattern_tokens = pattern.split()
        similar_patterns = []
        # TODO: Add augmentation methods
        if "file" in pattern and 'the' not in pattern_tokens and 'a' not in pattern_tokens:
            similar_patterns.append(pattern.replace(" file ", " a file ", 1))
            similar_patterns.append(pattern.replace(" file ", " the file ", 1))
        if "document" in pattern and 'the' not in pattern_tokens and 'a' not in pattern_tokens:
            similar_patterns.append(pattern.replace(" document ", " a document ", 1))
            similar_patterns.append(pattern.replace(" document ", " the document ", 1))
        #similar_patterns.append(pattern)
        return similar_patterns
    
    
def main():
    pattern = 'share file <filekeyword> with meeting.'

    da = DataAugmentation()
    print(da.get_similar_patterns(pattern))

if __name__ == '__main__':
    main()