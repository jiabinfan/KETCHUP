class TemplateHandler:
    def __init__(self, path, tokenizer):
        self.templates = []
        self.tokenizer = tokenizer
        with open(path, encoding='utf8')as fp:
            for line in fp:
                self.templates.append(line.strip())
    
    @property
    def size(self):
        return len(self.templates)

    def process(self, src_sents): # return a (T, B) list of str
        results = []
        for template in self.templates:
            temps = []
            for src in src_sents:
                temps.append(template.replace('[SRC]', src))
            results.append(temps)
        return results

class FewshotTemplateHandler:
    def __init__(self, path, tokenizer):
        self.templates = []
        self.tokenizer = tokenizer
        with open(path, encoding='utf8')as fp:
            self.templates=fp.read()
        
    
    @property
    def size(self):
        return len(self.templates)

    def process(self, src_sents): # return a (T, B) list of str
        results = []
        temps = []
        for src in src_sents:
            temps.append(self.templates.replace('[SRC]', src))
        results.append(temps)
        return results