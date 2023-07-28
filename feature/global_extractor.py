from third_party.global_feature.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from os.path import join

class Global_Extractors():
    def __init__(self, root,config):
        self.root=root
        self.extractor = config

    def netvlad(self, content):
        return NetVladFeatureExtractor(join(self.root,content['ckpt_path']), arch=content['arch'],
         num_clusters=content['num_clusters'],
         pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])

    def vlad(self, contend):
        pass

    def bovw(self, contend):
        pass

    def get(self):
        for extractor, content in self.extractor.items():
            print(extractor, content)
            if extractor == 'netvlad':
                return self.netvlad(content)
            if extractor == 'vlad':
                pass
            if extractor == 'bovw':
                pass