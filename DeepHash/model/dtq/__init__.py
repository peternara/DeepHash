from .dtq import DTQ
from .util import Dataset

def train(train_img, database_img, query_img, config):
    model = DTQ(config)
    img_database = Dataset(database_img, config.output_dim, config.subspace * config.subcenter)
    img_query = Dataset(query_img, config.output_dim, config.subspace * config.subcenter)
    img_train = Dataset(train_img, config.output_dim, config.subspace * config.subcenter)
    
    #print img_train.output[0] # init- all zero
    #print img_train.output.shape # (5000, 64)
    #print img_train.output[0].shape #(64,)
    #print img_train.codes[0] # init- all zero
    #print img_train.codes.shape # (5000, 1024) > 256x4 = 1024
    #print img_train.label[0] # [0 0 0 0 0 1 0 0 0 0]
    #print img_train.label.shape # (5000, 10)

    model.train_cq(img_train, img_query, img_database, config.R)
    return model.save_dir


def validation(database_img, query_img, config):
    model = DTQ(config)
    img_database = Dataset(database_img, config.output_dim, config.subspace * config.subcenter)
    img_query = Dataset(query_img, config.output_dim, config.subspace * config.subcenter)
    return model.validation(img_query, img_database, config.R)
