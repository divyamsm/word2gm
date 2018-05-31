import pandas as pd
import numpy as np
from ggplot import *
from word2gm_loader import Word2GM
from quantitative_eval import *
def show_nearest_neighbors(self, idx_or_word, cl=0, num_nns=20, plot=True, verbose=False):
        idx = idx_or_word
        idx_or_word = "b'"+idx_or_word+"'"
        if idx_or_word in self.word2id:
            print("entered block")
            idx = self.word2id[idx_or_word]
        dist = np.dot(self.mus_n, self.mus_n[idx*self.num_mixtures + cl])
        highsim_idxs = dist.argsort()[::-1]
        # select top num_nns (linear) indices with the highest cosine similarity
        highsim_idxs = highsim_idxs[:num_nns]
        dist_val = dist[highsim_idxs]
        words = self.idxs2words(highsim_idxs)
        var_val = np.array([self.detA[_idx] for _idx in highsim_idxs])
        # plot all the words
        if plot:
            df = pd.DataFrame({'logvar' : var_val ,'sim': dist_val , 'text': words})
            #df = pd.DataFrame()
            #print(len(words), len(dist_val),len(var_val))
            #df['text'] = words
            #df['sim'] = dist_val
            #df['logvar'] = var_val
            mix = self.mixture[idx, cl]
            plot = (ggplot(aes(x='sim', y='logvar', label='text'), data=df)
                    + geom_point(size=5)
                    + geom_text(size=10)
                    + ggtitle("Neighbors of [{}:{}] with mixture probability {:.4g}".format(self.id2word[idx], cl, mix)))
            fig = plot.draw()

def main():
    text8_model_dir = 'exps/modelfiles/t8-2s-e10-v05-lr05d-mc100-ss5-nwout-adg-win10'
    w2gm_text8_2s = Word2GM(text8_model_dir)
    w2gm_text8_2s.visualize_embeddings()
    w2gm_text8_2s.show_nearest_neighbors('rock',0)
    w2gm_text8_2s.show_nearest_neighbors('rock',1)
if __name__ == '__main__':
    main()
