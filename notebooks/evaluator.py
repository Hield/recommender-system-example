import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm_notebook as tqdm

class Evaluator():
    def __init__(self, k=10, training_ratings=None, testing_ratings=None, book_sim=None, novelty_scores=None):
        self.k = k
        self.book_sim = book_sim
        self.novelty_scores = novelty_scores
        if training_ratings is not None:
            self.training_ratings = training_ratings
            self.num_users = len(self.training_ratings.user_id.unique())
            self.num_books = len(self.training_ratings.book_id.unique())
        if testing_ratings is not None:
            self.testing_ratings = testing_ratings
            self.testing_idx = {}
            for user_id in tqdm(testing_ratings.user_id.unique()):
                self.testing_idx[user_id] = testing_ratings[testing_ratings.user_id==user_id].book_id.values
        self.result = {}
    
    def _average_precision(self, pred, truth):
        in_arr = np.in1d(pred, truth)
        score = 0.0
        num_hits = 0.0
        for idx, correct in enumerate(in_arr):
            if correct:
                num_hits += 1
                score += num_hits / (idx + 1)
        return score / min(len(truth), self.k)
    
    def _novelty_score(self, pred):
        # Recommend the top 10 books in novelty score results in ~11
        # Crop the score to 10.0 since it won't change anything and make the score range nicer
        return min(self.novelty_scores.loc[pred].novelty_score.mean(), 10.0)
    
    def _diversity_score(self, pred):
        matrix = self.book_sim.loc[pred, pred].values
        ils = matrix[np.triu_indices(len(pred), k=1)].mean()
        return (1 - ils) * 10
    
    def _personalization_score(self, preds, user_ids, book_ids):
        if len(user_ids) > 3000:
            np.random.seed(42)
            user_ids = np.random.permutation(user_ids)[:3000]
        df = pd.DataFrame(
            data=np.zeros([len(user_ids), len(book_ids)]),
            index=user_ids,
            columns=book_ids
        )
        for user_id in user_ids:
            df.loc[user_id, preds[user_id]] = 1

        matrix = sp.csr_matrix(df.values)

        #calculate similarity for every user's recommendation list
        similarity = cosine_similarity(X=matrix, dense_output=False)

        #get indicies for upper right triangle w/o diagonal
        upper_right = np.triu_indices(similarity.shape[0], k=1)

        #calculate average similarity
        personalization = np.mean(similarity[upper_right])
        
        return (1 - personalization) * 10
    
    def evaluate(self, model):
        print("Calculating recommendations:")
        if len(model.preds) == 0:
            model.fit(self.training_ratings)
        preds = model.all_recommendation()
        user_ids = list(preds.keys())
        book_ids = np.unique(np.concatenate(list(preds.values())))
        ap_sum = 0
        nov_score_sum = 0
        div_score_sum = 0
        print("Calculating metrics:")
        for user_id in tqdm(preds.keys()):
            pred = preds[user_id]
            truth = self.testing_idx[user_id]
            ap_sum += self._average_precision(pred, truth)
            nov_score_sum += self._novelty_score(pred)
            div_score_sum += self._diversity_score(pred)
        
        self.result[model.name] = {}
        self.result[model.name]['Mean Average Precision'] = "%.2f%%" % (ap_sum / self.num_users * 100)
        self.result[model.name]['Coverage'] = "%.2f%%" % (len(book_ids) / self.num_books * 100)
        self.result[model.name]['Novelty Score'] = "%.2f" % (nov_score_sum / self.num_users)
        self.result[model.name]['Diversity Score'] = "%.2f" % (div_score_sum / self.num_users)
        self.result[model.name]['Personalization Score'] = "%.2f" % self._personalization_score(preds, user_ids, book_ids)
        
    def print_result(self):
        print(pd.DataFrame(self.result).loc[['Mean Average Precision', 'Coverage', 'Novelty Score', 'Diversity Score', 'Personalization Score']])