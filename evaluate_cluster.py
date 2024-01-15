from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
from munkres import Munkres, print_matrix
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np

class linkpred_metrics():
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_roc_score(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(feas['adj_orig'][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(feas['adj_orig'][e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score, emb


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, adjscore



#kmeans
#embeddings_file = 'cora/cora.id_view2_0.7.128d.txt'
#embeddings_file = 'cora/cora.id_view2_0.7.64d.txt'
#embeddings_file = 'cora/cora.id_view2_0.7.128d_good2.txt'
#content_file = 'cora/cora.content'






#embeddings_file = 'citeseer/citeseer.id_view2_0.7.128d.txt'
#embeddings_file = 'citeseer/citeseer.id_view2_0.7.64d.txt'
#embeddings_file = 'citeseer/citeseer.id_view2_0.7.32d.txt'
#embeddings_file = 'citeseer/citeseer.id_view2_0.7.16d.txt'
embeddings_file = 'citeseer/citeseer.id_view2_0.7.64d.txt'
content_file = 'citeseer/citeseer.content'




#embeddings_file = 'pubmed/pubmed.id_view2_0.7.64d.txt'
#content_file = 'pubmed/pubmed.content'



embeddings = list()

with open(embeddings_file) as f_emb:
    for line in f_emb.readlines():
        values = line.split('\t')
        emb = list()
        for i in range(1,len(values)):
            emb.append(values[i].strip())
        embeddings.append(emb)

#print(embeddings[0])

set_lbl = set()
true_labels = list()
true_labels_idx = list()
lbl2idx = {}
cnt = 0
with open(content_file) as f_cnt:
    for line in f_cnt.readlines():
        values = line.split('\t')
        lbl = values[len(values)-1].strip()
        set_lbl.add(lbl)
        true_labels.append(lbl)
        if lbl not in lbl2idx:
            lbl2idx[lbl] = cnt
            cnt += 1


#print(set_lbl)
#print(len(set_lbl))
#print(len(true_labels))
#print(lbl2idx)

for l in true_labels:
    idx = lbl2idx[l]
    true_labels_idx.append(idx)



n_clusters = len(set_lbl) #7 for cora 6 for citeseer


print(n_clusters)

#for visualization
#with open('index_clusters_labels','w') as f_icl:
#    for i in range(len(true_labels_idx)):
#        f_icl.write(str(i)+'\t'+str(true_labels_idx[i])+'\n')








array_embeddings = np.array(embeddings)
#print(array_embeddings)

kmeans = KMeans(n_clusters=n_clusters, n_init = 20, max_iter=300,algorithm='auto').fit(array_embeddings)
predict_labels = kmeans.predict(array_embeddings)


#kmeans = SpectralClustering(n_clusters=n_clusters,affinity='nearest_neighbors')
#predict_labels = kmeans.fit_predict(array_embeddings)



#print(predict_labels)
#print(kmeans.labels_)

a = list()
a.append(1)
a.append(2)
a.append(3)
a.append(4)
a.append(4)
a.append(4)

b = list()
b.append(1)
b.append(2)
b.append(3)
b.append(4)
b.append(3)
b.append(3)

cm = clustering_metrics(true_labels_idx,predict_labels)
cm.evaluationClusterModelFromLabel()