from numpy import dot
from numpy.linalg import norm




#filename = 'cora/cora.cites'
#filename_content = 'cora/cora.content'
#filename_v2 = 'cora/cora.id_view2_0.7'



filename_content = 'citeseer/citeseer.content'
filename_v2 = 'citeseer/citeseer.id_view2_0.8'


#filename_content = 'pubmed/pubmed.content'
#filename_v2 = 'pubmed/pubmed.id_view2_0.7'


#filename_content = 'WebKB/webkb.content'
#filename_v2 = 'WebKB/webkb.id_view2_0.5'


#filename_content = 'TerrorAttack/terrorist_attack.nodes'
#filename_v2 = 'TerrorAttack/terrorist_attack.id_view2_0.8'

node2id = {}

f1 = open(filename_content)
cnt = 0
for line in f1:
    values = line.split('\t')
    #print(values)
    l = values[0].strip()
    #r = values[1].strip()
    if l not in node2id:
        node2id[l] = cnt
        cnt += 1
    #if r not in node2id:
    #    node2id[r] = cnt
    #    cnt += 1

f1.close()

#print(cnt)
#print(len(node2id))
#exit(0)

#print(node2id['198191'])
#print(node2id)
#exit(0)


id_features = {}
f3 = open(filename_content)
for line in f3:
    values = line.split('\t')
    nd_str = values[0].strip()
    #if nd_str =='198191':
    #    print(nd_str)
    #    print(node2id[nd_str])
    nd_id = node2id[nd_str]
    #if nd_id == 305:
    #    print(nd_id)
    features_str = values[1:len(values)-1] #abandon last column, the class label
    features = list(map(float, features_str))
    id_features[nd_id] = features
    #print(features)
    #print(len(features))
    #print(values)

f3.close()

#print(len(id_features))
#exit(0)



f4 = open(filename_v2,'w')
num_edges_v2 = 0
for i in range(len(id_features)):
#for i in range(10):
    for j in range(i+1, len(id_features)):
        a = id_features[i]
        b = id_features[j]
        cos_sim = dot(a, b) / (norm(a)*norm(b))
        if(cos_sim >= 0.8):
            num_edges_v2 += 1
            f4.write(str(i)+'\t'+str(j)+'\n')

    print(i,'/',len(id_features),': ', num_edges_v2)

f4.close()
print(num_edges_v2)
