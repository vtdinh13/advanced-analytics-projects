import sys
import networkx as nx
import json
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import LatentDirichletAllocation
import gensim.corpora as corpora
from sklearn.feature_extraction.text import CountVectorizer
import collections
pd.set_option('display.max_rows', 7000)
pd.set_option('display.max_columns', 7000)
pd.set_option('display.width', 7000)

def finished(): # Below functions are used for appending all relevant datasets and formatting it in a way amenable for Gephi
    #edges = pd.read_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/GraphSets/edges.xlsx')
    #nodes = pd.read_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/GraphSets/nodes.xlsx')
    #{'id': 112674, 'labels': ['streamer'], 'properties': {'followers': '148803', 'id': 'emeamasters', 'name': 'emeamasters', 'nr_streams': 1, 'views_avg': 1698, 'views_max': 1698, 'views_min': 1698}, 'type': 'node'}
    streamers = "/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/top10PercentStreamers/T5Pstreamers.json"
    games = "/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/top10PercentStreamers/T5Pgames.json"
    recommends = "/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/top10PercentStreamers/T5Precommends.json"
    streamers_df = pd.DataFrame({'id': [], 'label': [], 'name':[], 'views_avg':[], 'node_type':[]})
    games_df = pd.DataFrame({'id': [], 'label': [], 'name':[], 'views_avg':[], 'node_type':[]})
    recommends_df = pd.DataFrame({'source': [], 'target': [], 'id':[], 'name':[], 'edge_type':[]})
    plays = "/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/top10PercentStreamers/T5Pplays.json"
    plays_df = pd.DataFrame({'source': [], 'target': [], 'id':[], 'label':[], 'edge_type':[]})
    #{'r': {'id': 1468, 'start': 37694, 'end': 37667, 'label': 'recommends', 'properties': {}, 'type': 'relationship'}}
    with open(plays, "r", encoding="utf-8") as jsonl:
            for line in jsonl.readlines():

                json_line = json.loads(line)
                #print(json_line)
                #id = json_line['r']['id']
                #label = json_line['r']['labels'][0]
                #name = json_line['r']['properties']['id']
                #views_avg = 0#json_line['s']['properties']['views_avg']
                source = json_line['p']['start']
                target = json_line['p']['end']
                id = json_line['p']['id']
                label = json_line['p']['label']
                plays_df.loc[id] = [source, target, id, label, label]

    print(plays_df)
    plays_df.to_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/plays_df.xlsx')
    def irrelevant():
        nodes_dict = nodes.set_index('id').T.to_dict('list')
        edges = edges[edges[' label '] == 'plays']
        def match_all(id):
            found = []
            for idx, i in edges.iterrows():
                #print(i['source'])
                if i['source'] == id:
                    target_id = i[' target']
                    found.append(nodes_dict.get(target_id)[1])
            return found

        streamers_and_games = pd.DataFrame(nodes[['id', ' name']][nodes[' label'] == 'streamer'])
        streamers_and_games['nr_of_games'] = 0
        streamers_and_games['tfidf'] = 0
        streamers_and_games['games'] = ""

        for idx, i in streamers_and_games.iterrows():
            list_of_games = match_all(i['id'])
            nr_of_games = len(list_of_games)
            tf_idf = 0
            for i in list_of_games:
                idf = 1/int(games_dict.get(str(i)))
                tf_idf += (1/nr_of_games)*idf
            streamers_and_games.at[idx, 'tfidf'] = tf_idf
            streamers_and_games.at[idx, 'nr_of_games'] = nr_of_games
            streamers_and_games.at[idx, 'games'] = str(list_of_games)

        #streamers_and_games.to_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/tfidf_streamers.xlsx')


    # 6l62f9nl

# Below blocks of code compute Latent Dirichlet Allocation topic vectors and cluster-topic-probability distribution vectors for each cluster as calculated by modularity
edges = pd.DataFrame(pd.read_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/top10PercentStreamers/edge_set_full.xlsx'))
nodes = pd.DataFrame(pd.read_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/top10PercentStreamers/node_set_full.xlsx'))
print(nodes.columns)
games_dict = nodes[['id', 'name']][nodes['node_type'] == 'game']
games_dict = games_dict.set_index('id').T.to_dict('list')
#nodes = nodes[nodes['node_type'] == 'game']
nodes = nodes[nodes['node_type'] == 'streamer']
nodes.drop(['views_avg', 'label', 'node_type'],axis=1, inplace=True)
#nodes = nodes.set_index('id').T.to_dict('list')
edges = edges[edges['edge_type'] == 'plays']
edges.drop(['edge_type', 'id'], axis=1, inplace=True)
streamers_game_sets = edges.groupby('source')['target'].apply(list)
#game_counts = edges.groupby('target').count()
#tfidf = streamers_game_sets.apply(lambda x: sum([(1/len(x))*(1/game_counts.loc[i]['label']) for i in x]))
#tfidf.to_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/tfidf.xlsx')

# LDA on clusters
cluster_streamer_sets = nodes.groupby('modularity')['id'].apply(list)
cluster_sizes = pd.DataFrame(cluster_streamer_sets).apply(lambda x: len(x['id']), axis=1)
streamers_game_sets = pd.DataFrame(streamers_game_sets)
#print(cluster_streamer_sets)
#print(streamers_game_sets)
cluster_streamer_sets = pd.DataFrame(cluster_streamer_sets.apply(lambda x:
                                                    #pd.DataFrame(np.concatenate([np.array(streamers_game_sets.loc[i]['target']) for i in x])).to_dict()[0]
                                                    #list(pd.DataFrame(np.concatenate([np.array(streamers_game_sets.loc[i]['target']) for i in x])).to_dict()[0].items())
                                                   np.concatenate([np.array(streamers_game_sets.loc[i]['target']) for i in x])
                                                    ))



#test_cluster = cluster_streamer_sets.loc[0]['id']
def equalize_dict(doc):
    for i in games_dict.keys():
       # print(i)
        if i not in doc:
            doc[i] = 0
    doc = collections.OrderedDict(sorted(doc.items()))
    return doc

def cluster_to_dict(doc):
    cluster_dict = {}
    for i in doc:
        if i not in cluster_dict:
            cluster_dict[i] = 1
        else:
            cluster_dict[i] += 1
    cluster_dict = equalize_dict(cluster_dict)
    return cluster_dict

for idx, x in cluster_streamer_sets.iterrows():
    cluster_streamer_sets.at[idx, 'id'] = cluster_to_dict(x.loc['id'])


clusters = pd.DataFrame({})
for idx, x in cluster_streamer_sets.iterrows():
    clusters[idx] = x.loc['id']
clusters = clusters.transpose()


lda_clusters = LatentDirichletAllocation(n_components=10)
topic_clusters = lda_clusters.fit_transform(clusters)

# Topic distribution over games
topics = lda_clusters.components_

# Topic distribution per cluster
per_doc_topics = lda_clusters.transform(clusters)


def match_topic_with_games(doc):
    doc = np.array(doc)
    doc = doc/sum(doc)
    top_n_games = {}
    for d_game, d_key in zip(doc, games_dict.keys()):
            top_n_games[d_game] = games_dict[d_key][0]

    top_n_games = list(reversed(sorted(top_n_games.items())))#sorted(top_n_games.items())
    return top_n_games

topics_by_games = {}
for idx, t in enumerate(topics):
    topics_by_games[idx] = match_topic_with_games(t)[0:10]
#print(topics_by_games[2])
#print(topics_by_games[3])
#print(100*topic_clusters[17])

def get_cluster_topics(id, L):
    topic_dict = pd.DataFrame(topic_clusters[id]).to_dict()[0]
    topics = []
    for i in range(L):
        tid = max(topic_dict, key=topic_dict.get)
        topics.append(topics_by_games[tid])
        print('topic probability: ', topic_dict[tid])
        del topic_dict[tid]
    return topics
#print(get_cluster_topics(22, 1))

