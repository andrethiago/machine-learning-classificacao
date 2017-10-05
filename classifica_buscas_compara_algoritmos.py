from collections import Counter
# usando a lib pandas (http://pandas.pydata.org/)
import pandas as pd

df = pd.read_csv('busca2.csv') # df eh um data_frame

# teste inicial: home, busca, logado => comprou
# home,busca
# home,logado
# busca, logado
# busca

#pegando X e Y
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

# dummies permite variaveis categoricas em variaveis indicadoras
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

tamanho_de_treino = int(0.9 * len(Y))
treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

tamanho_de_teste = len(Y) - tamanho_de_treino
teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    diferencas = resultado - teste_marcacoes

    acertos = [d for d in diferencas if d == 0]
    total_acertos = len(acertos)
    total_elementos = len(teste_dados)
    taxa_acerto = 100.0 * total_acertos/total_elementos
    print("Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_acerto))


from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
fit_and_predict("MultinomialNB", modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier()
fit_and_predict("AdaBoostClassifier", modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

# a eficacia do algoritmo que chuta um unico valor
taxa_de_acerto_base = 100.0 * max(Counter(teste_marcacoes).itervalues()) / len(teste_marcacoes)
print("Taxa de acerto base : %f" % taxa_de_acerto_base)
print('Total de testes: %d' % len(teste_dados))
