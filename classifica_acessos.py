from dados import carregar_acessos

X,Y=carregar_acessos()

# pego os primeiros 90 itens dos arrays
treino_dados = X[:90]
treino_marcacoes = Y[:90]

# os testes sao as ultimas 9 linhas
teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_acertos = len(acertos)
total_elementos = len(teste_dados)
taxa_acerto = 100.0 * total_acertos/total_elementos
print(taxa_acerto)
print(total_elementos)
