# gordinho; perna curta; late;
porco1 =    [1,1,0];
porco2 =    [1,1,0];
porco3 =    [1,1,0];
cachorro1 = [1,1,1];
cachorro2 = [0,1,1];
cachorro3 = [0,1,1];

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3];
marcacoes = [1,1,1,-1,-1,-1];

misterioso1 = [1,1,1];
misterioso2 = [1,0,0];
misterioso3 = [0,0,1];
testes = [misterioso1, misterioso2, misterioso3]
marcacoes_esperadas = [-1,1,-1]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)
resultado = modelo.predict(testes)
print(resultado)
diferencas = resultado - marcacoes_esperadas

# calcula quantas vezes houve erro no algoritmo de classificacao
acertos = [d for d in diferencas if d == 0]
total_acertos = len(acertos)
total_elementos = len(testes)
taxa_acerto = total_acertos/total_elementos * 100.0
print(taxa_acerto)
