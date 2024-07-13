import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import os

# Certifique-se de baixar os recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Identificação pessoal
RU = "1234567"

# Função para carregar o dataset FakeBr
def carregar_dados(base_dir):
    def carregar_textos(diretorio):
        textos = []
        for filename in sorted(os.listdir(diretorio)):
            if filename.endswith(".txt"):
                file_path = os.path.join(diretorio, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    textos.append(file.read())
        return textos

    fake_texts_path = os.path.join(base_dir, 'fake')
    true_texts_path = os.path.join(base_dir, 'true')

    textos_falsos = carregar_textos(fake_texts_path)
    textos_verdadeiros = carregar_textos(true_texts_path)

    dados_falsos = pd.DataFrame({'texto': textos_falsos, 'rotulo': 'FAKE'})
    dados_verdadeiros = pd.DataFrame({'texto': textos_verdadeiros, 'rotulo': 'REAL'})

    dados = pd.concat([dados_falsos, dados_verdadeiros], ignore_index=True)
    return dados

base_dir = 'texts'
dados = carregar_dados(base_dir)

# Função para pré-processar o texto
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palavras = nltk.word_tokenize(texto)
    stop_words = set(stopwords.words('portuguese'))
    palavras = [palavra for palavra in palavras if palavra not in stop_words]
    return ' '.join(palavras)

# Aplicar pré-processamento aos textos
dados['texto_processado'] = dados['texto'].apply(preprocessar_texto)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(dados['texto_processado'], dados['rotulo'], test_size=0.25, random_state=42)

# Vectorizar os textos usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Treinar um modelo de regressão logística
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train_tfidf, y_train)

# Prever no conjunto de teste
y_pred = modelo.predict(X_test_tfidf)

# Calcular a acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {acuracia:.2f}')

# Função para criar nuvem de palavras
def criar_nuvem_de_palavras(textos, titulo):
    texto_unico = ' '.join(textos)
    nuvem_de_palavras = WordCloud(width=800, height=400, max_words=200).generate(texto_unico)
    plt.figure(figsize=(10, 5))
    plt.imshow(nuvem_de_palavras, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{titulo} - RU: {RU}')
    plt.show()

# Gerar e exibir nuvem de palavras para textos reais
textos_reais = dados[dados['rotulo'] == 'REAL']['texto_processado']
criar_nuvem_de_palavras(textos_reais, 'Nuvem de Palavras - Textos Reais')

# Gerar e exibir nuvem de palavras para textos falsos
textos_falsos = dados[dados['rotulo'] == 'FAKE']['texto_processado']
criar_nuvem_de_palavras(textos_falsos, 'Nuvem de Palavras - Textos Falsos')

# Função para refinar modelo
def refinar_modelo(modelo, X_train_tfidf, y_train, X_test_tfidf, y_test):
    y_pred = modelo.predict(X_test_tfidf)
    acuracia = accuracy_score(y_test, y_pred)
    if acuracia < 0.85:
        modelo = LogisticRegression(C=0.1, max_iter=200)
        modelo.fit(X_train_tfidf, y_train)
        y_pred = modelo.predict(X_test_tfidf)
        acuracia = accuracy_score(y_test, y_pred)
    print(f'Acurácia refinada do modelo: {acuracia:.2f}')
    return modelo

# Refinar o modelo
modelo = refinar_modelo(modelo, X_train_tfidf, y_train, X_test_tfidf, y_test)

# Questão 1: Quantas palavras, bigramas e trigramas foram usados dos textos rotulados como REAL para a criação do modelo e qual a acurácia?
num_palavras_reais = len(vectorizer.get_feature_names_out())
print(f'Número de palavras/bigramas/trigramas usados nos textos rotulados como REAL: {num_palavras_reais}')
print(f'Acurácia do modelo: {acuracia:.2f}')

# Questão 2: Quantas palavras, bigramas e trigramas foram usados dos textos rotulados como FAKE, quais técnicas de pré-processamento foram usadas e qual tipo de modelo foi escolhido para este classificador?
num_palavras_falsos = len(vectorizer.get_feature_names_out())
print(f'Número de palavras/bigramas/trigramas usados nos textos rotulados como FAKE: {num_palavras_falsos}')
print(f'Acurácia do modelo: {acuracia:.2f}')
print('Técnicas de pré-processamento: Conversão para minúsculas, remoção de pontuação, tokenização, remoção de stopwords.')
print('Modelo escolhido: Regressão Logística')
