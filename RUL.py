import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# Bibliotecas de Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# --- 1. CONFIGURAÇÃO DOS CAMINHOS ---
print("--- Iniciando Protótipo de Manutenção Preditiva ---")

# Define a pasta onde os arquivos estão (conforme sua imagem)
pasta_dados = 'CMAPSSData'
arquivos = {
    'treino': os.path.join(pasta_dados, 'train_FD001.txt'),
    'teste': os.path.join(pasta_dados, 'test_FD001.txt'),
    'rul': os.path.join(pasta_dados, 'RUL_FD001.txt')
}

# Verificação de segurança
for nome, caminho in arquivos.items():
    if not os.path.exists(caminho):
        print(f"ERRO: Não achei o arquivo '{caminho}'")
        print(f"Certifique-se que a pasta '{pasta_dados}' está junto com este script.")
        exit()

print("Arquivos encontrados. Carregando dados...")

# Carregamento
cols = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
cols += [f's_{i}' for i in range(1, 22)]

train = pd.read_csv(arquivos['treino'], sep=r'\s+', names=cols)
test = pd.read_csv(arquivos['teste'], sep=r'\s+', names=cols)
y_true = pd.read_csv(arquivos['rul'], sep=r'\s+', names=['RUL'])

# --- 2. PREPARAÇÃO DOS DADOS (FEATURE ENGINEERING) ---
def prepare_data(df, is_train=True):
    # Calcula RUL apenas para o treino (pois o teste não tem o ciclo final explicito)
    if is_train:
        max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycle.columns = ['unit_nr', 'max_life']
        df = df.merge(max_cycle, on='unit_nr', how='left')
        df['RUL'] = df['max_life'] - df['time_cycles']
        df.drop(columns=['max_life'], inplace=True)
    return df

train = prepare_data(train, is_train=True)

# Seleção de sensores úteis e Normalização
features_uteis = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
                  's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

scaler = MinMaxScaler(feature_range=(0, 1))
# Importante: O scaler aprende (fit) só no treino, mas transforma treino e teste
train[features_uteis] = scaler.fit_transform(train[features_uteis])
test[features_uteis] = scaler.transform(test[features_uteis])

print("Dados processados e normalizados.")

# --- 3. JANELAMENTO (Criação das Sequências para LSTM) ---
sequence_length = 30  # A IA olha 30 ciclos para trás

def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

print(f"Gerando janelas temporais de {sequence_length} ciclos...")

# Gera X_train e y_train
seq_gen = (list(gen_sequence(train[train['unit_nr']==id], sequence_length, features_uteis)) 
           for id in train['unit_nr'].unique())
X_train = np.concatenate(list(seq_gen)).astype(np.float32)

label_gen = (gen_labels(train[train['unit_nr']==id], sequence_length, ['RUL']) 
             for id in train['unit_nr'].unique())
y_train = np.concatenate(list(label_gen)).astype(np.float32)

print(f"Shape de entrada: {X_train.shape}")

# --- 4. CONSTRUÇÃO E TREINAMENTO DA IA (LSTM) ---
print("\n--- Iniciando Treinamento da Rede Neural ---")

model = Sequential()
# Camada 1: LSTM com 100 neurônios (return_sequences=True pois tem outra LSTM depois)
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2)) # Evita decorar os dados (Overfitting)

# Camada 2: LSTM com 50 neurônios
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))

# Camada de Saída: 1 neurônio (prevê o valor da RUL)
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Treinamento (Usando 20% dos dados para validar se está aprendendo mesmo)
history = model.fit(X_train, y_train, epochs=60, batch_size=512, validation_split=0.2, verbose=1)

print("Treinamento concluído!")

# --- 5. RESULTADOS E GRÁFICO ---
# Plota a curva de aprendizado
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Erro no Treino (Loss)')
plt.plot(history.history['val_loss'], label='Erro na Validação (Val Loss)')
plt.title('Curva de Aprendizado do Modelo LSTM')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.xlabel('Épocas')
plt.legend()
plt.show()

# Salva o modelo para usar depois
model.save('modelo_rul_lstm.h5')
print("Modelo salvo como 'modelo_rul_lstm.h5'")