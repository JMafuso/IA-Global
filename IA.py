import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Simulação de dados para múltiplos usuários e seus aparelhos
np.random.seed(42)
usuarios = [1, 2, 3, 4, 5]  # ID dos usuários
aparelhos = ['Geladeira', 'Ar Condicionado', 'Televisão', 'Microondas', 'Lâmpada']
tipos = ['Essencial', 'Conforto', 'Entretenimento', 'Cozinha', 'Iluminação']
tempos_uso = np.random.uniform(1, 5, len(aparelhos)) * 30  # horas por mês

def simular_dados(meses, usuarios):
    # Simulando consumo de energia e gastos para n meses e múltiplos usuários
    data = []
    for mes in range(meses):
        for usuario in usuarios:
            tarifa = np.random.uniform(0.3, 0.7)  # Tarifa variável por mês
            consumo_energia = np.random.uniform(0.1, 1.5, len(aparelhos)) * tempos_uso
            gasto = consumo_energia * tarifa
            mes_data = {
                'usuario_id': usuario,
                'mes': mes + 1,
                'tarifa': tarifa,
                'nome_aparelho': aparelhos,
                'tipo': tipos,
                'tempo_uso': tempos_uso,
                'consumo_energia': consumo_energia,
                'gasto': gasto
            }
            data.append(pd.DataFrame(mes_data))
    return pd.concat(data, ignore_index=True)

# Simulando dados para 6 meses e 5 usuários
dados_historicos = simular_dados(6, usuarios)
dados_historicos.to_csv('dados_historicos.csv', index=False)

# Carregando dados históricos
dados_historicos = pd.read_csv('dados_historicos.csv')

# Filtrando dados para o usuário específico
usuario_id = 1

# Dados do mês atual (considerando o mês 6 como o atual)
dados_mes_atual = dados_historicos[(dados_historicos['mes'] == 6) & (dados_historicos['usuario_id'] == usuario_id)]
gasto_total_mes = dados_mes_atual['gasto'].sum()

# Dados do mês anterior (mês 5)
dados_mes_anterior = dados_historicos[(dados_historicos['mes'] == 5) & (dados_historicos['usuario_id'] == usuario_id)]
gasto_total_mes_anterior = dados_mes_anterior['gasto'].sum()

# Treinando o modelo de RandomForest com dados de meses anteriores do usuário específico
dados_treino = dados_historicos[(dados_historicos['mes'] < 6) & (dados_historicos['usuario_id'] == usuario_id)]
X = dados_treino[['tempo_uso', 'consumo_energia']]
y = dados_treino['gasto']
modelo = RandomForestRegressor(random_state=42)
modelo.fit(X, y)

# Fazendo previsões para o mês atual
X_atual = dados_mes_atual[['tempo_uso', 'consumo_energia']]
y_pred = modelo.predict(X_atual)

# Função para gerar recomendações
def gerar_recomendacoes(usuario_id, aparelho, gasto_atual, gasto_recomendado, gasto_anterior, historico):
    economia = gasto_atual - gasto_recomendado
    percentual_economia = (economia / gasto_atual) * 100
    economia_relativa = gasto_anterior - gasto_atual
    percentual_economia_relativa = (economia_relativa / gasto_anterior) * 100

    # Encontrando o mês de maior economia
    historico_aparelho = historico[historico['nome_aparelho'] == aparelho].copy()
    historico_aparelho['economia'] = historico_aparelho['gasto'].diff().abs()  # Mudança absoluta no gasto
    mes_maior_economia = historico_aparelho.loc[historico_aparelho['economia'].idxmax(), 'mes']
    valor_maior_economia = historico_aparelho.loc[historico_aparelho['economia'].idxmax(), 'economia']

    if economia > 0:
        return f"Usuário {usuario_id}: Você pode economizar R${economia:.2f} ({percentual_economia:.2f}%) no {aparelho} reduzindo o uso ou otimizando horários. Em relação ao mês passado, você economizou R${economia_relativa:.2f} ({percentual_economia_relativa:.2f}%). O mês que você mais economizou no {aparelho} foi o mês {mes_maior_economia} com uma economia de R${valor_maior_economia:.2f}."
    else:
        return f"Usuário {usuario_id}: Seu consumo no {aparelho} está dentro do recomendado. Continue assim! Em relação ao mês passado, você economizou R${economia_relativa:.2f} ({percentual_economia_relativa:.2f}%). O mês que você mais economizou no {aparelho} foi o mês {mes_maior_economia} com uma economia de R${valor_maior_economia:.2f}."

# Geração de recomendações
recomendacoes = [
    gerar_recomendacoes(
        usuario_id=usuario_id,
        aparelho=aparelho,
        gasto_atual=gasto_atual,
        gasto_recomendado=gasto_recomendado,
        gasto_anterior=gasto_anterior,
        historico=dados_historicos[dados_historicos['usuario_id'] == usuario_id]
    )
    for aparelho, gasto_atual, gasto_recomendado, gasto_anterior in zip(
        dados_mes_atual['nome_aparelho'], dados_mes_atual['gasto'], y_pred, dados_mes_anterior['gasto']
    )
]

# Comparando gastos
economia_mes = gasto_total_mes_anterior - gasto_total_mes
percentual_economia_mes = (economia_mes / gasto_total_mes_anterior) * 100

# Comparativo com outros usuários
gasto_medio_usuarios = dados_historicos[dados_historicos['mes'] == 6].groupby('usuario_id')['gasto'].sum()
gasto_usuario_atual = gasto_medio_usuarios.loc[usuario_id]

# Visualização dos Resultados com Matplotlib
plt.figure(figsize=(12, 6))
largura_barra = 0.35
indices = np.arange(len(dados_mes_atual))

plt.bar(indices, dados_mes_atual['gasto'], largura_barra, label='Gasto Atual', alpha=0.7, color='blue')
plt.bar(indices + largura_barra, y_pred, largura_barra, label='Gasto Predito', alpha=0.7, color='red')

plt.title('Gasto Atual x Gasto Recomendado por Aparelho', fontsize=14)
plt.xlabel('Aparelho', fontsize=12)
plt.ylabel('Gasto (R$)', fontsize=12)
plt.xticks(indices + largura_barra / 2, dados_mes_atual['nome_aparelho'], rotation=45)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Gráfico de Comparação com Outros Usuários
plt.figure(figsize=(12, 6))
usuarios = gasto_medio_usuarios.index
gastos_medios = gasto_medio_usuarios.values
indices_usuarios = np.arange(len(usuarios))

plt.bar(indices_usuarios, gastos_medios, label='Gasto Total dos Usuários', alpha=0.7, color='gray')
plt.bar(indices_usuarios[usuario_id-1], gasto_usuario_atual, label='Seu Gasto', alpha=0.7, color='green')

plt.title('Comparação de Gasto com Outros Usuários', fontsize=14)
plt.xlabel('Usuário', fontsize=12)
plt.ylabel('Gasto (R$)', fontsize=12)
plt.xticks(indices_usuarios, usuarios)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Exibindo a recomendação geral
print(f"Gasto Total do Mês Anterior: R${gasto_total_mes_anterior:.2f}")
print(f"Gasto Total do Mês Atual: R${gasto_total_mes:.2f}")
print(f"Economia Total: R${economia_mes:.2f} ({percentual_economia_mes:.2f}%)")
print("\nRecomendações:")
for rec in recomendacoes:
    print(rec)

# Exibindo a comparação com outros usuários
print("\nComparação com Outros Usuários:")
print(gasto_medio_usuarios)
