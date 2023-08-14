import math
import igraph as ig
import numpy as np
from random import randint
from random import sample
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import pylab
import pandas as pd

def numero_de_vertices(grafo):
    return grafo.vcount()

def numero_de_arestas(grafo):
    return grafo.ecount()

def lista_de_arestas(grafo):
    return grafo.get_edgelist()

def fluxo_total(grafo):
    return sum(grafo.es['weight'])

def componentes(grafo):
    return grafo.components()

def obter_grau_do_vertice(grafo, index):
    return grafo.vs[index].degree()

def numero_de_componentes(grafo):
    return len(grafo.components())

def tamanho_do_componente_gigante(grafo):
    return float(max(grafo.components().sizes()))

def deletar_vertice(grafo, indice_do_vertice):
    return grafo.delete_vertices(indice_do_vertice)

def gerador_de_falhas_aleatorias(n):
    return sample(range(0, n), 1)

def obter_nome_do_vertice(grafo, indice):
    return grafo.vs[indice]['label']

def p(grafo, numero_total_de_vertices):
    qtde_de_vertices_removidos = numero_total_de_vertices - numero_de_vertices(grafo)
    taxa_percentual_de_remocao = float(qtde_de_vertices_removidos / numero_total_de_vertices)
    try:
        tcg = tamanho_do_componente_gigante(grafo)
    except ValueError:
        tcg = 0
    return taxa_percentual_de_remocao, tcg

def aplicar_falhas_aleatorias(grafo):
    n = numero_de_vertices(grafo)  # ex.: n = 342
    historico_de_falhas_aleatorias = [(p(grafo, n), fluxo_total(grafo))]  # ex.: retorna: [((0, p(0)), fluxo_remanesc)]
    while numero_de_vertices(grafo) > 0:
        falha_aleatoria = gerador_de_falhas_aleatorias(numero_de_vertices(grafo))
        # print(f'Ponto de ataque: {obter_nome_do_vertice(grafo, falha_aleatoria)}')
        deletar_vertice(grafo, falha_aleatoria)
        historico_de_falhas_aleatorias.append((p(grafo, n), fluxo_total(grafo)))
    return historico_de_falhas_aleatorias  # retorna: [((f, p(f)), w)]

def aplicar_ataques_coordenados_grau(grafo):
    n = numero_de_vertices(grafo)
    historico_de_ataque = [(p(grafo, n), fluxo_total(grafo))]
    while numero_de_vertices(grafo) > 0:
        ataque_maior_grau = encontrar_maior_grau(grafo)
        deletar_vertice(grafo, ataque_maior_grau)
        historico_de_ataque.append((p(grafo, n), fluxo_total(grafo)))
    return historico_de_ataque

def aplicar_ataques_coordenados_betweenness(grafo):
    n = numero_de_vertices(grafo)
    historico_de_ataque = [(p(grafo, n), fluxo_total(grafo))]
    while numero_de_vertices(grafo) > 0:
        ataque_maior_betweenness = encontrar_maior_betweenness(grafo)
        deletar_vertice(grafo, ataque_maior_betweenness)
        historico_de_ataque.append((p(grafo, n), fluxo_total(grafo)))
    return historico_de_ataque

# retorna o vértice de maior grau na rede.
def encontrar_maior_grau(grafo):
    lista_de_graus = grafo.degree()
    maior_grau = max(lista_de_graus)
    return lista_de_graus.index(maior_grau)

# retorna o vértice de maior betweenness na rede.
def encontrar_maior_betweenness(grafo):
    lista_de_betweenness = grafo.betweenness()
    maior_betweenness = max(lista_de_betweenness)
    return lista_de_betweenness.index(maior_betweenness)

# retorna o vértice de maior fluxo na rede.
def encontrar_maior_fluxo(grafo):
    n = numero_de_vertices(grafo)
    fluxo = []
    fluxo_vertice = 0
    # para cada vértice da rede, verifique o peso de cada incidência
    for indice_vertice in range(n):
        for indice_incidencia in range(len(grafo.vs[indice_vertice].incident())):
            fluxo_vertice += grafo.vs[indice_vertice].incident()[indice_incidencia]['weight']
        fluxo.append(fluxo_vertice)
        fluxo_vertice = 0
    maior_fluxo = max(fluxo)
    return fluxo.index(maior_fluxo)

def obter_fluxo_total_de_cada_vertice(grafo):
    n = numero_de_vertices(grafo)
    fluxo = []
    fluxo_vertice = 0
    # para cada vértice da rede, verifique o peso de cada incidência
    for indice_vertice in range(n):
        for indice_incidencia in range(len(grafo.vs[indice_vertice].incident())):
            fluxo_vertice += grafo.vs[indice_vertice].incident()[indice_incidencia]['weight']
        fluxo.append(fluxo_vertice)
        fluxo_vertice = 0
    return fluxo

def cria_copias_da_rede(grafo, nro_de_copias):
    copias_da_rede = []
    for id_copia in range(nro_de_copias):
        copias_da_rede.append(grafo.copy())
    return copias_da_rede

def simular_falhas_aleatorias(grafo, qtde_de_simulacoes):
    resultados_das_simulacoes = []
    copias_da_rede = cria_copias_da_rede(grafo, qtde_de_simulacoes)
    # print(f'Quantidade de copias da rede: {len(copias_da_rede)}')
    # print(copias_da_rede)
    for rede in copias_da_rede:
        resultados_das_simulacoes.append(aplicar_falhas_aleatorias(rede))
    return resultados_das_simulacoes

def simular_ataques_coordenados_grau(grafo, qtde_de_simulacoes):
    resultados_das_simulacoes = []
    copias_da_rede = cria_copias_da_rede(grafo, qtde_de_simulacoes)
    for rede in copias_da_rede:
        resultados_das_simulacoes.append(aplicar_ataques_coordenados_grau(rede))
    return resultados_das_simulacoes

def simular_ataques_coordenados_betweenness(grafo, qtde_de_simulacoes):
    resultados_das_simulacoes = []
    copias_da_rede = cria_copias_da_rede(grafo, qtde_de_simulacoes)
    for rede in copias_da_rede:
        resultados_das_simulacoes.append(aplicar_ataques_coordenados_betweenness(rede))
    return resultados_das_simulacoes

def obter_historico_medio(resultados_das_simulacoes):
    historico_medio = []
    soma_f = 0
    soma_pf = 0
    soma_w = 0
    qtde_de_simulacoes = len(resultados_das_simulacoes)
    iteracao = 0
    # print(f'Quantidade de simulacoes: {qtde_de_simulacoes}')
    while iteracao < 173:
        for simulacao in range(qtde_de_simulacoes):
            ((f, pf), w) = resultados_das_simulacoes[simulacao][iteracao]
            soma_f += f
            soma_pf += pf
            soma_w += w
            # print(f'Simulacao: {simulacao} - Iteracao: {iteracao} -> soma_f: {soma_f}, soma_pf: {soma_pf}')
        historico_medio.append(
            ((soma_f / qtde_de_simulacoes, soma_pf / qtde_de_simulacoes), soma_w / qtde_de_simulacoes))
        soma_f = 0
        soma_pf = 0
        soma_w = 0
        iteracao += 1
    return historico_medio

def densidade(grafo):
    return grafo.density()

def grau_medio(grafo):
    return np.mean(grafo.degree())

def forca_media(grafo):
    return np.mean(grafo.strength(weights="weight"))

def diametro(grafo):
    return np.mean(grafo.get_diameter())

def coeficiente_de_clusterizacao(grafo):
    return grafo.transitivity_avglocal_undirected()

def assortatividade(grafo):
    return grafo.assortativity_degree()

def ponto_de_articulacao(grafo):
    return grafo.articulation_points()

def arvore_geradora(grafo):
    return grafo.spanning_tree()

def plotar_grafo(grafo):
    lyt = []
    for i in range(grafo.vcount()):
        lyt.append((grafo.vs[i]["X"], grafo.vs[i]["Y"] * (-1)))

    style = {"vertex_size": grafo.vs["size"], "edge_width": 0.4, "layout": lyt, "bbox": (1200, 1400), "margin": 30}

    ig.plot(grafo, **style)

def analisar_fluxo_total_remanescente(historico_de_perturbacoes):
    falhas_aleatorias = historico_de_perturbacoes[0]
    ataques_coordenados_grau = historico_de_perturbacoes[1]
    ataques_coordenados_betweenness = historico_de_perturbacoes[2]

    # x = f -> taxa de remoção dos nós
    # y = W -> fluxo total remanescente

    # falhas aleatórias
    eixo_x1 = []
    eixo_y1 = []
    # ataques coordenados por grau
    eixo_x2 = []
    eixo_y2 = []
    # ataques coordenados por betweeness
    eixo_x3 = []
    eixo_y3 = []

    # pegando o valor de w a partir do meu historico de perturbação da rede
    ((_, _), w) = falhas_aleatorias[0]
    """ Montando os eixos das abcissas (f) e das ordenadas (W) para as falhas aleatórias """
    for ((f, _), w) in falhas_aleatorias:
        eixo_x1.append(f)
        eixo_y1.append(w)

    robustez_falhas_aleatorias = sum(eixo_y1[1:]) / (len(eixo_y1) - 1)

    """ Montando os eixos das abcissas (f) e das ordenadas (W) para as ataques coordenados por grau"""
    for ((f, _), w) in ataques_coordenados_grau:
        eixo_x2.append(f)
        eixo_y2.append(w)

    robustez_ataques_coordenados_grau = sum(eixo_y2[1:]) / (len(eixo_y2) - 1)

    """ Montando os eixos das abcissas (f) e das ordenadas (W) para as ataques coordenados por betweenness"""
    for ((f, _), w) in ataques_coordenados_betweenness:
        eixo_x3.append(f)
        eixo_y3.append(w)

    robustez_ataques_coordenados_betweeness = sum(eixo_y3[1:]) / (len(eixo_y3) - 1)

    # plotagem do gráfico
    fig, ax = plt.subplots()

    ax.set_xlabel(r'$f$', fontsize=14)
    ax.set_ylabel(r'$||W||$', fontsize=14)
    ax.text(-0.1, 1.1, "Quantidade de ligações remanescentes x Fator comprometido da rede", transform=ax.transAxes, size=10, weight='bold')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(eixo_y1)])

    plt.plot(eixo_x1, eixo_y1, label="Falhas Aleatórias (" + str("{:.4f}".format(robustez_falhas_aleatorias)) + ")")
    plt.plot(eixo_x2, eixo_y2, label="Ataque por grau (" + str("{:.4f}".format(robustez_ataques_coordenados_grau)) +
                                     ")")
    plt.plot(eixo_x3, eixo_y3, label="Ataque por betweenness (" + str("{:.4f}".format(
        robustez_ataques_coordenados_betweeness)) + ")")

    plt.legend()
    plt.margins(x=0.02, y=0.02)
    plt.savefig("estacoes" + ".png")
    plt.show()

    return None

def ataques_coordenados(grafo, modo_de_ataque):
    n = numero_de_vertices(grafo)
    historico_de_ataque = [(p(grafo, n), fluxo_total(grafo))]
    cod_id = 0
    while numero_de_vertices(grafo) > 0:
        ataque_maior_metrica = grafo.vs.find(codigo=modo_de_ataque[cod_id][0]).index
        deletar_vertice(grafo, ataque_maior_metrica)
        historico_de_ataque.append((p(grafo, n), fluxo_total(grafo)))
        cod_id += 1
    return historico_de_ataque

# analisa o estado atual da rede para efetuar a estratégia de ataque.
def analise_de_robustez(g, modal):
    falhas_aleatorias = simular_falhas_aleatorias(g, 1)
    ataques_coordenados_grau = simular_ataques_coordenados_grau(g, 1)
    ataques_coordenados_betweenness = simular_ataques_coordenados_betweenness(g, 1)

    historico_medio_falhas_aleatorias = obter_historico_medio(falhas_aleatorias)
    historico_medio_ataques_grau = obter_historico_medio(ataques_coordenados_grau)
    historico_medio_ataques_betweenness = obter_historico_medio(ataques_coordenados_betweenness)

    historico_de_perturbacoes = [
        historico_medio_falhas_aleatorias,
        historico_medio_ataques_grau,
        historico_medio_ataques_betweenness
    ]

    # analisar_tamanho_do_componente_gigante(historico_de_perturbacoes, versao="_" + modal + "_v1")
    analisar_fluxo_total_remanescente(historico_de_perturbacoes)

def executa_processo(arquivo):
    path = 1
    while path < 2:
        dado_de_mobilidade = pd.read_csv(arquivo.get(path), delimiter=';', header=None)
        estacoes_de_sp = pd.read_csv(arquivo.get(0), delimiter=';', encoding="latin1")
        ligacoes = dado_de_mobilidade.values

        modal = arquivo.get(path).split('/')[1].split('.')[0].split('_')[1]

        g = ig.Graph.Weighted_Adjacency(ligacoes.tolist(), attr="weight", mode=ig.ADJ_MAX)
        g.to_undirected()
        g.es['weight'] = ligacoes[ligacoes.nonzero()]
        g.vs['label'] = estacoes_de_sp['NomeZona'].tolist()
        # coordendas
        g.vs["X"] = estacoes_de_sp['COORD_X'].tolist()
        g.vs["Y"] = estacoes_de_sp['COORD_Y'].tolist()

        # normalizacao dos graus dos vértices
        # deixar o tamanho do nó de acordo com o grau
        node_size = g.degree()
        menor_grau_existente = min(node_size)
        maior_grau_existente = max(node_size)
        # min and max desejáveis
        max_size_wanted = 100
        min_size_wanted = 5
        # normalização do tamanho do nó
        node_size = np.array(node_size)
        node_size = min_size_wanted + \
                    ((node_size - menor_grau_existente) *
                     (max_size_wanted - min_size_wanted) /
                     (maior_grau_existente - menor_grau_existente))
        # saving new sizes
        g.vs["size"] = node_size.tolist()

        plotar_grafo(g)
        g.write_graphml("modal_" + modal + ".GraphML")

        analise_de_robustez(g, modal=modal)
        path += 1