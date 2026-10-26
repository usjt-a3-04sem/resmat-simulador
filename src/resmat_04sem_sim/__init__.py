#!/usr/bin/env python3
"""
    python cadeira_resmat.py                      pergunta os dados (Enter aceita o valor entre colchetes)
    python cadeira_resmat.py --massa 90 --espessura 25 --material aco
    python cadeira_resmat.py --sem-janela         só salva a imagem e o memorial
"""
import argparse
import math
import sys

import numpy as np
import matplotlib

if "--sem-janela" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

G = 9.81  # m/s²

# E = módulo de elasticidade (MPa); lim = tensão limite (MPa).
# Valores típicos. Troque pelos da tabela da disciplina com --E-gpa e --limite-mpa.
MATERIAIS = {
    "madeira":  {"nome": "Madeira", "E": 12000.0, "lim": 40.0, "lim_nome": "resistência à compressão"},
    "aco":      {"nome": "Aço estrutural (A36)", "E": 200000.0, "lim": 250.0, "lim_nome": "tensão de escoamento"},
    "aluminio": {"nome": "Alumínio (6061-T6)", "E": 69000.0, "lim": 240.0, "lim_nome": "tensão de escoamento"},
}

# K = coeficiente de flambagem; fM = fração de H·h que vira momento na base;
# fD = fator da fórmula do deslocamento lateral.
APOIOS = {
    "livre":   {"nome": "base engastada e topo livre (conservador)", "K": 2.0, "fM": 1.0, "fD": 1 / 3},
    "travado": {"nome": "base engastada e topo travado em rotação pelo assento", "K": 1.0, "fM": 0.5, "fD": 1 / 12},
}

AZUL, AZUL_CLARO = "#2457C5", "#D6E1F9"
VERDE, VERMELHO = "#17703F", "#A8231B"
CINZA, GRAFITE = "#C9D1D9", "#16202B"


# ----------------------------------------------------------------------------
# Formatação (padrão: 1.234,56)
# ----------------------------------------------------------------------------
def n(x, d=2):
    s = f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s.replace("-", "−")


# ----------------------------------------------------------------------------
# Cálculo
# ----------------------------------------------------------------------------
def validar(p):
    erros = []

    def fora(x, lo, hi):
        return not (lo <= x <= hi)

    if fora(p["massa"], 1, 500):
        erros.append("Peso da pessoa deve ficar entre 1 e 500 kg.")
    if fora(p["h"], 100, 2000):
        erros.append("Altura da cadeira deve ficar entre 100 e 2000 mm.")
    if fora(p["L"], 100, 1500):
        erros.append("Largura do assento deve ficar entre 100 e 1500 mm.")
    if fora(p["a"], 2, 300):
        erros.append("Espessura das pernas deve ficar entre 2 e 300 mm.")
    elif p["a"] >= p["L"] / 2:
        erros.append("Espessura das pernas deve ser menor que metade da largura do assento.")
    if fora(p["exc"], 0, p["L"] / 2):
        erros.append("Excentricidade deve ficar entre 0 e L/2 (metade da largura do assento).")
    if fora(p["k"], 0, 100):
        erros.append("Carga horizontal deve ficar entre 0 e 100 % do peso.")
    if fora(p["phi"], 0.5, 5):
        erros.append("Fator de majoração deve ficar entre 0,5 e 5.")
    if fora(p["FS"], 1, 10):
        erros.append("Fator de segurança deve ficar entre 1 e 10.")
    if p["E"] <= 0 or p["lim"] <= 0:
        erros.append("Módulo de elasticidade e tensão limite devem ser positivos.")
    return erros


def calcular(p):
    ap = APOIOS[p["apoio"]]
    h, L, a, e = p["h"], p["L"], p["a"], p["exc"]

    W = p["massa"] * G * p["phi"]                 # peso em força (N)
    Nd = W / 4 * (1 + 2 * e / L)                  # perna dianteira (N)
    Nt = W / 4 * (1 - 2 * e / L)                  # perna traseira (N)
    H = p["k"] / 100 * W                          # força horizontal total (N)
    Hp = H / 4                                    # por perna (N)

    A = a * a                                     # área (mm²)
    I = a ** 4 / 12                               # momento de inércia (mm⁴)
    Wf = a ** 3 / 6                               # módulo resistente (mm³)
    r = a / math.sqrt(12)                         # raio de giração (mm)

    N = Nd                                        # perna crítica
    Mb = ap["fM"] * Hp * h                        # momento na base (N·mm)
    Mt = Mb - Hp * h                              # momento no topo (N·mm)

    sN = N / A                                    # tensão da compressão (MPa)
    sM = Mb / Wf                                  # tensão da flexão (MPa)
    sMax = sN + sM                                # face da frente (compressão)
    sMin = sN - sM                                # face de trás
    sAdm = p["lim"] / p["FS"]
    util = sMax / sAdm

    lam = ap["K"] * h / r
    lamC = math.sqrt(2 * math.pi ** 2 * p["E"] / p["lim"])
    euler = lam >= lamC
    if euler:
        sCr = math.pi ** 2 * p["E"] / lam ** 2
    else:
        sCr = p["lim"] * (1 - p["lim"] * lam ** 2 / (4 * math.pi ** 2 * p["E"]))
    Pcr = sCr * A
    FSfl = Pcr / N

    dAx = N * h / (p["E"] * A)
    dLat = Hp * h ** 3 * ap["fD"] / (p["E"] * I)

    fRes = sAdm / sMax
    fFl = (Pcr / p["FS"]) / N
    mMax = p["massa"] * min(fRes, fFl)
    okRes, okFl = util <= 1, FSfl >= p["FS"]

    return dict(ap=ap, W=W, Nd=Nd, Nt=Nt, H=H, Hp=Hp, A=A, I=I, Wf=Wf, r=r, N=N, Mb=Mb, Mt=Mt,
                sN=sN, sM=sM, sMax=sMax, sMin=sMin, sAdm=sAdm, util=util, lam=lam, lamC=lamC,
                euler=euler, sCr=sCr, Pcr=Pcr, FSfl=FSfl, dAx=dAx, dLat=dLat, fRes=fRes, fFl=fFl,
                mMax=mMax, okRes=okRes, okFl=okFl, ok=okRes and okFl)


# ----------------------------------------------------------------------------
# Memorial de cálculo explicado
# ----------------------------------------------------------------------------
def memorial(p, r):
    mat = MATERIAIS[p["material"]]
    ap = r["ap"]
    L = []

    def passo(num, titulo, ideia, contas):
        L.append(f"PASSO {num} - {titulo}")
        L.append("-" * 70)
        L.append("Ideia: " + ideia)
        L.append("")
        for c in contas:
            L.append("    " + c)
        L.append("")

    L.append("MEMORIAL DE CÁLCULO - CADEIRA")
    L.append("=" * 70)
    L.append(f"Pessoa: {n(p['massa'], 1)} kg | altura da cadeira: {n(p['h'], 0)} mm | "
             f"largura do assento: {n(p['L'], 0)} mm")
    L.append(f"Pernas quadradas de {n(p['a'], 0)} mm de lado | material: {mat['nome']} "
             f"(E = {n(p['E'] / 1000, 1)} GPa, limite = {n(p['lim'], 1)} MPa)")
    L.append(f"Hipóteses: peso {n(p['exc'], 0)} mm à frente do centro, carga horizontal de "
             f"{n(p['k'], 1)}% do peso, FS = {n(p['FS'], 2)}, φ = {n(p['phi'], 2)}")
    L.append(f"Vinculação da perna: {ap['nome']}")
    L.append("")

    passo(1, "Do peso (kg) para força (N)",
          "As fórmulas de resistência usam força, não massa. A força do peso é a massa vezes a "
          "gravidade. O fator φ vale 1,0 para pessoa parada; sentar de repente exigiria mais.",
          [f"W = m · g · φ = {n(p['massa'], 1)} · {n(G, 2)} · {n(p['phi'], 2)} = {n(r['W'], 1)} N"])

    passo(2, "Quanto do peso cada perna recebe",
          "O assento reparte o peso entre as 4 pernas. Com a pessoa no centro, cada uma recebe W/4. "
          "Se o peso está deslocado para a frente, as pernas da frente recebem mais e as de trás "
          "menos, mas a soma continua sendo W. A perna da frente é a mais carregada, então é ela "
          "que verificamos.",
          [f"dianteira = (W/4) · (1 + 2e/L) = ({n(r['W'], 1)}/4) · (1 + 2·{n(p['exc'], 0)}/{n(p['L'], 0)})"
           f" = {n(r['Nd'], 1)} N",
           f"traseira  = (W/4) · (1 − 2e/L) = ({n(r['W'], 1)}/4) · (1 − 2·{n(p['exc'], 0)}/{n(p['L'], 0)})"
           f" = {n(r['Nt'], 1)} N",
           f"conferência: 2·{n(r['Nd'], 1)} + 2·{n(r['Nt'], 1)} = {n(2 * r['Nd'] + 2 * r['Nt'], 1)} N = W"])

    passo(3, "Força horizontal",
          "Uma pessoa real se mexe, se inclina no encosto, empurra a cadeira. Isso cria uma força "
          "horizontal no assento. Adotamos um percentual do peso e dividimos pelas 4 pernas. É essa "
          "força que tenta envergar a perna.",
          [f"H = k · W = {n(p['k'] / 100, 2)} · {n(r['W'], 1)} = {n(r['H'], 1)} N",
           f"por perna: H/4 = {n(r['Hp'], 2)} N"])

    passo(4, "Características da seção da perna",
          "A perna tem seção quadrada de lado a. A área A diz quanto material resiste à compressão. "
          "O módulo resistente W diz quanto a seção resiste à flexão. Ele cresce com a³: dobrar a "
          "espessura multiplica a resistência à flexão por 8.",
          [f"A = a² = {n(p['a'], 0)}² = {n(r['A'], 0)} mm²",
           f"W = a³/6 = {n(p['a'], 0)}³/6 = {n(r['Wf'], 0)} mm³"])

    if p["apoio"] == "livre":
        conta_m = f"M na base = H_perna · h = {n(r['Hp'], 2)} · {n(p['h'], 0)} = {n(r['Mb'], 0)} N·mm"
        expl_m = ("A força horizontal, vezes a altura (braço de alavanca), tenta envergar a perna. "
                  "O momento é máximo na base, onde a perna está presa ao piso, e zero no topo.")
    else:
        conta_m = f"M na base = H_perna · h / 2 = {n(r['Hp'], 2)} · {n(p['h'], 0)} / 2 = {n(r['Mb'], 0)} N·mm"
        expl_m = ("Com o topo travado pelo assento, o momento se divide: metade de H·h na base e "
                  "metade, com sinal oposto, no topo.")
    passo(5, "Esforços dentro da perna",
          "N é o peso comprimindo a perna. V é a força horizontal 'cortando' a perna. M é o momento "
          "fletor. " + expl_m,
          [f"N = {n(r['N'], 1)} N (compressão, igual ao longo da perna)",
           f"V = {n(r['Hp'], 2)} N (igual ao longo da perna)",
           conta_m,
           f"M no topo = {n(r['Mt'], 0)} N·mm"])

    passo(6, "Tensões normais na perna",
          "Tensão é força dividida por área (1 MPa = 1 N/mm²). A compressão do peso se espalha igual "
          "por toda a seção (σN). A flexão comprime a face da frente e traciona a de trás (σM). Na "
          "face da frente os dois efeitos se somam, e essa é a pior região da perna.",
          [f"σN = N / A = {n(r['N'], 1)} / {n(r['A'], 0)} = {n(r['sN'], 3)} MPa",
           f"σM = M / W = {n(r['Mb'], 0)} / {n(r['Wf'], 0)} = {n(r['sM'], 3)} MPa",
           f"face da frente (máxima): σmax = σN + σM = {n(r['sMax'], 3)} MPa (compressão)",
           f"face de trás: σmin = σN − σM = {n(r['sMin'], 3)} MPa "
           f"({'tração' if r['sMin'] < 0 else 'compressão'})"])

    passo(7, "A perna aguenta? Comparação com o limite do material",
          f"O material tem um limite ({mat['lim_nome']}). Dividimos esse limite pelo fator de "
          "segurança FS, uma folga contra imprevistos, e chamamos o resultado de tensão admissível. "
          "A perna passa se a tensão máxima ficar abaixo dela.",
          [f"σadm = limite / FS = {n(p['lim'], 1)} / {n(p['FS'], 2)} = {n(r['sAdm'], 2)} MPa",
           f"σmax = {n(r['sMax'], 3)} MPa  ->  {'MENOR' if r['okRes'] else 'MAIOR'} que σadm  ->  "
           f"{'ATENDE' if r['okRes'] else 'NÃO ATENDE'}",
           f"utilização = σmax / σadm = {n(r['util'] * 100, 1)} %"])

    regime = "Euler (perna esbelta)" if r["euler"] else "Johnson (perna curta, trecho inelástico)"
    passo(8, "Flambagem (verificação extra)",
          "Perna fina e comprida pode flambar, ou seja, envergar de repente sob compressão antes de "
          "esmagar. Calcula-se a carga crítica Pcr e compara-se com a carga real. Esbeltez λ é o "
          "quanto a perna é comprida em relação à sua espessura.",
          [f"λ = K · h / raio = {n(ap['K'], 1)} · {n(p['h'], 0)} / {n(r['r'], 2)} = {n(r['lam'], 1)}",
           f"λ limite = {n(r['lamC'], 1)}  ->  regime de {regime}",
           f"σcr = {n(r['sCr'], 2)} MPa   ->   Pcr = σcr · A = {n(r['Pcr'], 0)} N",
           f"FS à flambagem = Pcr / N = {n(r['FSfl'], 2)}  (exigido {n(p['FS'], 2)})  ->  "
           f"{'ATENDE' if r['okFl'] else 'NÃO ATENDE'}"])

    passo(9, "Deformações (informativo)",
          "Quanto a perna encurta com o peso e quanto o topo se desloca para o lado com a força "
          "horizontal. Em cadeira real esses valores são pequenos.",
          [f"encurtamento = N · h / (E · A) = {n(r['dAx'], 4)} mm",
           f"deslocamento lateral do topo = {n(r['dLat'], 3)} mm"])

    passo(10, "Peso máximo admissível (só pelas pernas)",
          "As tensões crescem na mesma proporção do peso. Então o peso máximo é o peso testado "
          "multiplicado pela folga que sobrou, na resistência ou na flambagem, a que for menor. "
          "Isso considera somente as pernas: assento e ligações não foram verificados.",
          [f"folga na resistência = σadm / σmax = {n(r['fRes'], 2)} vezes",
           f"folga na flambagem = (Pcr / FS) / N = {n(r['fFl'], 2)} vezes",
           f"peso máximo = {n(p['massa'], 1)} · {n(min(r['fRes'], r['fFl']), 2)} = {n(r['mMax'], 0)} kg"])

    L.append("CONCLUSÃO")
    L.append("=" * 70)
    if r["ok"]:
        L.append(f"A cadeira SUPORTA a carga de {n(p['massa'], 1)} kg. Estrutura SEGURA: a tensão "
                 f"máxima nas pernas ({n(r['sMax'], 3)} MPa) fica abaixo da admissível "
                 f"({n(r['sAdm'], 2)} MPa) e a flambagem está dentro do fator de segurança.")
    else:
        motivos = []
        if not r["okRes"]:
            motivos.append("a tensão máxima passa da admissível")
        if not r["okFl"]:
            motivos.append("o fator de segurança à flambagem é menor que o exigido")
        L.append(f"A cadeira NÃO SUPORTA a carga de {n(p['massa'], 1)} kg. Estrutura NÃO SEGURA: "
                 + " e ".join(motivos) + ".")
    if p["k"] == 0:
        L.append("Atenção: sem carga horizontal não há flexão; a perna trabalha só à compressão.")
    if r["mMax"] > 1000:
        L.append("Atenção: só as pernas foram verificadas. Em uma cadeira real o limite costuma "
                 "estar no assento e nas ligações, que este modelo não calcula.")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def hachura(ax, x0, x1, y, passo, dx, dy):
    ax.plot([x0, x1], [y, y], color="#4A5865", lw=1.5, zorder=2)
    for x in np.arange(x0, x1, passo):
        ax.plot([x, x - dx], [y, y - dy], color="#4A5865", lw=0.9, zorder=2)


def seta(ax, ini, fim, cor=AZUL, lw=2.6):
    ax.annotate("", xy=fim, xytext=ini,
                arrowprops=dict(arrowstyle="-|>", color=cor, lw=lw, shrinkA=0, shrinkB=0), zorder=5)


def esquema(ax, p, r):
    h, L, a, e = p["h"], p["L"], p["a"], p["exc"]
    xs, xd = -L / 2, L / 2
    esp = max(0.035 * h, 8)
    ax.add_patch(Rectangle((xs - a / 2, 0), a, h, fc=CINZA, ec="#4A5865", lw=1.2, zorder=3))
    ax.add_patch(Rectangle((xd - a / 2, 0), a, h, fc=AZUL_CLARO, ec=AZUL, lw=2, zorder=3))
    ax.add_patch(Rectangle((xs - a / 2, h), L + a, esp, fc=GRAFITE, zorder=4))
    hachura(ax, xs - 0.3 * L, xd + 0.3 * L, 0, 0.04 * L + 2, 0.035 * h, 0.035 * h)
    ax.plot([0, 0], [h + esp, h + esp + 0.4 * h], ls="--", color="#4A5865", lw=1, zorder=1)
    seta(ax, (e, h + esp + 0.36 * h), (e, h + esp))
    ax.text(e + 0.03 * L, h + esp + 0.27 * h, f"W = {n(r['W'], 1)} N", color=AZUL,
            fontsize=11, fontweight="bold", va="center")
    seta(ax, (xs - a / 2 - 0.34 * L, h + esp / 2), (xs - a / 2, h + esp / 2))
    ax.text(xs - a / 2 - 0.34 * L, h + esp + 0.04 * h, f"H = {n(r['H'], 1)} N", color=AZUL,
            fontsize=11, fontweight="bold")
    # cotas
    xc = xd + a / 2 + 0.16 * L
    ax.plot([xc, xc], [0, h], color="#4A5865", lw=1)
    for y in (0, h):
        ax.plot([xc - 0.02 * L, xc + 0.02 * L], [y, y], color="#4A5865", lw=1)
    ax.text(xc + 0.04 * L, h / 2, f"h = {n(h, 0)} mm", rotation=90, va="center", fontsize=10, color="#4A5865")
    yc = -0.14 * h
    ax.plot([xs, xd], [yc, yc], color="#4A5865", lw=1)
    for x in (xs, xd):
        ax.plot([x, x], [yc - 0.02 * h, yc + 0.02 * h], color="#4A5865", lw=1)
    ax.text(0, yc - 0.03 * h, f"L = {n(L, 0)} mm", ha="center", va="top", fontsize=10, color="#4A5865")
    ax.text(xs, -0.30 * h, "perna traseira", ha="right", va="top", fontsize=10)
    ax.text(xd, -0.30 * h, "perna dianteira (crítica)", ha="left", va="top", fontsize=10,
            color=AZUL, fontweight="bold")
    ax.set_xlim(-0.95 * L, 0.95 * L + a)
    ax.set_ylim(-0.42 * h, h + esp + 0.5 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Modelo estrutural (vista lateral, em escala)", fontsize=12, fontweight="bold", loc="left")


def painel(ax, tipo, titulo, un, v0, v1, h, casas):
    m = max(abs(v0), abs(v1)) or 1.0
    sinal = -1 if tipo == "M" else 1          # M é desenhado do lado tracionado (traseiro)
    x0, x1 = sinal * v0, sinal * v1
    kw = dict(fc=AZUL_CLARO, ec=AZUL, lw=1.5, zorder=2)
    if v0 * v1 < 0:
        yz = h * v0 / (v0 - v1)
        ax.fill([0, x0, 0], [0, 0, yz], **kw)
        ax.fill([0, 0, x1], [yz, h, h], **kw)
    else:
        ax.fill([0, x0, x1, 0], [0, 0, h, h], **kw)
    w = 0.10 * m
    ax.add_patch(Rectangle((-w / 2, 0), w, h, fc=AZUL_CLARO, ec=AZUL, lw=1.8, zorder=4))
    hachura(ax, -1.5 * m, 1.5 * m, 0, 0.16 * m, 0.12 * m, 0.04 * h)
    if tipo == "N":
        seta(ax, (0, 1.24 * h), (0, h * 1.01))
        ax.text(0.12 * m, 1.14 * h, "N", color=AZUL, fontweight="bold", fontsize=11)
    else:
        seta(ax, (-1.35 * m, h), (-w / 2, h))
        ax.text(-1.35 * m, 1.05 * h, "H da perna", color=AZUL, fontweight="bold", fontsize=10)
    ax.set_xlim(-1.6 * m, 1.6 * m)
    ax.set_ylim(-0.1 * h, 1.34 * h)
    ax.set_xticks([])
    ax.set_yticks([0, h])
    ax.set_yticklabels(["base", "topo"])
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel(f"base: {n(v0, casas)} {un}\ntopo: {n(v1, casas)} {un}", fontsize=10)
    for lado in ("top", "right", "bottom"):
        ax.spines[lado].set_visible(False)


def painel_tensoes(ax, p, r):
    a = p["a"]
    xs = np.linspace(-a / 2, a / 2, 60)
    flex = r["sM"] * (2 * xs / a)              # −σM atrás, +σM na frente
    soma = r["sN"] + flex
    ax.axhline(0, color="#4A5865", lw=1)
    ax.plot(xs, np.full_like(xs, r["sN"]), ls="--", color="#4A5865", lw=1.6, label=f"compressão do peso (σN = {n(r['sN'], 3)})")
    ax.plot(xs, flex, ls=":", color="#B9770E", lw=2, label=f"flexão (±σM = ±{n(r['sM'], 3)})")
    ax.plot(xs, soma, color=AZUL, lw=2.8, label="soma (tensão real)")
    ax.fill_between(xs, 0, soma, color=AZUL_CLARO, alpha=0.8, zorder=0)
    ax.annotate(f"{n(r['sMin'], 3)}", (xs[0], soma[0]), textcoords="offset points", xytext=(4, -14), fontsize=9)
    ax.annotate(f"{n(r['sMax'], 3)}", (xs[-1], soma[-1]), textcoords="offset points", xytext=(-4, 6),
                ha="right", fontsize=9, fontweight="bold", color=AZUL)
    topo = max(r["sMax"], abs(r["sMin"]), r["sN"])
    if r["sAdm"] <= 3 * topo:
        for s in (r["sAdm"], -r["sAdm"]):
            ax.axhline(s, color=VERMELHO, ls="--", lw=1.3)
        ax.text(xs[0], r["sAdm"], " σadm", color=VERMELHO, va="bottom", fontsize=9)
        ax.text(xs[0], -r["sAdm"], " −σadm", color=VERMELHO, va="top", fontsize=9)
        ax.set_ylim(min(-1.15 * r["sAdm"], 1.1 * r["sMin"]), 1.25 * max(r["sAdm"], r["sMax"]))
    else:
        ax.text(0.98, 0.04, f"σadm = {n(r['sAdm'], 1)} MPa\n(fora da escala)", transform=ax.transAxes,
                ha="right", va="bottom", color=VERMELHO, fontsize=9)
        ax.set_ylim(min(-0.25 * topo, 1.3 * r["sMin"]), 1.35 * topo)
    ax.set_xlim(-a / 2, a / 2)
    ax.set_xticks([-a / 2, 0, a / 2])
    ax.set_xticklabels(["trás", "", "frente"])
    ax.set_ylabel("tensão (MPa), + compressão")
    ax.set_title("Tensões na base da perna crítica", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def resumo(ax, p, r):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    T = dict(transform=ax.transAxes)
    cor = VERDE if r["ok"] else VERMELHO
    ax.text(0.0, 0.97, "SEGURA" if r["ok"] else "NÃO SEGURA", fontsize=30, fontweight="bold",
            color=cor, va="top", **T)
    frase = (f"A cadeira suporta a carga de {n(p['massa'], 0)} kg" if r["ok"]
             else f"A cadeira não suporta a carga de {n(p['massa'], 0)} kg")
    ax.text(0.0, 0.80, frase, fontsize=14, va="top", **T)
    linhas = [
        ("Tensão máxima nas pernas", f"{n(r['sMax'], 3)} MPa"),
        ("Tensão admissível", f"{n(r['sAdm'], 2)} MPa"),
        ("Utilização", f"{n(r['util'] * 100, 1)} %"),
        ("Flambagem (FS obtido / exigido)", f"{n(r['FSfl'], 2)} / {n(p['FS'], 2)}"),
        ("Peso máx. admissível (só pernas)", f"{n(r['mMax'], 0)} kg"),
    ]
    y = 0.62
    for nome, val in linhas:
        ax.text(0.0, y, nome, fontsize=11, color="#4A5865", va="center", **T)
        ax.text(0.98, y, val, fontsize=12, fontweight="bold", ha="right", va="center",
                family="monospace", **T)
        y -= 0.085
    # barra de utilização (0 a 150 %)
    ax.add_patch(Rectangle((0, 0.09), 0.98, 0.05, fc="#EEF1F4", ec=CINZA, **T))
    ax.add_patch(Rectangle((0, 0.09), 0.98 * min(r["util"], 1.5) / 1.5, 0.05, fc=cor, **T))
    xl = 0.98 / 1.5
    ax.plot([xl, xl], [0.06, 0.17], color=GRAFITE, lw=2, **T)
    ax.text(xl, 0.04, "100% = σadm", ha="center", va="top", fontsize=9, color="#4A5865", **T)
    ax.text(0.0, 0.04, "0%", ha="left", va="top", fontsize=9, color="#4A5865", **T)
    ax.text(0.98, 0.04, "150%", ha="right", va="top", fontsize=9, color="#4A5865", **T)


def desenhar(p, r, salvar, mostrar):
    mat = MATERIAIS[p["material"]]["nome"]
    fig = plt.figure(figsize=(15, 9.6), constrained_layout=True)
    fig.suptitle(f"Cadeira Resmat - {mat}, pernas de {n(p['a'], 0)} mm, {n(p['massa'], 0)} kg",
                 fontsize=14, fontweight="bold", x=0.01, ha="left")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])
    esquema(fig.add_subplot(gs[0, 0:2]), p, r)
    resumo(fig.add_subplot(gs[0, 2:4]), p, r)
    painel(fig.add_subplot(gs[1, 0]), "N", "Esforço normal N", "N", -r["N"], -r["N"], p["h"], 1)
    painel(fig.add_subplot(gs[1, 1]), "V", "Esforço cortante V", "N", r["Hp"], r["Hp"], p["h"], 1)
    painel(fig.add_subplot(gs[1, 2]), "M", "Momento fletor M", "N·mm", r["Mb"], r["Mt"], p["h"], 0)
    painel_tensoes(fig.add_subplot(gs[1, 3]), p, r)
    if salvar:
        fig.savefig(salvar, dpi=130)
        print(f"Imagem salva em {salvar}")
    if mostrar:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# Entrada de dados
# ----------------------------------------------------------------------------
def perguntar(texto, padrao):
    while True:
        bruto = input(f"{texto} [{n(padrao, 0) if float(padrao).is_integer() else n(padrao, 2)}]: ").strip()
        if not bruto:
            return float(padrao)
        try:
            return float(bruto.replace(",", "."))
        except ValueError:
            print("  Valor inválido, tente de novo.")


def entrada_interativa(p):
    print("Cadeira Resmat - tecle Enter para aceitar o valor entre colchetes.\n")
    p["massa"] = perguntar("Peso da pessoa (kg)", p["massa"])
    p["h"] = perguntar("Altura da cadeira, piso ao assento (mm)", p["h"])
    p["L"] = perguntar("Largura do assento (mm)", p["L"])
    p["a"] = perguntar("Espessura das pernas, lado do quadrado (mm)", p["a"])
    nomes = list(MATERIAIS)
    print("Material: " + " | ".join(f"{i + 1} = {MATERIAIS[k]['nome']}" for i, k in enumerate(nomes)))
    esc = perguntar("Escolha o material (1, 2 ou 3)", 1)
    p["material"] = nomes[int(esc) - 1] if 1 <= int(esc) <= 3 else "madeira"
    p["E"], p["lim"] = MATERIAIS[p["material"]]["E"], MATERIAIS[p["material"]]["lim"]
    if input("Quer alterar as hipóteses de carga? (s/N): ").strip().lower() == "s":
        p["exc"] = perguntar("Excentricidade do peso (mm)", p["exc"])
        p["k"] = perguntar("Carga horizontal (% do peso)", p["k"])
        p["phi"] = perguntar("Fator de majoração φ", p["phi"])
        p["FS"] = perguntar("Fator de segurança FS", p["FS"])
    return p


def main():
    ap = argparse.ArgumentParser(description="Verificação estrutural de uma cadeira (Projeto A3).")
    ap.add_argument("--massa", type=float, default=80, help="peso da pessoa (kg)")
    ap.add_argument("--altura", type=float, default=450, help="altura da cadeira, piso ao assento (mm)")
    ap.add_argument("--largura", type=float, default=400, help="largura do assento (mm)")
    ap.add_argument("--espessura", type=float, default=30, help="lado da seção quadrada da perna (mm)")
    ap.add_argument("--material", choices=list(MATERIAIS), default="madeira")
    ap.add_argument("--E-gpa", type=float, help="módulo de elasticidade (GPa); troca o valor típico")
    ap.add_argument("--limite-mpa", type=float, help="tensão limite (MPa); troca o valor típico")
    ap.add_argument("--excentricidade", type=float, default=50, help="peso deslocado para a frente (mm)")
    ap.add_argument("--horizontal", type=float, default=10, help="carga horizontal (%% do peso)")
    ap.add_argument("--phi", type=float, default=1.0, help="fator de majoração da carga")
    ap.add_argument("--fs", type=float, default=2.0, help="fator de segurança")
    ap.add_argument("--apoio", choices=list(APOIOS), default="livre")
    ap.add_argument("--saida", default="resultado_cadeira.png", help="arquivo da imagem")
    ap.add_argument("--sem-janela", action="store_true", help="não abre a janela, só salva os arquivos")
    a = ap.parse_args()

    mat = MATERIAIS[a.material]
    p = dict(massa=a.massa, h=a.altura, L=a.largura, a=a.espessura, material=a.material,
             E=(a.E_gpa * 1000 if a.E_gpa else mat["E"]), lim=(a.limite_mpa or mat["lim"]),
             exc=a.excentricidade, k=a.horizontal, phi=a.phi, FS=a.fs, apoio=a.apoio)
    if len(sys.argv) == 1:
        p = entrada_interativa(p)

    erros = validar(p)
    if erros:
        print("Revise as entradas:")
        for e in erros:
            print(" -", e)
        sys.exit(1)

    r = calcular(p)
    texto = memorial(p, r)
    print(texto)
    with open("memorial_cadeira.txt", "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print("\nMemorial salvo em memorial_cadeira.txt")
    desenhar(p, r, a.saida, mostrar=not a.sem_janela)


if __name__ == "__main__":
    main()
