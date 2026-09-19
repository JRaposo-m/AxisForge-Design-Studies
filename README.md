# AxisForge Design Studies

Ponte entre [AxisForge](https://github.com/JRaposo-m/AxisForge-Shaft-Bearing-Gear-System-Analysis-Platform) — motor CAE determinístico, baseado em normas, para veios, rolamentos e sistemas de engrenagens de eixos paralelos — e [SlipPY](https://github.com/FrictionTribologyEnigma/slippy), uma biblioteca de mecânica de contacto/tribologia. O AxisForge constrói e resolve o sistema mecânico; o SlipPY analisa o contacto local e a lubrificação; este repositório é onde os dois se encontram.

Nem o AxisForge nem o SlipPY sabem que este repositório existe — o acoplamento é unidirecional: `axisforge-design-studies` usa os dois como bibliotecas, nunca o inverso.

## Arquitetura

```
axisforge_bridge/       tudo o que fala com o AxisForge
    construction/          constrói sistemas (veio + rolamentos/engrenagens)
    extract/               extrai dados de contacto de resultados resolvidos
    studies/               estudos só-AxisForge (ex. convergência de malha,
                            comparação de solvers) — validação de um só lado,
                            muitas vezes input para um estudo conjunto
    validation/            casos de referência + resultado esperado, para
                            regressão

slippy_bridge/           tudo o que fala com o SlipPY
    build/                 monta casos SlipPY por elemento de máquina
                            (bearing_case.py, gear_case.py)
    (sem studies/ interno — validação de SlipPY isolado precisa sempre de
    dados vindos do AxisForge, logo é sempre um estudo conjunto)

results/                 forma do output de uma análise conjunta (pressão de
                          contacto, espessura de filme, mais tarde λ /
                          temperatura flash)

studies/                 só estudos conjuntos AxisForge+SlipPY — a única
                          camada que pode depender de axisforge_bridge,
                          slippy_bridge e results (ver a lei completa na
                          vault, 00_Master/WIRING.md)
    <domínio>/contract.py   dataclass neutro que liga as duas bridges,
                            específico de cada domínio de estudo (ex.
                            studies/contact/contract.py) — não um contrato
                            universal do repositório
```

Decidido em 2026-09-18/19 — ver `40_ADR/ADR-001_folder_organization.md` na
vault Obsidian deste repositório para o raciocínio completo por decisão.

## Estado atual (19/09/2026)

- `axisforge_bridge/construction/` e `axisforge_bridge/studies/01_shaft_static_analysis/` — construídos, com validação cruzada contra Abaqus para o FEM do veio (Euler-Bernoulli e Timoshenko, cargas pontuais e distribuídas).
- `axisforge_bridge/validation/` — migração em curso do conteúdo de convergência/comparação FEM para esta forma mais definitiva (casos de referência + `results_library`).
- `axisforge_bridge/studies/{02_deep_groove_bearing_life,03_helical_gearbox_drivetrain}` — ainda por construir.
- `slippy_bridge/` — vazio. Nada foi ainda escrito do lado SlipPY.
- `results/` e `studies/` (topo) — ainda por criar. O primeiro conteúdo real de `studies/` vai ser o spike descrito no doc de arquitetura: um rolamento, um ponto de contacto, dados extraídos "à bruta" do AxisForge, alimentando o SlipPY.

## Setup

```bash
git clone https://github.com/JRaposo-m/AxisForge-Shaft-Bearing-Gear-System-Analysis-Platform.git axisforge
git clone <this-repo-url> axisforge-design-studies
cd axisforge-design-studies
pip install -r requirements.txt
pip install -e ../axisforge
```

## Vault

Este repositório tem uma vault Obsidian irmã das do AxisForge
(`axisforge-agents/vault`) e do SlipPY (`slippy-study/vault`), em
`axisforge-design-studies/` dentro de `Projeto_Universidade` — gerada pelas
mesmas três ferramentas (`vault_sync.py`, `mindmap_gen.py`,
`dependency_cascade_gen.py`), com a lei de camadas adaptada a este
repositório. Ver `README_SETUP.md` aí para o bootstrap.
