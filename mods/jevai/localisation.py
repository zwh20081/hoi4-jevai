"""Write the JevAI mod's localisation: one UTF-8-with-BOM file per game language (localisation/<lang>/*_l_<lang>.yml),
the format HOI4 reads. Run after editing TEXT:  python -m mods.jevai.localisation
"""
from __future__ import annotations

import os

MOD = os.path.dirname(os.path.abspath(__file__))

# key -> {language: text}. \n is a line break in HOI4 text.
TEXT: dict[str, dict[str, str]] = {
    "jevai.1.t": {
        "english": "JevAI: Who Should the Model Command?",
        "simp_chinese": "JevAI：由模型指挥哪些国家？",
        "japanese": "JevAI：モデルが指揮する国は？",
        "korean": "JevAI: 모델이 지휘할 국가는?",
        "russian": "JevAI: кем будет командовать модель?",
        "german": "JevAI: Wen soll das Modell führen?",
        "french": "JevAI : quels pays le modèle doit-il diriger ?",
        "spanish": "JevAI: ¿a quién debe dirigir el modelo?",
        "braz_por": "JevAI: quem o modelo deve comandar?",
        "polish": "JevAI: kim ma dowodzić model?",
    },
    "jevai.1.d": {
        "english": "JevAI replaces part of the AI's strategy with a trained model. It picks a strategic posture (defensive buildup, industrial expansion, armored offensive, air power, naval power or total war economy) for the AI countries it commands, as often as you choose next. Your own country is never affected.\n\nThe model runs in jevai.exe next to the game; while it is not running, countries keep their last posture.",
        "simp_chinese": "JevAI 用训练好的模型接管部分 AI 战略。它为其指挥的 AI 国家选择一种战略姿态（防御建设、工业扩张、装甲攻势、空中力量、海军力量或全面战争经济），决策频率由你在下一步选择。你自己的国家不会受到影响。\n\n模型在游戏旁边的 jevai.exe 中运行；未运行时，各国保持上一次的姿态。",
        "japanese": "JevAI は AI の戦略の一部を学習済みモデルに置き換えます。指揮する AI 国家の戦略方針（防衛強化、工業拡張、機甲攻勢、航空戦力、海軍力、総力戦経済）を、次に選ぶ頻度で決めます。あなたの国は影響を受けません。\n\nモデルはゲームと並行して jevai.exe で動作します。動作していない間、各国は直前の方針を維持します。",
        "korean": "JevAI는 AI 전략의 일부를 학습된 모델로 대체합니다. 지휘하는 AI 국가의 전략 태세(방어 증강, 산업 확장, 기갑 공세, 공군력, 해군력, 총력전 경제)를 다음에 고르는 주기로 결정합니다. 플레이어의 국가는 영향을 받지 않습니다.\n\n모델은 게임 옆의 jevai.exe에서 실행되며, 실행 중이 아닐 때는 각 국가가 마지막 태세를 유지합니다.",
        "russian": "JevAI заменяет часть стратегии ИИ обученной моделью. Она выбирает стратегическую линию (оборонительное строительство, промышленное расширение, бронетанковое наступление, авиация, флот или экономика тотальной войны) для стран ИИ, которыми командует, так часто, как вы выберете далее. Ваша страна не затрагивается.\n\nМодель работает в jevai.exe рядом с игрой; пока она не запущена, страны сохраняют последнюю линию.",
        "german": "JevAI ersetzt einen Teil der KI-Strategie durch ein trainiertes Modell. Es wählt für die KI-Länder, die es führt, eine strategische Ausrichtung (defensiver Ausbau, industrielle Expansion, Panzeroffensive, Luftmacht, Seemacht oder totale Kriegswirtschaft), so oft, wie Sie als Nächstes festlegen. Ihr eigenes Land bleibt unberührt.\n\nDas Modell läuft in jevai.exe neben dem Spiel; solange es nicht läuft, behalten die Länder ihre letzte Ausrichtung.",
        "french": "JevAI remplace une partie de la stratégie de l'IA par un modèle entraîné. Il choisit une posture stratégique (renforcement défensif, expansion industrielle, offensive blindée, puissance aérienne, puissance navale ou économie de guerre totale) pour les pays IA qu'il dirige, à la fréquence que vous choisirez ensuite. Votre propre pays n'est jamais concerné.\n\nLe modèle tourne dans jevai.exe à côté du jeu ; tant qu'il ne tourne pas, les pays gardent leur dernière posture.",
        "spanish": "JevAI sustituye parte de la estrategia de la IA por un modelo entrenado. Elige una postura estratégica (refuerzo defensivo, expansión industrial, ofensiva acorazada, poder aéreo, poder naval o economía de guerra total) para los países de la IA que dirige, con la frecuencia que elijas a continuación. Tu propio país nunca se ve afectado.\n\nEl modelo funciona en jevai.exe junto al juego; mientras no esté en marcha, los países mantienen su última postura.",
        "braz_por": "O JevAI substitui parte da estratégia da IA por um modelo treinado. Ele escolhe uma postura estratégica (reforço defensivo, expansão industrial, ofensiva blindada, poder aéreo, poder naval ou economia de guerra total) para os países da IA que comanda, com a frequência que você escolher a seguir. Seu próprio país nunca é afetado.\n\nO modelo roda no jevai.exe ao lado do jogo; enquanto não estiver rodando, os países mantêm a última postura.",
        "polish": "JevAI zastępuje część strategii SI wytrenowanym modelem. Wybiera postawę strategiczną (rozbudowa obrony, ekspansja przemysłowa, ofensywa pancerna, lotnictwo, marynarka lub gospodarka wojny totalnej) dla krajów SI, którymi dowodzi, tak często, jak wybierzesz w następnym kroku. Twój kraj nigdy nie jest objęty.\n\nModel działa w jevai.exe obok gry; gdy nie działa, kraje zachowują ostatnią postawę.",
    },
    "jevai.1.a": {
        "english": "Major powers only",
        "simp_chinese": "仅主要国家",
        "japanese": "列強のみ",
        "korean": "주요 강대국만",
        "russian": "Только великие державы",
        "german": "Nur Großmächte",
        "french": "Grandes puissances seulement",
        "spanish": "Solo las grandes potencias",
        "braz_por": "Apenas grandes potências",
        "polish": "Tylko mocarstwa",
    },
    "jevai.1.b": {
        "english": "All AI countries",
        "simp_chinese": "所有 AI 国家",
        "japanese": "すべての AI 国家",
        "korean": "모든 AI 국가",
        "russian": "Все страны ИИ",
        "german": "Alle KI-Länder",
        "french": "Tous les pays IA",
        "spanish": "Todos los países de la IA",
        "braz_por": "Todos os países da IA",
        "polish": "Wszystkie kraje SI",
    },
    "jevai.1.c": {
        "english": "None (vanilla AI)",
        "simp_chinese": "不启用（原版 AI）",
        "japanese": "使用しない（標準 AI）",
        "korean": "사용 안 함 (기본 AI)",
        "russian": "Никем (обычный ИИ)",
        "german": "Keine (Standard-KI)",
        "french": "Aucun (IA standard)",
        "spanish": "Ninguno (IA estándar)",
        "braz_por": "Nenhum (IA padrão)",
        "polish": "Żadnym (standardowa SI)",
    },
    "jevai.2.t": {
        "english": "JevAI: How Often Should the Model Decide?",
        "simp_chinese": "JevAI：模型多久决策一次？",
        "japanese": "JevAI：モデルはどのくらいの頻度で判断しますか？",
        "korean": "JevAI: 모델이 얼마나 자주 결정할까요?",
        "russian": "JevAI: как часто модель будет принимать решения?",
        "german": "JevAI: Wie oft soll das Modell entscheiden?",
        "french": "JevAI : à quelle fréquence le modèle doit-il décider ?",
        "spanish": "JevAI: ¿con qué frecuencia debe decidir el modelo?",
        "braz_por": "JevAI: com que frequência o modelo deve decidir?",
        "polish": "JevAI: jak często model ma podejmować decyzje?",
    },
    "jevai.2.d": {
        "english": "Weekly decisions react fastest to the war but cost the most computing: about 1.5 seconds per country on an Intel NPU and about 10 seconds on a CPU. Monthly matches the games the model learned from. Every 3 months is the lightest, for slower computers.",
        "simp_chinese": "每周决策对战局反应最快，但计算量最大：在 Intel NPU 上每个国家约 1.5 秒，在 CPU 上约 10 秒。每月决策与模型学习时所用的对局一致。每 3 个月决策最省资源，适合较慢的电脑。",
        "japanese": "毎週は戦況に最も早く反応しますが、計算量が最も多くなります。1 か国あたり Intel NPU で約 1.5 秒、CPU で約 10 秒かかります。毎月はモデルが学習した対局と同じ頻度です。3 か月ごとは最も軽く、性能の低い PC 向けです。",
        "korean": "매주 결정은 전황에 가장 빠르게 반응하지만 계산량이 가장 많습니다. 국가당 Intel NPU에서 약 1.5초, CPU에서 약 10초가 걸립니다. 매월 결정은 모델이 학습한 게임과 같은 주기입니다. 3개월마다 결정하는 것이 가장 가벼워 느린 컴퓨터에 알맞습니다.",
        "russian": "Еженедельные решения быстрее всего реагируют на ход войны, но требуют больше всего вычислений: около 1,5 секунды на страну на Intel NPU и около 10 секунд на процессоре. Ежемесячные соответствуют партиям, на которых училась модель. Раз в 3 месяца — самый лёгкий вариант для медленных компьютеров.",
        "german": "Wöchentliche Entscheidungen reagieren am schnellsten auf den Kriegsverlauf, brauchen aber die meiste Rechenleistung: etwa 1,5 Sekunden pro Land auf einer Intel-NPU und etwa 10 Sekunden auf einer CPU. Monatlich entspricht den Partien, aus denen das Modell gelernt hat. Alle 3 Monate ist am sparsamsten und passt zu langsameren Rechnern.",
        "french": "Des décisions hebdomadaires réagissent le plus vite à la guerre, mais demandent le plus de calcul : environ 1,5 seconde par pays sur un NPU Intel et environ 10 secondes sur un processeur. Le rythme mensuel correspond aux parties dont le modèle a appris. Tous les 3 mois est le plus léger, adapté aux ordinateurs plus lents.",
        "spanish": "Las decisiones semanales reaccionan antes a la guerra, pero requieren más cálculo: unos 1,5 segundos por país en una NPU de Intel y unos 10 segundos en una CPU. La frecuencia mensual coincide con las partidas de las que aprendió el modelo. Cada 3 meses es la opción más ligera, para ordenadores más lentos.",
        "braz_por": "Decisões semanais reagem mais rápido à guerra, mas exigem mais processamento: cerca de 1,5 segundo por país em uma NPU Intel e cerca de 10 segundos em uma CPU. A frequência mensal corresponde às partidas com que o modelo aprendeu. A cada 3 meses é a opção mais leve, para computadores mais lentos.",
        "polish": "Decyzje co tydzień najszybciej reagują na przebieg wojny, ale wymagają najwięcej obliczeń: około 1,5 sekundy na kraj na Intel NPU i około 10 sekund na procesorze. Co miesiąc odpowiada rozgrywkom, na których uczył się model. Co 3 miesiące to najlżejsza opcja dla wolniejszych komputerów.",
    },
    "jevai.2.a": {
        "english": "Every week",
        "simp_chinese": "每周",
        "japanese": "毎週",
        "korean": "매주",
        "russian": "Каждую неделю",
        "german": "Jede Woche",
        "french": "Chaque semaine",
        "spanish": "Cada semana",
        "braz_por": "Toda semana",
        "polish": "Co tydzień",
    },
    "jevai.2.b": {
        "english": "Every month",
        "simp_chinese": "每月",
        "japanese": "毎月",
        "korean": "매월",
        "russian": "Каждый месяц",
        "german": "Jeden Monat",
        "french": "Chaque mois",
        "spanish": "Cada mes",
        "braz_por": "Todo mês",
        "polish": "Co miesiąc",
    },
    "jevai.2.c": {
        "english": "Every 3 months",
        "simp_chinese": "每 3 个月",
        "japanese": "3 か月ごと",
        "korean": "3개월마다",
        "russian": "Раз в 3 месяца",
        "german": "Alle 3 Monate",
        "french": "Tous les 3 mois",
        "spanish": "Cada 3 meses",
        "braz_por": "A cada 3 meses",
        "polish": "Co 3 miesiące",
    },
}
LANGUAGES = ["english", "simp_chinese", "japanese", "korean", "russian", "german", "french", "spanish", "braz_por", "polish"]


def main():
    for lang in LANGUAGES:
        d = os.path.join(MOD, "localisation", lang)
        os.makedirs(d, exist_ok=True)
        lines = [f"l_{lang}:\n"]
        for key, texts in TEXT.items():
            t = texts.get(lang, texts["english"]).replace('"', '\\"').replace("\n", "\\n")
            lines.append(f' {key}: "{t}"\n')
        with open(os.path.join(d, f"jevai_l_{lang}.yml"), "w", encoding="utf-8-sig", newline="\n") as f:
            f.writelines(lines)
        print("wrote", os.path.join(d, f"jevai_l_{lang}.yml"))


if __name__ == "__main__":
    main()
