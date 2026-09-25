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
        "english": "JevAI replaces part of the AI's strategy with a trained model. Every few months it picks a strategic posture (defensive buildup, industrial expansion, armored offensive, air power, naval power or total war economy) for the AI countries it commands. Your own country is never affected.\n\nThe model runs in jevai.exe next to the game; while it is not running, countries keep their last posture.",
        "simp_chinese": "JevAI 用训练好的模型接管部分 AI 战略。每隔几个月，它会为其指挥的 AI 国家选择一种战略姿态（防御建设、工业扩张、装甲攻势、空中力量、海军力量或全面战争经济）。你自己的国家不会受到影响。\n\n模型在游戏旁边的 jevai.exe 中运行；未运行时，各国保持上一次的姿态。",
        "japanese": "JevAI は AI の戦略の一部を学習済みモデルに置き換えます。数か月ごとに、指揮する AI 国家の戦略方針（防衛強化、工業拡張、機甲攻勢、航空戦力、海軍力、総力戦経済）を選びます。あなたの国は影響を受けません。\n\nモデルはゲームと並行して jevai.exe で動作します。動作していない間、各国は直前の方針を維持します。",
        "korean": "JevAI는 AI 전략의 일부를 학습된 모델로 대체합니다. 몇 달마다 지휘하는 AI 국가의 전략 태세(방어 증강, 산업 확장, 기갑 공세, 공군력, 해군력, 총력전 경제)를 선택합니다. 플레이어의 국가는 영향을 받지 않습니다.\n\n모델은 게임 옆의 jevai.exe에서 실행되며, 실행 중이 아닐 때는 각 국가가 마지막 태세를 유지합니다.",
        "russian": "JevAI заменяет часть стратегии ИИ обученной моделью. Раз в несколько месяцев она выбирает стратегическую линию (оборонительное строительство, промышленное расширение, бронетанковое наступление, авиация, флот или экономика тотальной войны) для стран ИИ, которыми командует. Ваша страна не затрагивается.\n\nМодель работает в jevai.exe рядом с игрой; пока она не запущена, страны сохраняют последнюю линию.",
        "german": "JevAI ersetzt einen Teil der KI-Strategie durch ein trainiertes Modell. Alle paar Monate wählt es für die KI-Länder, die es führt, eine strategische Ausrichtung (defensiver Ausbau, industrielle Expansion, Panzeroffensive, Luftmacht, Seemacht oder totale Kriegswirtschaft). Ihr eigenes Land bleibt unberührt.\n\nDas Modell läuft in jevai.exe neben dem Spiel; solange es nicht läuft, behalten die Länder ihre letzte Ausrichtung.",
        "french": "JevAI remplace une partie de la stratégie de l'IA par un modèle entraîné. Tous les quelques mois, il choisit une posture stratégique (renforcement défensif, expansion industrielle, offensive blindée, puissance aérienne, puissance navale ou économie de guerre totale) pour les pays IA qu'il dirige. Votre propre pays n'est jamais concerné.\n\nLe modèle tourne dans jevai.exe à côté du jeu ; tant qu'il ne tourne pas, les pays gardent leur dernière posture.",
        "spanish": "JevAI sustituye parte de la estrategia de la IA por un modelo entrenado. Cada pocos meses elige una postura estratégica (refuerzo defensivo, expansión industrial, ofensiva acorazada, poder aéreo, poder naval o economía de guerra total) para los países de la IA que dirige. Tu propio país nunca se ve afectado.\n\nEl modelo funciona en jevai.exe junto al juego; mientras no esté en marcha, los países mantienen su última postura.",
        "braz_por": "O JevAI substitui parte da estratégia da IA por um modelo treinado. A cada poucos meses, ele escolhe uma postura estratégica (reforço defensivo, expansão industrial, ofensiva blindada, poder aéreo, poder naval ou economia de guerra total) para os países da IA que comanda. Seu próprio país nunca é afetado.\n\nO modelo roda no jevai.exe ao lado do jogo; enquanto não estiver rodando, os países mantêm a última postura.",
        "polish": "JevAI zastępuje część strategii SI wytrenowanym modelem. Co kilka miesięcy wybiera postawę strategiczną (rozbudowa obrony, ekspansja przemysłowa, ofensywa pancerna, lotnictwo, marynarka lub gospodarka wojny totalnej) dla krajów SI, którymi dowodzi. Twój kraj nigdy nie jest objęty.\n\nModel działa w jevai.exe obok gry; gdy nie działa, kraje zachowują ostatnią postawę.",
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
