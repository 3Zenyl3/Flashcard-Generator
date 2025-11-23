# test_flashcard_generator.py
import pytest
from pathlib import Path
import sys
import json
import csv

# добавляем корень проекта в sys.path, чтобы импорты работали
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.flashcard_generator import (
    split_sentences,
    normalize_text,
    lemmatize_sentence,
    extract_key_sentences_tfidf,
    extract_key_sentences_textrank,
    heuristic_question_from_sentence,
    generate_flashcards_from_text,
    export_json,
    export_csv,
    export_markdown,
    morph,  # pymorphy2.MorphAnalyzer или None
)

# --- Fixtures ---

@pytest.fixture
def sample_text():
    return """Python — это язык программирования. Он был создан в 1991 году.
Модули позволяют структурировать код. Функция print() выводит данные на экран."""

@pytest.fixture
def temp_path(tmp_path):
    return tmp_path

# --- Tests for utils ---

def test_split_sentences(sample_text):
    sents = split_sentences(sample_text)
    assert len(sents) >= 3
    assert "Python — это язык программирования." in sents

def test_normalize_text():
    raw = "  Привет   мир  "
    norm = normalize_text(raw)
    assert norm == "Привет мир"

def test_lemmatize_sentence():
    if morph is None:
        pytest.skip("pymorphy2 не установлен — пропускаем лемматизацию")
    sent = "Кошки гуляют по крышам"
    lem = lemmatize_sentence(sent)
    words = set(lem.split())
    assert "кошка" in words
    assert "гулять" in words

# --- Tests for key sentence extraction ---

def test_extract_key_sentences_tfidf(sample_text):
    key = extract_key_sentences_tfidf(sample_text, top_k=2)
    assert len(key) == 2
    assert all(isinstance(s[1], str) for s in key)

def test_extract_key_sentences_textrank(sample_text):
    key = extract_key_sentences_textrank(sample_text, top_k=2)
    assert len(key) == 2
    assert all(isinstance(s[1], str) for s in key)

# --- Tests for heuristic question generation ---

def test_heuristic_definition_question():
    sent = "Python — это язык программирования."
    q = heuristic_question_from_sentence(sent)
    assert q['question'].startswith("Что такое Python")
    assert q['answer'] == sent

def test_heuristic_date_question():
    sent = "Python был создан в 1991 году."
    q = heuristic_question_from_sentence(sent)
    assert "Когда" in q['question']
    assert q['answer'] == sent

# --- Tests for pipeline ---

def test_generate_flashcards_from_text(sample_text):
    cards = generate_flashcards_from_text(sample_text, top_k_sentences=3, mode='heuristic')
    assert isinstance(cards, list)
    assert len(cards) > 0
    assert 'question' in cards[0]
    assert 'answer' in cards[0]

# --- Tests for exporters ---

def test_export_json(temp_path, sample_text):
    cards = generate_flashcards_from_text(sample_text, top_k_sentences=2)
    out_file = temp_path / "cards.json"
    export_json(cards, out_file)
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding='utf-8'))
    assert len(data) == len(cards)

def test_export_csv(temp_path, sample_text):
    cards = generate_flashcards_from_text(sample_text, top_k_sentences=2)
    out_file = temp_path / "cards.csv"
    export_csv(cards, out_file)
    assert out_file.exists()
    with open(out_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == len(cards)

def test_export_markdown(temp_path, sample_text):
    cards = generate_flashcards_from_text(sample_text, top_k_sentences=2)
    out_file = temp_path / "cards.md"
    export_markdown(cards, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert "Вопрос" in content
    assert "Ответ" in content
