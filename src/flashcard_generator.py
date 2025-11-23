"""
Flashcard Generator — идеальный проект (один файл-стартер)
PEP8-совместимая версия
"""

from __future__ import annotations
import sys
import argparse
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

# --- optional imports with graceful degradation ---
try:
    from razdel import sentenize
except Exception:
    sentenize = None

try:
    import pymorphy2

    morph = pymorphy2.MorphAnalyzer()
except Exception:
    morph = None

try:
    from natasha import (
        Segmenter,
        NewsEmbedding,
        NewsMorphTagger,
        NewsSyntaxParser,
        NewsNERPredictor,
    )

    _NATASHA_AVAILABLE = True
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_predictor = NewsNERPredictor(emb)
except Exception:
    _NATASHA_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Streamlit импортируется только внутри функции, чтобы избежать F401 и F811
_STREAMLIT_AVAILABLE = False
try:
    import streamlit  # noqa: F401
    _STREAMLIT_AVAILABLE = True
except Exception:
    _STREAMLIT_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("flashcard")


# -------------------- Utilities -------


def read_text_file(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def read_pdf(path: Path) -> str:
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is required to read PDF files. Install it "
            "or provide txt/md input."
        )
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text_parts.append(p.extract_text() or "")
    return "\n".join(text_parts)


def split_sentences(text: str) -> List[str]:
    """Возвращает список предложений — русский-aware, при наличии razdel."""
    if sentenize is not None:
        return [s.text.strip() for s in sentenize(text)]
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def normalize_text(text: str) -> str:
    """Простая нормализация: убирает лишние пробелы."""
    return re.sub(r'\s+', ' ', text).strip()


def lemmatize_sentence(sentence: str) -> str:
    """Лемматизация через pymorphy2, если доступна."""
    if morph is None:
        return sentence
    tokens = re.findall(r"\w+", sentence.lower())
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return " ".join(lemmas)


# -------------------- Key sentence extraction --------------------


def extract_key_sentences_tfidf(
    text: str, top_k: int = 10
) -> List[Tuple[int, str]]:
    """Выбираем ключевые предложения с помощью TF-IDF."""
    sentences = split_sentences(text)
    if TfidfVectorizer is None:
        ranked = sorted(enumerate(sentences), key=lambda x: -len(x[1]))
        return ranked[:top_k]

    vectorizer = TfidfVectorizer(max_df=0.8, min_df=1, ngram_range=(1, 2))
    try:
        X = vectorizer.fit_transform(sentences)
    except Exception as e:
        logger.warning("TF-IDF fitting failed: %s", e)
        ranked = sorted(enumerate(sentences), key=lambda x: -len(x[1]))
        return ranked[:top_k]

    scores = X.sum(axis=1).A1
    ranked_idx = np.argsort(-scores)[:top_k]
    return [(int(i), sentences[int(i)]) for i in ranked_idx]


def extract_key_sentences_textrank(
    text: str, top_k: int = 10
) -> List[Tuple[int, str]]:
    """Простой TextRank-подобный метод."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    if TfidfVectorizer is None:
        ranked = sorted(enumerate(sentences), key=lambda x: -len(x[1]))
        return ranked[:top_k]
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=1)
    X = vectorizer.fit_transform(sentences)
    sim = cosine_similarity(X)
    scores = sim.sum(axis=1)
    ranked_idx = np.argsort(-scores)[:top_k]
    return [(int(i), sentences[int(i)]) for i in ranked_idx]


# -------------------- Question generation --------------------


def heuristic_question_from_sentence(
    sentence: str, lemma_window: int = 4
) -> Optional[Dict[str, str]]:
    s = sentence.strip()
    m = re.match(r"^(.+?)\s+[—-]\s+(.+)$", s)
    if m:
        term = m.group(1).strip()
        q = f"Что такое {term}?"
        return {"question": q, "answer": s}

    if re.search(r"\b\d{4}\b", s):
        q = f"Когда: {s[:60]}... ?"
        return {"question": q, "answer": s}

    if re.search(r"\b(является|называется|это|являюсь|являются)\b", s):
        parts = re.split(r"\b(является|называется|это)\b", s, maxsplit=1)
        if len(parts) >= 3:
            subject = parts[0].strip().strip(' ,;:')
            q = f"Что такое {subject}?"
            return {"question": q, "answer": s}

    tokens = re.findall(r"\w+", s)
    if not tokens:
        return None

    if morph is not None:
        for i, t in enumerate(tokens[:lemma_window]):
            p = morph.parse(t)[0]
            if 'NOUN' in p.tag or 'Name' in p.tag:
                q_tokens = tokens.copy()
                q_tokens[i] = 'Что'
                q = ' '.join(q_tokens) + '?'
                return {"question": q, "answer": s}
    else:
        q = 'Что такое ' + ' '.join(tokens[:3]) + ' ?'
        return {"question": q, "answer": s}

    return None


class TransformerQuestionGenerator:
    """Обёртка для Seq2Seq модели."""

    def __init__(self, model_name: str = 'cointegrated/rut5-base-question-generator'):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Transformers not available. Install transformers and torch."
            )
        logger.info("Loading transformer model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, sentence: str, max_length: int = 64) -> Optional[str]:
        input_text = sentence.strip()
        inputs = self.tokenizer.encode(input_text, return_tensors='pt', truncation=True)
        out = self.model.generate(inputs, max_length=max_length, num_beams=4)
        q = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return q


# -------------------- Exporters --------------------


def export_json(cards: List[Dict[str, str]], path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    logger.info('Saved JSON: %s', path)


def export_csv(cards: List[Dict[str, str]], path: Path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer'])
        for c in cards:
            writer.writerow([c.get('question', ''), c.get('answer', '')])
    logger.info('Saved CSV: %s', path)


def export_markdown(cards: List[Dict[str, str]], path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        for i, c in enumerate(cards, 1):
            f.write(f"### Карточка {i}\n\n")
            f.write(f"**Вопрос:** {c.get('question', '')}\n\n")
            f.write(f"**Ответ:** {c.get('answer', '')}\n\n---\n\n")
    logger.info('Saved Markdown: %s', path)


def split_text_to_lines(text: str, max_chars: int = 80) -> List[str]:
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_chars:
            lines.append(' '.join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        lines.append(' '.join(cur))
    return lines


def export_pdf_simple(cards: List[Dict[str, str]], path: Path):
    if not _REPORTLAB_AVAILABLE:
        raise RuntimeError('reportlab required for PDF export')
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 40
    for i, card in enumerate(cards, 1):
        q = card.get('question', '')
        a = card.get('answer', '')
        c.setFont('Helvetica-Bold', 12)
        c.drawString(40, y, f'Q{i}: {q[:100]}')
        y -= 18
        c.setFont('Helvetica', 11)
        for line in split_text_to_lines(a, max_chars=90):
            c.drawString(40, y, line)
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 40
    c.save()
    logger.info('Saved PDF: %s', path)


# -------------------- Pipeline --------------------


def generate_flashcards_from_text(
    text: str,
    top_k_sentences: int = 30,
    mode: str = 'heuristic',
    transformer_model: Optional[str] = None
) -> List[Dict[str, str]]:
    text = normalize_text(text)
    key_sentences = extract_key_sentences_textrank(text, top_k=top_k_sentences)
    if not key_sentences:
        key_sentences = extract_key_sentences_tfidf(text, top_k=top_k_sentences)

    cards: List[Dict[str, str]] = []
    transformer = None
    if mode == 'transformer' and _TRANSFORMERS_AVAILABLE:
        try:
            transformer = TransformerQuestionGenerator(model_name=transformer_model)
        except Exception as e:
            logger.warning('Transformer model could not be loaded: %s', e)
            transformer = None

    for _, sentence in key_sentences:
        q = None
        if transformer is not None:
            try:
                q = transformer.generate(sentence)
            except Exception as e:
                logger.warning(
                    'Transformer generation failed for sentence %s: %s',
                    sentence[:30], e
                )
                q = None

        if q is None:
            res = heuristic_question_from_sentence(sentence)
            if res:
                cards.append(res)
            else:
                short = ' '.join(sentence.split()[:6])
                q = f'Что означает: "{short}..." ?'
                cards.append({'question': q, 'answer': sentence})
        else:
            cards.append({'question': q, 'answer': sentence})

    # Убираем дубликаты по вопросу
    seen = set()
    unique = []
    for c in cards:
        key = c['question'].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


# -------------------- CLI & Streamlit --------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Flashcard Generator — создать карточки '
                    '(вопрос-ответ) из текста (русский)'
    )
    p.add_argument(
        '--input', '-i', required=False,
        help='Путь к файлу (txt/md/pdf) или папке с текстами'
    )
    p.add_argument(
        '--output', '-o', required=False,
        help='Путь к выходному файлу (json/csv/md/pdf). '
             'Без --output запускается Streamlit если передан --serve'
    )
    p.add_argument(
        '--mode', choices=['heuristic', 'transformer'],
        default='heuristic', help='Режим генерации вопросов'
    )
    p.add_argument(
        '--top', type=int, default=30,
        help='Сколько ключевых предложений брать'
    )
    p.add_argument(
        '--serve', action='store_true',
        help='Запустить Streamlit UI (если установлен)'
    )
    p.add_argument(
        '--transformer-model', default=None,
        help='Имя трансформер-модели для генерации вопросов (если поддерживается)'
    )
    return p


def run_cli(args: argparse.Namespace):
    input_path = args.input
    out = args.output

    if args.serve:
        if not _STREAMLIT_AVAILABLE:
            logger.error(
                'Streamlit не установлен. Установите streamlit '
                'или не используйте --serve'
            )
            return
        run_streamlit_app()
        return

    if not input_path:
        logger.error('Нужно передать --input FILE')
        return

    p = Path(input_path)
    texts = []
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in ('.txt', '.md'):
                texts.append(read_text_file(f))
            elif f.suffix.lower() == '.pdf':
                texts.append(read_pdf(f))
    else:
        if p.suffix.lower() in ('.txt', '.md'):
            texts = [read_text_file(p)]
        elif p.suffix.lower() == '.pdf':
            texts = [read_pdf(p)]
        else:
            logger.error('Unsupported input file type: %s', p.suffix)
            return

    full_text = '\n'.join(texts)
    cards = generate_flashcards_from_text(
        full_text,
        top_k_sentences=args.top,
        mode=args.mode,
        transformer_model=args.transformer_model
    )

    outpath = Path(out) if out else Path('../output/flashcards.json')
    suffix = outpath.suffix.lower()
    if suffix == '.json' or outpath.name == 'flashcards.json':
        export_json(cards, outpath)
    elif suffix == '.csv':
        export_csv(cards, outpath)
    elif suffix in ('.md', '.markdown'):
        export_markdown(cards, outpath)
    elif suffix == '.pdf':
        export_pdf_simple(cards, outpath)
    else:
        export_json(cards, outpath)

    logger.info('Generated %d flashcards', len(cards))


def run_streamlit_app():
    import streamlit as st
    st.title('Flashcard Generator (русский)')
    uploaded = st.file_uploader(
        'Загрузите .txt/.md/.pdf файл', type=['txt', 'md', 'pdf']
    )
    mode = st.selectbox('Режим генерации', ['heuristic', 'transformer'])
    top_k = st.slider('Максимум карточек', 5, 100, 30)
    if uploaded is not None:
        raw = uploaded.read()
        tpath = Path('tmp_upload')
        tpath.write_bytes(raw)
        if uploaded.name.endswith('.pdf'):
            txt = read_pdf(tpath)
        else:
            txt = read_text_file(tpath)
        with st.spinner('Генерируем...'):
            cards = generate_flashcards_from_text(
                txt, top_k_sentences=top_k, mode=mode
            )
        for i, c in enumerate(cards, 1):
            st.markdown(f"**Q{i}. {c['question']}**")
            with st.expander('Показать ответ'):
                st.write(c['answer'])


# -------------------- Main --------------------


if __name__ == '__main__':
    parser = build_argparser()
    ns, _ = parser.parse_known_args()
    if '--serve' in sys.argv:
        run_streamlit_app()
        sys.exit(0)
    run_cli(ns)