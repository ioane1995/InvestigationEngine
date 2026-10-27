from app.ingestion.chunking import FixedWindowChunker, StructureAwareChunker, get_chunker
from app.models import RawDocument
from app.text import hamming, identifiers, simhash, stem, tokenize


def test_tokenize_keeps_ids_whole_and_split():
    t = tokenize("Incident INC-0012 hit payments-api")
    assert "inc-0012" in t and "0012" in t and "payments-api" in t and "payment" in t


def test_stem_unifies_cause_forms():
    assert len({stem(w) for w in ["cause", "caused", "causes", "causing"]}) == 1


def test_identifiers():
    assert identifiers("see INC-0012 and chg-345, not ABC") == ["inc-0012", "chg-345"]


def test_simhash_near_duplicates_are_close():
    a = "Drain traffic from the service, restart the instances one by one and verify health checks before continuing."
    b = a.replace("one by one", "one by one,")
    c = "Completely different text about quarterly financial planning and budget allocation for next year."
    assert hamming(simhash(a), simhash(b)) <= 3 < hamming(simhash(a), simhash(c))


def _doc(text):
    return RawDocument("d", "Title", text, "d.md", "md")


def test_fixed_chunker_window_and_overlap():
    words = " ".join(f"w{i}" for i in range(100))
    ch = FixedWindowChunker(max_tokens=40, overlap=10).split(_doc(words))
    assert [c.n_tokens for c in ch][:2] == [40, 40]
    assert ch[0].text.split()[-10:] == ch[1].text.split()[:10]
    assert ch[-1].text.split()[-1] == "w99"


def test_fixed_chunker_rejects_bad_overlap():
    import pytest
    with pytest.raises(ValueError):
        FixedWindowChunker(10, 10)


def test_structure_chunker_keeps_headings_and_respects_budget():
    text = "# Title\n\n## Alpha\n" + " ".join(["Sentence number one is here."] * 60) + "\n\n## Beta\nShort beta section about beta."
    ch = StructureAwareChunker(max_tokens=50, overlap=10).split(_doc(text))
    assert all(c.n_tokens <= 70 for c in ch)
    assert any(c.heading.endswith("Alpha") for c in ch) and any(c.heading.endswith("Beta") for c in ch)
    assert [c.index for c in ch] == list(range(len(ch)))
    assert len({c.chunk_id for c in ch}) == len(ch)


def test_get_chunker_unknown():
    import pytest
    with pytest.raises(ValueError):
        get_chunker("nope")
