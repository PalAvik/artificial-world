"""CPU tests for the corpus builders.

The failure modes worth guarding here all produce a *plausible number* rather
than an exception: a control identical to its span sends the MSG denominator to
zero, a control rendered in the same font does the same, and a probe label that
encodes surface rather than content turns the anti-collapse check into a font
classifier. None of those raise on their own.

    python -m pytest tests/test_data.py -q
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from freeflow.data import tier_a, tier_b, tier_c, views, vocab
from freeflow.data.render import FontSet, RenderConfig, render_pair, render_span

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg() -> RenderConfig:
    """The frozen Gate 0 geometry if present, else its recorded values."""
    try:
        return RenderConfig.load()
    except FileNotFoundError:
        return RenderConfig(height=48, pad=4, min_pixels=1024,
                            visual_tokens_per_span=6)


@pytest.fixture(scope="module")
def fonts() -> FontSet:
    try:
        fs = FontSet.load()
        if len(fs.train) >= 2:
            return fs
    except FileNotFoundError:
        pass
    import matplotlib
    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    return FontSet(train={"a": str(ttf / "DejaVuSans.ttf"),
                          "b": str(ttf / "DejaVuSerif.ttf"),
                          "c": str(ttf / "DejaVuSansMono.ttf")},
                   held_out={"h1": str(ttf / "STIXGeneral.ttf"),
                             "h2": str(ttf / "cmr10.ttf")})


# ------------------------------------------------------------ the control ---

class TestSurfaceVariant:
    @pytest.mark.parametrize("word", ["the", "from", "dog", "perspicacity",
                                      "left of", "A", "iPhone"])
    def test_never_returns_the_span_unchanged(self, word):
        """An identical control collapses the MSG denominator to zero."""
        assert views.surface_variant(word) != word

    def test_handles_uncased_spans(self):
        for span in ["123", "!?", "  "]:
            assert views.surface_variant(span) != span

    def test_is_deterministic(self):
        assert views.surface_variant("harbour") == views.surface_variant("harbour")


class TestValidate:
    def _item(self, **kw):
        base = dict(prefix="a ", span_text="dog", suffix=" b.",
                    span_image=object(), span_paraphrase="DOG",
                    span_image_alt=object(), span_id="dog")
        base.update(kw)
        return views.SpanItem(**base)

    def test_rejects_an_empty_suffix(self):
        with pytest.raises(ValueError, match="merge position"):
            views.validate([self._item(suffix="")])

    def test_rejects_a_control_identical_to_the_span(self):
        with pytest.raises(ValueError, match="denominator"):
            views.validate([self._item(span_paraphrase="dog")])

    def test_rejects_a_shared_image_object(self):
        img = object()
        with pytest.raises(ValueError, match="exactly zero"):
            views.validate([self._item(span_image=img, span_image_alt=img)])

    def test_rejects_an_empty_corpus(self):
        with pytest.raises(ValueError, match="empty corpus"):
            views.validate([])


class TestBatchBySuffix:
    def test_every_batch_shares_one_suffix(self):
        # Built directly rather than via a tier: only suffix grouping is tested.
        made = [views.SpanItem(prefix="p", span_text=f"w{i}", suffix=s,
                               span_image=object(), span_paraphrase="X",
                               span_image_alt=object(), span_id=f"w{i}")
                for i, s in enumerate([" a.", " bb.", " a.", " ccc."] * 5)]
        for batch in views.batch_by_suffix(made, 3):
            assert len({it.suffix for it in batch}) == 1
        assert sum(len(b) for b in views.batch_by_suffix(made, 3)) == len(made)


# ---------------------------------------------------------------- vocab ---

class TestVocab:
    def test_the_matched_sets_share_a_length_distribution_exactly(self):
        """Gate 0 showed read-back tracks length, so a class difference in MSG
        could otherwise just be length. These are the sets any word-class claim
        must rest on, and they are matched histogram-for-histogram, not merely
        in mean."""
        from collections import Counter
        hists = {n: Counter(len(w) for w in ws)
                 for n, ws in vocab.MATCHED.items()}
        assert set(hists) == {"function", "concrete", "abstract"}
        first = hists["function"]
        for name, h in hists.items():
            assert h == first, (name, h, first)
        assert len(vocab.MATCHED["function"]) > 100

    def test_concrete_and_abstract_are_length_matched_at_scale(self):
        """The larger pair carries the abstractness contrast and is matched to
        each other; function cannot join them without costing most of their
        size, which is what MATCHED is for."""
        lens = vocab.length_summary()
        assert lens["concrete"]["mean"] == lens["abstract"]["mean"]
        assert lens["concrete"]["sd"] == lens["abstract"]["sd"]
        assert lens["concrete"]["n"] == lens["abstract"]["n"] > 1000

    def test_rare_long_is_the_long_class(self):
        lens = vocab.length_summary()
        assert lens["rare_long"]["mean"] > lens["concrete"]["mean"] + 3.0

    def test_the_span_pool_can_support_the_map_nulls(self):
        """A [D, D] map memorises when D >= distinct spans. At D = 2048 the pool
        must clear ~4096 for the linear-map null to conclude anything; this is
        the corpus half of the 2026-08-31 failure."""
        assert len(vocab.SPANS) >= 2 * 2048
        assert len(set(vocab.SPANS)) == len(vocab.SPANS)

    def test_unbalanced_sampling_wastes_no_distinct_spans(self):
        picks = vocab.sample(8000, random.Random(0), balanced=False)
        assert len({w for w, _ in picks}) == 8000

    def test_words_carry_their_class_and_bulk_claims_nothing(self):
        assert vocab.class_of(vocab.CLASSES["function"][0]) == "function"
        assert vocab.class_of("thisisnotaword") == "bulk"

    def test_the_pool_is_not_an_alphabetical_slice(self):
        """The labelled classes exceed the pool size, so something is dropped.
        Truncating a sorted list would keep a-m and cut the tail off every
        class: a spelling bias and an uneven cut across classes at once."""
        from collections import Counter
        initials = Counter(w[0] for w in vocab.SPANS)
        assert len(initials) >= 20                      # not a-m only
        shares = Counter(vocab.class_of(w) for w in vocab.SPANS)
        for name, words in vocab.CLASSES.items():
            if len(words) >= 2000:                      # the large classes
                assert shares[name] >= 1500, shares

    def test_the_gate0_words_are_held_out(self):
        """Gate 0 chose the render config on those words; reusing them would
        report a number partly selected for."""
        assert "quixotic" not in vocab.SPANS and "pulchritude" not in vocab.SPANS

    def test_sampling_is_balanced_across_classes(self):
        from collections import Counter
        picks = vocab.sample(80, random.Random(0))
        counts = Counter(c for _, c in picks)
        assert set(counts) == set(vocab.CLASSES)
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_sampling_is_deterministic_under_seed(self):
        assert vocab.sample(20, random.Random(3)) == vocab.sample(20, random.Random(3))

    def test_can_exceed_the_vocabulary_size(self):
        assert len(vocab.sample(500, random.Random(0))) == 500


# -------------------------------------------------------------- rendering ---

class TestRenderPair:
    def test_the_two_fonts_always_differ(self, fonts, cfg):
        """Same font would make the image half of the denominator exactly zero."""
        rng = random.Random(0)
        for _ in range(25):
            _, _, a, b = render_pair("harbour", fonts, cfg, rng)
            assert a != b

    def test_the_two_renders_are_not_pixel_identical(self, fonts, cfg):
        img, alt, _, _ = render_pair("harbour", fonts, cfg, random.Random(1))
        assert img.tobytes() != alt.tobytes()

    def test_uses_the_frozen_height(self, fonts, cfg):
        img = render_span("dog", next(iter(fonts.train.values())), cfg)
        assert img.size[1] == cfg.height

    def test_refuses_a_pool_too_small_to_form_a_control(self, cfg):
        one = FontSet(train={"only": "x"}, held_out={})
        with pytest.raises(ValueError, match=">=2 fonts"):
            render_pair("dog", one, cfg, random.Random(0))

    def test_held_out_pool_is_separate(self, fonts, cfg):
        rng = random.Random(0)
        _, _, a, b = render_pair("dog", fonts, cfg, rng, held_out=True)
        assert a in fonts.held_out and b in fonts.held_out


# ---------------------------------------------------------------- tier B ---

class TestTierB:
    def test_builds_valid_items_with_distinct_controls(self, fonts, cfg):
        items = tier_b.build(24, fonts, cfg, seed=0)
        assert len(items) == 24
        for it in items:
            assert it.span_paraphrase != it.span_text
            assert it.span_image.tobytes() != it.span_image_alt.tobytes()
            assert it.meta["font"] != it.meta["font_control"]

    def test_probe_label_is_content_not_surface(self, fonts, cfg):
        """Two items showing the same word in different fonts must share a
        label, or the probe becomes a font classifier."""
        items = tier_b.build(200, fonts, cfg, seed=0)
        by_word = {}
        for it in items:
            by_word.setdefault(it.span_text.lower(), set()).add(it.span_id)
        assert all(len(ids) == 1 for ids in by_word.values())

    def test_covers_every_word_class(self, fonts, cfg):
        items = tier_b.build(40, fonts, cfg, seed=0)
        assert {it.group for it in items} == set(vocab.CLASSES)

    def test_held_out_fonts_are_marked_and_used(self, fonts, cfg):
        items = tier_b.build(8, fonts, cfg, seed=0, held_out_fonts=True)
        assert all(it.meta["held_out_fonts"] for it in items)
        assert all(it.meta["font"] in fonts.held_out for it in items)

    def test_synonym_control_falls_back_rather_than_returning_the_span(
            self, fonts, cfg):
        items = tier_b.build(40, fonts, cfg, seed=0,
                             control=views.ControlKind.SYNONYM)
        for it in items:
            assert it.span_paraphrase != it.span_text

    def test_is_deterministic_under_seed(self, fonts, cfg):
        a = tier_b.build(12, fonts, cfg, seed=7)
        b = tier_b.build(12, fonts, cfg, seed=7)
        assert [x.span_text for x in a] == [x.span_text for x in b]
        assert [x.meta["font"] for x in a] == [x.meta["font"] for x in b]

    def test_summary_reports_composition(self, fonts, cfg):
        s = tier_b.summary(tier_b.build(40, fonts, cfg, seed=0))
        assert s["n"] == 40 and s["tier"] == "B"
        assert set(s["groups"]) == set(vocab.CLASSES)


# ---------------------------------------------------------------- tier C ---

class TestTierC:
    def test_builds_valid_items_across_relations(self):
        items = tier_c.build(20, seed=0)
        assert {it.group for it in items} == {r.name for r in tier_c.RELATIONS}
        for it in items:
            assert it.span_image.tobytes() != it.span_image_alt.tobytes()

    def test_every_relation_renders_distinguishably_in_every_style(self):
        """The bug this catches: with both markers drawn alike, "above" and
        "below" are the same picture and the image carries no relation."""
        for style in tier_c.STYLES:
            seen = {}
            for rel in tier_c.RELATIONS:
                key = tier_c.draw_relation(rel, style).tobytes()
                assert key not in seen, (
                    f"{rel.name} renders identically to {seen.get(key)} "
                    f"in style {style!r}")
                seen[key] = rel.name

    def test_the_style_control_preserves_the_relation(self):
        """Changing style must change the picture but not which relation it is:
        a relation drawn in one style must not collide with a *different*
        relation drawn in the other."""
        rendered = {(rel.name, st): tier_c.draw_relation(rel, st).tobytes()
                    for rel in tier_c.RELATIONS for st in tier_c.STYLES}
        for (name_a, st_a), img_a in rendered.items():
            for (name_b, st_b), img_b in rendered.items():
                if name_a != name_b:
                    assert img_a != img_b, f"{name_a}/{st_a} == {name_b}/{st_b}"
                elif st_a != st_b:
                    assert img_a != img_b, "style control changed nothing"

    def test_rejects_a_control_kind_it_cannot_honour(self):
        with pytest.raises(ValueError, match="only ControlKind.SURFACE"):
            tier_c.build(4, control=views.ControlKind.SYNONYM)

    def test_the_control_keeps_the_relation_and_changes_only_the_style(self):
        """The image control must vary the drawing, not the proposition."""
        for it in tier_c.build(15, seed=0):
            assert it.meta["style"] != it.meta["style_control"]

    def test_diagrams_are_not_blank(self):
        for rel in tier_c.RELATIONS:
            img = tier_c.draw_relation(rel)
            assert len(img.getcolors(65536)) > 1, rel.name

    def test_is_deterministic_under_seed(self):
        a, b = tier_c.build(10, seed=2), tier_c.build(10, seed=2)
        assert [x.meta["style"] for x in a] == [x.meta["style"] for x in b]


# ---------------------------------------------------------------- tier A ---

class TestFlickrParsing:
    def test_extracts_entity_phrases_from_markup(self):
        line = ("[/EN#283585/people A man] in a [/EN#283586/clothing red shirt] "
                "is riding")
        # Only the bracketed text is the phrase — the "a" before the second
        # entity sits outside the markup and is not part of it.
        assert tier_a.parse_flickr_sentence(line) == [
            ("283585", "a man"), ("283586", "red shirt")]

    def test_ignores_unmarked_text(self):
        assert tier_a.parse_flickr_sentence("just a plain sentence") == []

    def test_drops_phrases_longer_than_the_cap(self):
        long = "[/EN#1/x one two three four five six]"
        assert tier_a.parse_flickr_sentence(long) == []

    def test_parses_boxes_and_skips_nobox_entities(self, tmp_path):
        xml = tmp_path / "1.xml"
        xml.write_text("""<annotation>
          <object><name>100</name>
            <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>60</xmax><ymax>70</ymax></bndbox>
          </object>
          <object><name>200</name><nobox>1</nobox></object>
        </annotation>""")
        boxes = tier_a.parse_flickr_boxes(xml)
        assert boxes == {"100": [(1, 2, 60, 70)]}


class TestTierAAssembly:
    def _regions(self, tmp_path):
        from PIL import Image
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.jpg"
            Image.new("RGB", (200, 200), (i * 60, 100, 200)).save(p)
            paths.append(p)
        # "a dog" appears in three images; "a rare thing" in only one.
        return [
            tier_a.Region("a dog", paths[0], (0, 0, 100, 100)),
            tier_a.Region("a dog", paths[1], (50, 50, 150, 150)),
            tier_a.Region("a dog", paths[2], (10, 10, 110, 110)),
            tier_a.Region("a rare thing", paths[0], (0, 0, 80, 80)),
        ]

    def test_phrases_with_one_instance_are_excluded(self, tmp_path):
        grouped = tier_a.group_by_phrase(self._regions(tmp_path))
        assert set(grouped) == {"a dog"}

    def test_control_prefers_a_different_image(self, tmp_path):
        items = tier_a.build(self._regions(tmp_path), n=12, seed=0)
        assert items and all(it.meta["cross_image"] for it in items)

    def test_builds_valid_items(self, tmp_path):
        items = tier_a.build(self._regions(tmp_path), n=6, seed=0)
        for it in items:
            assert it.tier == "A"
            assert it.span_paraphrase != it.span_text
            assert it.span_image.size[0] > 0

    def test_raises_when_no_phrase_has_a_second_instance(self, tmp_path):
        from PIL import Image
        p = tmp_path / "solo.jpg"
        Image.new("RGB", (100, 100)).save(p)
        with pytest.raises(ValueError, match="second instance"):
            tier_a.build([tier_a.Region("lonely", p, (0, 0, 50, 50))], n=2)

    def test_visual_genome_loader_reads_the_documented_format(self, tmp_path):
        from PIL import Image
        (tmp_path / "images").mkdir()
        Image.new("RGB", (300, 300)).save(tmp_path / "images" / "7.jpg")
        (tmp_path / "regions.json").write_text(json.dumps([{
            "id": 7,
            "regions": [
                {"phrase": "A DOG  ", "x": 0, "y": 0, "width": 90,
                 "height": 90, "image_id": 7},
                {"phrase": "too small", "x": 0, "y": 0, "width": 8,
                 "height": 8, "image_id": 7},
            ]}]))
        regions = tier_a.load_visual_genome(tmp_path / "regions.json",
                                            tmp_path / "images")
        assert len(regions) == 1
        assert regions[0].phrase == "a dog"      # cleaned and lowercased
